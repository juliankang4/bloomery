//! Host computation behind the expert tier's existing stream protocol.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cuda_core::{CudaContext, CudaStream, sys};
use model::ops::Tensor2;

use super::HostExperts;
use super::run::HostRun;
use super::tier::{TierBlock, TierExperts, TierInput, TierIo, TierSet, TierShape, copy_page};
use crate::graph::MappedHost;
use crate::{Gpu, GpuError};

const WHAT: &str = "host expert tier";

struct State {
    run: HostRun,
    set: TierSet,
    page: MappedHost,
    x: Tensor2,
    output: Vec<f32>,
    hidden: usize,
    act: super::tier::TierAct,
    q3_words: usize,
    scale_at: usize,
    used: usize,
    sel_at: usize,
    out_at: usize,
    failure: Option<String>,
}

impl State {
    fn serve(&mut self, layer: usize, cols: usize) -> Result<(), GpuError> {
        let ids = self
            .set
            .ids(layer)
            .ok_or_else(|| GpuError::shape(WHAT, format!("layer {layer} is outside the tier")))?;
        let len = cols * self.used * self.hidden;
        // SAFETY: the preceding stream copy initialized this output prefix.
        // The callback owns it until the stream's following read.
        let previous = unsafe {
            std::slice::from_raw_parts(self.page.host_at(self.out_at).cast::<f32>(), len)
        };
        self.output[..len].copy_from_slice(previous);
        for col in 0..cols {
            // SAFETY: sel_at follows the complete activation span. The
            // preceding stream copies completed before this host callback.
            let input = unsafe { std::slice::from_raw_parts(self.page.host_at(0), self.sel_at) };
            for (i, value) in self.x.data.iter_mut().enumerate() {
                *value = match self.act {
                    super::tier::TierAct::F32 => {
                        let at = 4 * (col * self.hidden + i);
                        f32::from_ne_bytes(input[at..at + 4].try_into().unwrap())
                    }
                    super::tier::TierAct::Q8 => {
                        let v4 = u32::try_from(i / 4).expect("Q8Act bounds its width");
                        let word =
                            2 * crate::cores::q3_slot(v4) as usize + ((v4 >> 3) & 1) as usize;
                        let code = input[4 * (col * self.q3_words + word) + i % 4] as i8;
                        let at = self.scale_at + 4 * (col * (self.hidden / 128) + i / 128);
                        let scale = f32::from_ne_bytes(input[at..at + 4].try_into().unwrap());
                        f32::from(code) * scale
                    }
                };
            }
            let mut columns = 1;
            for slot in 0..self.used {
                let index = col * self.used + slot;
                // SAFETY: the page holds cols * used selection words at
                // sel_at. No later stream operation runs during the callback.
                let selected = unsafe {
                    self.page
                        .host_at(self.sel_at + 4 * index)
                        .cast::<u32>()
                        .read()
                };
                if selected == super::slots::HOST {
                    continue;
                }
                let id = *ids.get(selected as usize).ok_or_else(|| {
                    GpuError::shape(
                        WHAT,
                        format!("layer {layer}: tier slot {selected} is outside its set"),
                    )
                })?;
                let out = &mut self.output[index * self.hidden..(index + 1) * self.hidden];
                self.run
                    .experts_into(layer, &self.x, &[(id, 1.0)], out, columns)?;
                columns = 0;
            }
            if columns != 0
                && let Some(tier) = self.run.tier()
            {
                drop(
                    model::moe::TierSlots::prefetch_pinned(
                        tier.as_ref(),
                        layer,
                        &[],
                        model::moe::Claim::Step,
                        columns,
                        None,
                    )
                    .and_then(|ticket| ticket.wait())
                    .map_err(model::ModelError::from)?,
                );
            }
        }
        Ok(())
    }
}

struct Call {
    state: Arc<Mutex<State>>,
    layer: usize,
    cols: usize,
}

unsafe extern "C" fn serve(data: *mut c_void) {
    // SAFETY: CpuExperts holds its fixed Call table until its stream and
    // every graph referring to it have drained. The table never reallocates.
    let call = unsafe { &*data.cast::<Call>() };
    let mut state = match call.state.lock() {
        Ok(state) => state,
        Err(poisoned) => {
            let mut state = poisoned.into_inner();
            state.failure = Some("the host expert state was poisoned".into());
            state
        }
    };
    if state.failure.is_none() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.serve(call.layer, call.cols)
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => state.failure = Some(e.to_string()),
            Err(_) => state.failure = Some("the host expert service panicked".into()),
        }
    }
    if state.failure.is_some() {
        state.output.fill(f32::NAN);
    }
    // SAFETY: out_at begins the page's output span, large enough for output.
    // The stream cannot read it until this callback returns. The source Vec
    // is disjoint from the mapped page.
    unsafe {
        std::ptr::copy_nonoverlapping(
            state.output.as_ptr(),
            state.page.host_at(state.out_at).cast::<f32>(),
            state.output.len(),
        );
    }
}

pub(super) struct CpuExperts {
    calls: Vec<Call>,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    state: Arc<Mutex<State>>,
    cols: usize,
    shape: TierShape,
}

impl CpuExperts {
    pub(super) fn new(
        gpu: &Gpu,
        mut run: HostRun,
        tier: Option<Arc<super::nvtier::NvTier>>,
        set: TierSet,
        shape: TierShape,
        cols: usize,
    ) -> Result<CpuExperts, GpuError> {
        if (shape.act == super::tier::TierAct::Q8 && !shape.hidden.is_multiple_of(256))
            || cols == 0
            || shape.hidden == 0
            || shape.n_used == 0
            || shape.rows == 0
        {
            return Err(GpuError::shape(
                WHAT,
                "the host tier requires nonempty columns and whole Q8 super-blocks",
            ));
        }
        let q3_words = 128 * (shape.hidden / 256).div_ceil(2);
        let input_words = match shape.act {
            super::tier::TierAct::F32 => shape.hidden,
            super::tier::TierAct::Q8 => q3_words + shape.hidden / 128,
        };
        let x = cols.checked_mul(input_words);
        let sel = cols.checked_mul(shape.n_used);
        let out = sel.and_then(|s| s.checked_mul(shape.hidden));
        let bytes = x
            .zip(sel)
            .zip(out)
            .and_then(|((x, sel), out)| x.checked_add(sel)?.checked_add(out)?.checked_mul(4))
            .ok_or_else(|| GpuError::shape(WHAT, "the host tier's staging bytes overflow"))?;
        let ((x, sel), out) = x
            .zip(sel)
            .zip(out)
            .expect("byte check proved all sizes fit");
        run.attach_tier(tier)?;
        run.prepare_union(1)?;
        let state = Arc::new(Mutex::new(State {
            run,
            set: set.clone(),
            page: MappedHost::new(gpu.context(), bytes, WHAT)?,
            x: Tensor2 {
                ne0: shape.hidden,
                ne1: 1,
                data: vec![0.0; shape.hidden],
            },
            output: vec![0.0; out],
            hidden: shape.hidden,
            act: shape.act,
            q3_words,
            scale_at: 4 * cols * q3_words,
            used: shape.n_used,
            sel_at: 4 * x,
            out_at: 4 * (x + sel),
            failure: None,
        }));
        // CUDA invokes the host service on its callback thread.
        let _: &(dyn Send + Sync) = &state;
        let calls = set
            .layers()
            .flat_map(|layer| {
                let state = &state;
                (1..=cols).map(move |cols| Call {
                    state: Arc::clone(state),
                    layer,
                    cols,
                })
            })
            .collect();
        Ok(CpuExperts {
            calls,
            ctx: Arc::clone(gpu.context()),
            stream: gpu.stream_handle(),
            state,
            cols,
            shape,
        })
    }
}

impl Drop for CpuExperts {
    fn drop(&mut self) {
        let settled = self
            .ctx
            .bind_to_thread()
            .map_err(GpuError::from)
            .and_then(|()| super::step::stream_idle(&self.stream));
        let why = match settled {
            Ok(true) => return,
            Ok(false) => "work is still queued".to_string(),
            Err(e) => e.to_string(),
        };
        eprintln!("{WHAT}: retaining callback storage for an unsettled stream: {why}");
        std::mem::forget(std::mem::take(&mut self.calls));
    }
}

impl TierExperts for CpuExperts {
    fn host_fills(&self) -> Result<(u64, u64), GpuError> {
        let state = self
            .state
            .lock()
            .map_err(|_| GpuError::shape(WHAT, "the host expert state was poisoned"))?;
        Ok(match state.run.tier() {
            Some(tier) => {
                let stats = tier.stats();
                (stats.fills, stats.fill_bytes)
            }
            None => (0, 0),
        })
    }

    fn enqueue_layer(
        &mut self,
        gpu: &Gpu,
        _: &crate::weights::Weights,
        layer: usize,
        io: TierIo<'_>,
    ) -> Result<(), GpuError> {
        if !Arc::ptr_eq(&self.ctx, gpu.context())
            || self.stream.cu_stream() != gpu.stream().cu_stream()
        {
            return Err(GpuError::shape(
                WHAT,
                "the host expert tier belongs to another stream",
            ));
        }
        let state = self
            .state
            .lock()
            .map_err(|_| GpuError::shape(WHAT, "the host expert state was poisoned"))?;
        let n = io
            .cols
            .checked_mul(self.shape.n_used)
            .ok_or_else(|| GpuError::shape(WHAT, "selection size overflow"))?;
        let input_ok = match io.act {
            TierInput::F32(x) if self.shape.act == super::tier::TierAct::F32 => {
                x.len() >= io.cols * self.shape.hidden
            }
            TierInput::Q8(x) if self.shape.act == super::tier::TierAct::Q8 => {
                x.k() == self.shape.hidden && x.m() >= io.cols
            }
            _ => {
                return Err(GpuError::shape(
                    WHAT,
                    "the activation format differs from the host tier",
                ));
            }
        };
        if !(1..=self.cols).contains(&io.cols)
            || !input_ok
            || io.sel.len() < n
            || io.rows.len() < n * self.shape.hidden
        {
            return Err(GpuError::shape(WHAT, "short host tier buffers"));
        }
        let at = layer
            .checked_sub(state.set.layers().start)
            .and_then(|l| l.checked_mul(self.cols))
            .and_then(|l| l.checked_add(io.cols - 1))
            .filter(|&i| i < self.calls.len())
            .ok_or_else(|| GpuError::shape(WHAT, "a layer outside the host tier"))?;
        let (input, scales, selected, out) = (
            state.page.dev_at(0),
            state.page.dev_at(state.scale_at),
            state.page.dev_at(state.sel_at),
            state.page.dev_at(state.out_at),
        );
        drop(state);
        let stream = gpu.stream();
        match io.act {
            TierInput::F32(x) => copy_page(
                stream,
                input,
                x.cu_deviceptr(),
                4 * io.cols * self.shape.hidden,
            )?,
            TierInput::Q8(x) => {
                copy_page(
                    stream,
                    input,
                    x.q3.cu_deviceptr(),
                    x.q3.num_bytes() / x.m() * io.cols,
                )?;
                copy_page(
                    stream,
                    scales,
                    x.d8.cu_deviceptr(),
                    4 * io.cols * 2 * x.n_sb(),
                )?;
            }
        }
        copy_page(stream, selected, io.sel.cu_deviceptr(), 4 * n)?;
        copy_page(
            stream,
            out,
            io.rows.cu_deviceptr(),
            4 * n * self.shape.hidden,
        )?;
        let data = (&self.calls[at] as *const Call).cast_mut().cast::<c_void>();
        // SAFETY: the fixed call table and mapped buffers outlive the stream's
        // work and captured graphs. The callback calls no CUDA API.
        let rc = unsafe { sys::cuLaunchHostFunc(stream.cu_stream(), Some(serve), data) };
        crate::graph::cu(rc, "cuLaunchHostFunc (host expert tier)")?;
        copy_page(
            stream,
            io.rows.cu_deviceptr(),
            out,
            4 * n * self.shape.hidden,
        )
    }

    fn enqueue_block(
        &mut self,
        _: &Gpu,
        _: &crate::weights::Weights,
        _: usize,
        _: TierBlock<'_>,
    ) -> Result<(), GpuError> {
        Err(GpuError::shape(
            WHAT,
            "a prompt block on the host expert tier: use step columns",
        ))
    }

    fn block_bytes(&self) -> usize {
        0
    }

    fn check(&self) -> Result<(), GpuError> {
        let state = self
            .state
            .lock()
            .map_err(|_| GpuError::shape(WHAT, "the host expert state was poisoned"))?;
        match &state.failure {
            Some(e) => Err(GpuError::shape(WHAT, e.clone())),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests;
