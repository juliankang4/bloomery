//! Recurrent state a layer keeps on the card: the state lanes, the conv
//! ring and, for a delta layer whose prompt runs the chunked rule, the carry
//! (`linear::delta`'s module doc). A delta layer passes
//! [`crate::linear::LinearShape::lens`] and `carry_len`; a later scan passes
//! its own lengths. The lengths are the only architecture facts this module
//! takes.

use crate::GpuError;
use crate::linear::delta::NEVER;
use cuda_core::{CudaStream, DeviceBuffer};

/// One layer's recurrent store: `state_f32` f32s of state (every lane) and
/// `ring_f32` f32s of conv ring, and — when a load keeps a lane per verify
/// row — the lanes' stamps ([`fresh_stamps`]).
pub(crate) struct RecStore {
    pub(crate) state: DeviceBuffer<f32>,
    pub(crate) ring: DeviceBuffer<f32>,
    pub(crate) lanes: usize,
    /// `lanes` u32 stamps ([`fresh_stamps`]); `None` on a store whose
    /// launches do not read stamps.
    pub(crate) stamp: Option<DeviceBuffer<u32>>,
    /// The carry ([`RecStore::carried`]); `None` on a store whose prompt
    /// does not run the chunked rule.
    pub(crate) carry: Option<DeviceBuffer<f32>>,
}

impl RecStore {
    /// A zeroed store of `lanes` lanes over `lens` (`[state_f32, ring_f32]`),
    /// with no stamps. Load-time only.
    pub(crate) fn new(
        stream: &CudaStream,
        lens: [usize; 2],
        lanes: usize,
    ) -> Result<RecStore, GpuError> {
        let [state, ring] = lens;
        Ok(RecStore {
            state: DeviceBuffer::zeroed(stream, state)?,
            ring: DeviceBuffer::zeroed(stream, ring)?,
            lanes,
            stamp: None,
            carry: None,
        })
    }

    /// [`RecStore::new`] with a zeroed carry of `carry` f32s: a delta layer
    /// whose prompt runs the chunked rule. Load-time only.
    pub(crate) fn carried(
        stream: &CudaStream,
        lens: [usize; 2],
        lanes: usize,
        carry: usize,
    ) -> Result<RecStore, GpuError> {
        let mut r = RecStore::new(stream, lens, lanes)?;
        r.carry = Some(DeviceBuffer::zeroed(stream, carry)?);
        Ok(r)
    }

    /// [`RecStore::new`] with its stamps fresh ([`fresh_stamps`]). Load-time
    /// only.
    pub(crate) fn stamped(
        stream: &CudaStream,
        lens: [usize; 2],
        lanes: usize,
    ) -> Result<RecStore, GpuError> {
        let mut r = RecStore::new(stream, lens, lanes)?;
        r.stamp = Some(DeviceBuffer::from_host(stream, &fresh_stamps(lanes))?);
        Ok(r)
    }

    /// The stamps back to a fresh store's ([`fresh_stamps`]); a store with
    /// none is refused by name. Never inside a capture.
    pub(crate) fn restamp(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        let fresh = fresh_stamps(self.lanes);
        self.stamp_with(stream, &fresh)
    }

    /// The stamps with lane `lane` at position `at` and every other lane
    /// never written ([`stamps_at`]). A store with no stamps is refused by
    /// name. Never inside a capture.
    pub(crate) fn restamp_at(
        &mut self,
        stream: &CudaStream,
        lane: u32,
        at: u32,
    ) -> Result<(), GpuError> {
        let st = stamps_at(self.lanes, lane, at)?;
        self.stamp_with(stream, &st)
    }

    fn stamp_with(&mut self, stream: &CudaStream, st: &[u32]) -> Result<(), GpuError> {
        match self.stamp.as_mut() {
            Some(stamp) => Ok(stamp.copy_from_host(stream, st)?),
            None => Err(GpuError::state(
                "recur::RecStore::restamp",
                "a store of stamped lanes",
            )),
        }
    }

    /// Every lane of the state, the whole ring and the carry back to zero on
    /// the device, ordered on `stream`. The stamps are unchanged.
    pub(crate) fn clear(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.state.zero_async(stream)?;
        self.ring.zero_async(stream)?;
        if let Some(carry) = self.carry.as_mut() {
            carry.zero_async(stream)?;
        }
        Ok(())
    }

    pub(crate) fn bytes(&self) -> usize {
        self.state.num_bytes()
            + self.ring.num_bytes()
            + self.stamp.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.carry.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }
}

/// A fresh store's stamps over `lanes` lanes: lane 0 holds position 0, every
/// other lane was never written ([`NEVER`]).
pub(crate) fn fresh_stamps(lanes: usize) -> Vec<u32> {
    let mut st = vec![NEVER; lanes];
    if let Some(first) = st.first_mut() {
        *first = 0;
    }
    st
}

/// The stamps of a store over `lanes` lanes whose lane `lane` stands at
/// position `at` and whose every other lane was never written. A lane at or
/// past `lanes` is refused by name.
pub(crate) fn stamps_at(lanes: usize, lane: u32, at: u32) -> Result<Vec<u32>, GpuError> {
    let mut st = vec![NEVER; lanes];
    *st.get_mut(lane as usize).ok_or_else(|| {
        GpuError::shape("recur::stamps_at", format!("lane {lane} of {lanes} lanes"))
    })? = at;
    Ok(st)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_clear_keeps_stamps_and_addresses() -> Result<(), GpuError> {
        let (_ctx, stream) = crate::capsync::fresh_stream(0)?;
        for (lanes, carry) in [(1, 0), (2, 7), (4, 9)] {
            let mut store = RecStore::stamped(&stream, [17 * lanes, 13], lanes)?;
            if carry > 0 {
                store.carry = Some(DeviceBuffer::zeroed(&stream, carry)?);
            }
            let lane = u32::try_from(lanes - 1).expect("test lane fits u32");
            store.restamp_at(&stream, lane, 29)?;
            let stamps = store
                .stamp
                .as_ref()
                .expect("stamped store")
                .to_host_vec(&stream)?;
            let addresses = (
                store.state.cu_deviceptr(),
                store.ring.cu_deviceptr(),
                store.carry.as_ref().map(DeviceBuffer::cu_deviceptr),
            );
            for buf in [&mut store.state, &mut store.ring]
                .into_iter()
                .chain(store.carry.iter_mut())
            {
                buf.copy_from_host(&stream, &vec![f32::NAN; buf.len()])?;
            }
            store.clear(&stream)?;
            for buf in [&store.state, &store.ring]
                .into_iter()
                .chain(store.carry.iter())
            {
                assert!(buf.to_host_vec(&stream)?.iter().all(|v| v.to_bits() == 0));
            }
            assert_eq!(
                addresses,
                (
                    store.state.cu_deviceptr(),
                    store.ring.cu_deviceptr(),
                    store.carry.as_ref().map(DeviceBuffer::cu_deviceptr),
                )
            );
            assert_eq!(
                store
                    .stamp
                    .as_ref()
                    .expect("stamped store")
                    .to_host_vec(&stream)?,
                stamps
            );
        }
        Ok(())
    }
}
