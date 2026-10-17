//! The embedding architectures a side seat serves, with each model's input bound.

use crate::side_seat::Row;

#[cfg(feature = "gemma-embedding2")]
use crate::side_seat::gemma_embedding2::{bytes, open};
#[cfg(not(feature = "gemma-embedding2"))]
use crate::side_seat::{no_encoder_bytes as bytes, no_encoder_open as open};

/// The architectures a side seat serves.
pub const ROWS: &[Row] = &[
    Row {
        arch: model::arch::GEMMA_EMBEDDING,
        ctx: model::arch::gemma_embedding::hparams::CARD_CTX,
        bytes,
        open,
        projector_file: None,
        projector_types: &[],
        audio_bytes: |_| Ok(0),
    },
    // The file's context_length is 262144; the model card names 8192 tokens.
    Row {
        arch: model::arch::GEMMA_EMBEDDING2,
        ctx: model::arch::gemma_embedding2::hparams::CARD_CTX,
        bytes,
        open,
        projector_file: Some("mmproj-embeddinggemma-2-BF16.gguf"),
        projector_types: &[vision::arch::gemma4v::PROJECTOR_TYPE],
        audio_bytes,
    },
];

fn audio_bytes(file: &gguf::Gguf) -> Result<u64, crate::GateError> {
    #[cfg(feature = "gemma-embedding2")]
    {
        Ok(if vision::arch::open_audio(file)?.is_some() {
            bloomery_gpu_gemma4a::arch::gemma4a::card_bytes()
        } else {
            0
        })
    }
    #[cfg(not(feature = "gemma-embedding2"))]
    {
        let _ = file;
        Ok(0)
    }
}

#[cfg(feature = "gemma-embedding2")]
pub(super) fn open_audio(
    gpu: &bloomery_gpu::Gpu,
    file: &gguf::Gguf,
) -> Result<Box<dyn bloomery_gpu_vision::Tower>, bloomery_gpu::GpuError> {
    Ok(Box::new(
        bloomery_gpu_gemma4a::arch::gemma4a::Encoder::load(gpu, file)?,
    ))
}

#[cfg(feature = "gemma-embedding2")]
pub(super) fn gemma_embedding_dims(split: &gguf::Split) -> Result<Vec<usize>, crate::GateError> {
    const MRL: [usize; 3] = [128, 256, 512];
    let read = model::arch::spec(split)?;
    let n_out = *read
        .spec
        .encoder
        .as_ref()
        .and_then(|e| e.proj.last())
        .ok_or("embedding model: no output projection")? as usize;
    if n_out < MRL[0] {
        return Err(format!(
            "embedding model: embedding_length_out {n_out} cannot serve the {}-wide MRL cut",
            MRL[0]
        )
        .into());
    }
    let mut dims: Vec<_> = MRL.into_iter().filter(|&width| width < n_out).collect();
    dims.push(n_out);
    Ok(dims)
}
