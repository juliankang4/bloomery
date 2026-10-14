//! EmbeddingGemma 2's side-seat row; the encoder owns its projection and pooling.

use std::sync::Arc;

use bloomery_gpu::Gpu;
use bloomery_gpu_gemma_embedding2::{Encoder, card_bytes};
use gguf::Split;
use serve::Tokenizer;
use serve::embed::{Embed, EmbedError};

use super::{Side, architectures};
use crate::bind::Vocab;
use crate::vision_seat::EncoderSeat;

pub(super) fn bytes(split: &Split, ctx: usize) -> Result<u64, String> {
    card_bytes(split, ctx).map_err(|e| e.to_string())
}

pub(super) fn open(side: &Side, seat: &EncoderSeat) -> Result<Box<dyn Embed>, String> {
    let load = || -> Result<Box<dyn Embed>, crate::GateError> {
        let split = Split::open(&side.path)?;
        let dims = architectures::gemma_embedding2_dims(&split)?;
        let actual = card_bytes(&split, side.ctx)?;
        if actual != side.bytes || seat.reserve.is_some_and(|b| b != actual) {
            return Err(format!(
                "embedding model: load needs {actual} B, sized {} B, reserved {:?}",
                side.bytes, seat.reserve
            )
            .into());
        }
        let tok = Arc::new(Vocab::new(tokenizer::Tokenizer::from_gguf(&side.path)?)?);
        let device = crate::gpu_census::census()?
            .into_iter()
            .find(|d| d.ordinal == seat.ordinal)
            .ok_or_else(|| {
                format!(
                    "embedding model: card {} is no longer visible",
                    seat.ordinal
                )
            })?
            .device();
        let gpu = Gpu::open_card(&seat.name, Some(device))?;
        let encoder = Encoder::open(split, side.ctx, gpu)?;
        let name = side
            .path
            .file_name()
            .ok_or("embedding model: the path has no file name")?
            .to_string_lossy()
            .into_owned();
        Ok(Box::new(Model {
            encoder,
            tok,
            name,
            dims,
        }))
    };
    load().map_err(|e| e.to_string())
}

struct Model {
    encoder: Encoder,
    tok: Arc<Vocab>,
    name: String,
    dims: Vec<usize>,
}

impl Embed for Model {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.tok.clone()
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn ctx(&self) -> usize {
        self.encoder.ctx()
    }

    fn dims(&self) -> Vec<usize> {
        self.dims.clone()
    }

    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>, EmbedError> {
        self.encoder
            .pass(ids, false)
            .map(|pass| pass.pooled)
            .map_err(|e| EmbedError(e.to_string()))
    }
}
