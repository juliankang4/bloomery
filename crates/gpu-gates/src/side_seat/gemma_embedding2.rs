//! The EmbeddingGemma side seats; the encoder owns projection and pooling.

use std::sync::Arc;

use bloomery_gpu::Gpu;
use bloomery_gpu_gemma_embedding2::{Encoder, card_bytes};
use gguf::Split;
use serve::Tokenizer;
use serve::embed::{Embed, EmbedError};

use super::{Side, architectures};
use crate::bind::Vocab;
use crate::vision_seat::EncoderSeat;
#[cfg(feature = "vision")]
use crate::vision_seat::{Image, Opened, Resident};

pub(super) fn bytes(split: &Split, ctx: usize) -> Result<u64, String> {
    card_bytes(split, ctx).map_err(|e| e.to_string())
}

pub(super) fn open(side: &Side, seat: &EncoderSeat) -> Result<Box<dyn Embed>, String> {
    let load = || -> Result<Box<dyn Embed>, crate::GateError> {
        let split = Split::open(&side.path)?;
        let dims = architectures::gemma_embedding_dims(&split)?;
        let actual = card_bytes(&split, side.ctx)?
            .checked_add(side.projector.as_ref().map_or(0, |p| p.card.total()))
            .and_then(|n| n.checked_add(side.audio_bytes))
            .ok_or("embedding model and tower byte count overflows")?;
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
        #[cfg(feature = "vision")]
        let vision = side
            .projector
            .as_ref()
            .map(|p| {
                let at = Opened {
                    path: p.path.clone(),
                    seat: seat.clone(),
                    media: p.media.clone(),
                    cap: p.cap,
                    card: p.card,
                    limits: p.limits,
                };
                Resident::open(encoder.context(), &at)
            })
            .transpose()?;
        let (audio, audio_media) = if side.audio_bytes != 0 {
            let projector = side
                .projector
                .as_ref()
                .ok_or("audio tower has no projector file")?;
            let file = gguf::Gguf::open(&projector.path)?;
            let tower = architectures::open_audio(encoder.runtime(), &file)?;
            if tower.weight_bytes() + tower.scratch_bytes() != side.audio_bytes {
                return Err(format!(
                    "audio tower loaded {} B, reserved {} B",
                    tower.weight_bytes() + tower.scratch_bytes(),
                    side.audio_bytes
                )
                .into());
            }
            (Some(tower), vision::arch::open_audio(&file)?)
        } else {
            (None, None)
        };
        let name = side
            .path
            .file_name()
            .ok_or("embedding model: the path has no file name")?
            .to_string_lossy()
            .into_owned();
        Ok(Box::new(Model {
            encoder,
            audio,
            audio_media,
            #[cfg(feature = "vision")]
            vision,
            tok,
            name,
            path: side.path.display().to_string(),
            dims,
        }))
    };
    load().map_err(|e| e.to_string())
}

struct Model {
    encoder: Encoder,
    audio: Option<Box<dyn bloomery_gpu_vision::Tower>>,
    audio_media: Option<serve::media::SharedMediaModel>,
    #[cfg(feature = "vision")]
    vision: Option<Resident>,
    tok: Arc<Vocab>,
    name: String,
    path: String,
    dims: Vec<usize>,
}

impl Embed for Model {
    fn audio(&self) -> Option<serve::media::SharedMediaModel> {
        self.audio_media.clone()
    }

    fn media(&self) -> Option<serve::media::SharedMediaModel> {
        #[cfg(feature = "vision")]
        {
            self.vision.as_ref().map(Resident::media)
        }
        #[cfg(not(feature = "vision"))]
        {
            None
        }
    }

    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.tok.clone()
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn path(&self) -> String {
        self.path.clone()
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

    #[cfg(feature = "vision")]
    fn embed_media(
        &mut self,
        ids: &[u32],
        images: &[serve::media::MediaFeed],
        audio: &[serve::embed::AudioFeed],
    ) -> Result<Vec<f32>, EmbedError> {
        let mut run = || -> Result<Vec<f32>, crate::GateError> {
            let encoded = if images.is_empty() {
                None
            } else {
                let tower = self
                    .vision
                    .as_mut()
                    .ok_or("image input needs --embd-mmproj")?;
                Some(tower.encode_with(
                    self.encoder.context(),
                    images.iter().map(|i| Image::of(i.at, &i.prepared)),
                    crate::vision_seat::merged_cells,
                )?)
            };
            if audio.is_empty() {
                let spans = encoded
                    .as_ref()
                    .map(|e| e.rows().media_spans())
                    .unwrap_or_default();
                return Ok(self.encoder.pass_media(ids, &spans, false)?.pooled);
            }
            let tower = self
                .audio
                .as_mut()
                .ok_or("audio input needs a gemma4a --embd-mmproj")?;
            let mut rows = Vec::new();
            for feed in audio {
                let mut at = feed.at;
                for mel in &*feed.chunks {
                    let chunk = tower.encode_audio(mel)?;
                    rows.push((at..at + chunk.tokens, chunk.values));
                    at += chunk.tokens;
                }
            }
            if let Some(encoded) = &encoded {
                for span in encoded.rows().media_spans() {
                    let mut values = vec![0.0; span.rows.len()];
                    bloomery_gpu::media::widen_bf16(span.rows, &mut values);
                    rows.push((span.at, values));
                }
            }
            rows.sort_by_key(|(at, _)| at.start);
            let spans: Vec<_> = rows
                .iter()
                .map(|(at, values)| bloomery_gpu::media::MediaSpan {
                    at: at.clone(),
                    rows: values.as_slice(),
                })
                .collect();
            Ok(self.encoder.pass_media_f32(ids, &spans, false)?.pooled)
        };
        run().map_err(|e| EmbedError(e.to_string()))
    }
}
