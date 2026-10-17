//! Gemma4a's host mel input from decoded mono PCM at 16 kHz. The reference is
//! llama.cpp's `tools/mtmd/mtmd-audio.cpp`: semicausal padding per
//! 30-second chunk, a periodic Hann, magnitude FFT, HTK filters and log floor.

use crate::{MediaModel, Prepared, Rgb8, VisionError};

pub use crate::media::AudioChunk as Mel;

pub const BEGIN: &str = "<|audio>";
pub const END: &str = "<audio|>";
pub const SAMPLE_RATE: usize = 16_000;
pub const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE;
pub const MEL_BINS: usize = 128;
const FFT: usize = 512;
const _: () = assert!(FFT.is_power_of_two());
const WINDOW: usize = 320;
const HOP: usize = 160;
const BINS: usize = FFT / 2 + 1;

/// The fixed window, FFT twiddles and HTK filterbank, built once per encoder.
pub struct Preprocessor {
    hann: [f32; FFT],
    sin: [f32; FFT],
    cos: [f32; FFT],
    bitrev: [usize; FFT],
    filters: Vec<Filter>,
}

struct Filter {
    groups: Vec<(usize, [f32; 4])>,
    last: f32,
}

impl Default for Preprocessor {
    fn default() -> Self {
        let mut hann = [0.0; FFT];
        for (i, h) in hann[..WINDOW].iter_mut().enumerate() {
            *h = 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / WINDOW as f32).cos();
        }
        let sin = std::array::from_fn(|i| {
            ((2.0 * std::f64::consts::PI * i as f64 / FFT as f64) as f32).sin()
        });
        let cos = std::array::from_fn(|i| {
            ((2.0 * std::f64::consts::PI * i as f64 / FFT as f64) as f32).cos()
        });
        let hi = 2595.0 * (1.0 + SAMPLE_RATE as f64 / 2.0 / 700.0).log10();
        let edges: [f64; MEL_BINS + 2] = std::array::from_fn(|i| {
            let mel = hi * (i as f64 / (MEL_BINS + 1) as f64);
            700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
        });
        let mut filters = vec![0.0; MEL_BINS * BINS];
        for (m, row) in filters.as_chunks_mut::<BINS>().0.iter_mut().enumerate() {
            let [left, center, right] = [edges[m], edges[m + 1], edges[m + 2]];
            for (k, w) in row.iter_mut().enumerate() {
                let hz = k as f64 * (SAMPLE_RATE as f64 / FFT as f64);
                *w = if hz >= left && hz <= center {
                    ((hz - left) / (center - left).max(1e-30)) as f32
                } else if hz > center && hz <= right {
                    ((right - hz) / (right - center).max(1e-30)) as f32
                } else {
                    0.0
                };
            }
        }
        Self {
            hann,
            sin,
            cos,
            bitrev: std::array::from_fn(|i| i.reverse_bits() >> (usize::BITS - FFT.ilog2())),
            filters: filters
                .as_chunks::<BINS>()
                .0
                .iter()
                .map(|row| Filter {
                    groups: row[..BINS - 1]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(_, weights)| weights.iter().any(|&w| w != 0.0))
                        .map(|(i, weights)| (4 * i, weights))
                        .collect(),
                    last: row[BINS - 1],
                })
                .collect(),
        }
    }
}

impl MediaModel for Preprocessor {
    fn embedding_audio(&self) -> Option<(&str, &str, usize)> {
        Some((BEGIN, END, SAMPLE_RATE))
    }

    fn prepare_audio(&self, pcm: &[f32]) -> Result<Vec<Mel>, VisionError> {
        self.prepare(pcm)
    }

    fn image_placeholder(&self, _index: usize) -> std::borrow::Cow<'_, str> {
        panic!("the audio projector has no image placeholder")
    }

    fn image_marker(&self) -> &str {
        panic!("the audio projector has no image marker")
    }

    fn image_token(&self) -> Option<u32> {
        None
    }

    fn prepare(&self, _image: &Rgb8) -> Result<Prepared, VisionError> {
        Err(VisionError::Audio(
            "the audio projector does not support images".into(),
        ))
    }
}

impl Preprocessor {
    /// Prepare every chunk, in order. Input is already decoded, downmixed and
    /// resampled; no waveform normalization is applied. Empty input, a final
    /// one-sample chunk, non-finite samples and spectral overflow are refused.
    pub fn prepare(&self, pcm: &[f32]) -> Result<Vec<Mel>, VisionError> {
        if pcm.is_empty() || pcm.len() % CHUNK_SAMPLES == 1 {
            return Err(VisionError::Audio(
                "gemma4a needs at least two samples in every chunk".into(),
            ));
        }
        if let Some(i) = pcm.iter().position(|x| !x.is_finite()) {
            return Err(VisionError::Audio(format!(
                "gemma4a PCM sample {i} is not finite"
            )));
        }
        let mut chunks = Vec::with_capacity(pcm.len().div_ceil(CHUNK_SAMPLES));
        let mut input = [0.0; FFT];
        let mut output = [0.0; FFT * 2];
        for pcm in pcm.chunks(CHUNK_SAMPLES) {
            // C++ truncates the signed (samples - 161) / hop toward zero.
            let frames = pcm.len().saturating_sub(WINDOW / 2 + 1) / HOP + 1;
            let mut data = vec![0.0; frames * MEL_BINS];
            for frame in 0..frames {
                for (j, x) in input.iter_mut().enumerate() {
                    let at = frame * HOP + j;
                    *x = if at >= WINDOW / 2 && at - WINDOW / 2 < pcm.len() {
                        self.hann[j] * pcm[at - WINDOW / 2]
                    } else {
                        0.0
                    };
                }
                self.fft(&input, &mut output);
                for k in 0..BINS {
                    let power =
                        output[2 * k] * output[2 * k] + output[2 * k + 1] * output[2 * k + 1];
                    if !power.is_finite() {
                        return Err(VisionError::Audio("gemma4a spectrum is not finite".into()));
                    }
                    output[k] = power.sqrt();
                }
                for (m, filter) in self.filters.iter().enumerate() {
                    // Keep the reference's four-product groups and f64 accumulation order.
                    let mut sum = 0.0_f64;
                    for &(k, weights) in &filter.groups {
                        sum += (output[k] * weights[0]
                            + output[k + 1] * weights[1]
                            + output[k + 2] * weights[2]
                            + output[k + 3] * weights[3]) as f64;
                    }
                    sum += (output[BINS - 1] * filter.last) as f64;
                    data[m * frames + frame] = sum.max(0.001_f32 as f64).ln() as f32;
                }
            }
            chunks.push(Mel {
                frames,
                data,
                span_len: frames.div_ceil(4),
            });
        }
        Ok(chunks)
    }

    fn fft(&self, input: &[f32; FFT], output: &mut [f32; FFT * 2]) {
        for (i, &at) in self.bitrev.iter().enumerate() {
            output[2 * i] = input[at];
            output[2 * i + 1] = 0.0;
        }
        let mut n = 2;
        while n <= FFT {
            let half = n / 2;
            for chunk in output.chunks_exact_mut(2 * n) {
                for k in 0..half {
                    let re = self.cos[k * (FFT / n)];
                    let im = -self.sin[k * (FFT / n)];
                    let re_even = chunk[2 * k];
                    let im_even = chunk[2 * k + 1];
                    let re_odd = chunk[2 * (k + half)];
                    let im_odd = chunk[2 * (k + half) + 1];
                    chunk[2 * k] = re_even + re * re_odd - im * im_odd;
                    chunk[2 * k + 1] = im_even + re * im_odd + im * re_odd;
                    chunk[2 * (k + half)] = re_even - re * re_odd + im * im_odd;
                    chunk[2 * (k + half) + 1] = im_even - re * im_odd - im * re_odd;
                }
            }
            n *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_pcm() {
        let prep = Preprocessor::default();
        for pcm in [
            vec![],
            vec![0.0],
            vec![f32::NAN; 2],
            vec![f32::INFINITY; 2],
            vec![f32::MAX; 2],
            vec![0.0; CHUNK_SAMPLES + 1],
        ] {
            assert!(matches!(prep.prepare(&pcm), Err(VisionError::Audio(_))));
        }
    }
}
