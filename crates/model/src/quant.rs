//! The engine's one Q8_0 quantizer at the model crate's shared level, where
//! the qwen3 family's q8_0 cache seed reads it. It lives in `gguf::quant`
//! beside `Q8Block` and the activation round trips that code with it.

// PIN(2026-10-21): the gguf activation round-trip test owns this re-export's refusal matrix.
pub use gguf::quant::quantize_q8_0;
