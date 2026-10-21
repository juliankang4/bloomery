//! The engine's one Q8_0 quantizer: the 32-value block rule the reference's
//! weight requant runs (`quantize_row_q8_0`, ggml-quants.c's x86 branch),
//! shared by the deepseek2 requant that owned it first and the qwen3
//! family's q8_0 cache seed.

// PIN(2026-10-21): the gguf activation round-trip test owns this re-export's refusal matrix.
pub use gguf::quant::quantize_q8_0;
