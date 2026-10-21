//! The engine's one Q8_0 quantizer: the 32-value block rule the reference's
//! weight requant runs (`quantize_row_q8_0`, ggml-quants.c's x86 branch),
//! shared by the deepseek2 requant that owned it first and the qwen3
//! family's q8_0 cache seed.

pub use gguf::quant::quantize_q8_0;

#[cfg(test)]
mod tests {
    #[test]
    fn q8_0_refuses_non_finite_blocks() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut x = [1.0; 32];
            x[17] = bad;
            for x in [x, [bad; 32]] {
                let err = std::panic::catch_unwind(|| super::quantize_q8_0(&x))
                    .expect_err("non-finite Q8_0 input must fail");
                let msg = err.downcast_ref::<String>().unwrap();
                assert!(
                    msg.contains("quantize_q8_0: non-finite activation value"),
                    "{msg}"
                );
            }
        }
        let block = super::quantize_q8_0(&[127.0; 32]);
        assert_eq!(block.d, 0x3c00);
        assert_eq!(block.q, [127; 32]);
    }
}
