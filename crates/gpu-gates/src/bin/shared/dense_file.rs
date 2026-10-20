//! Synthetic weight files and activation columns for dense-format gates.

use bloomery_gpu::Gpu;
use bloomery_gpu_gates::{GateError, GateRng, par_map};
use cuda_core::DeviceBuffer;
use gguf::quant::{GgmlType, dequant_row};

pub(crate) type Decl = (String, Vec<u64>, GgmlType, Vec<u8>);
type Columns = (Vec<f32>, DeviceBuffer<f32>, DeviceBuffer<f32>);

/// `rows` Q8_0 rows of `k` values drawn from `seed`: f16 scales of
/// `2^-8` to `2^1`, codes uniform over all 256 bytes.
pub(crate) fn q8_0_bytes(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = GateRng(seed);
    let mut out = Vec::with_capacity(rows * k / 32 * 34);
    for _ in 0..rows * k / 32 {
        let bits = ((7 + rng.below(9)) << 10 | rng.below(1024)) as u16;
        out.extend(bits.to_le_bytes());
        out.extend((0..32).map(|_| rng.below(256) as u8));
    }
    out
}

/// A weight of type `ty` of `rows` rows of `k` as a tensor of a file.
pub(crate) fn decl(name: &str, ty: GgmlType, (rows, k): (usize, usize), bytes: Vec<u8>) -> Decl {
    (name.to_string(), vec![k as u64, rows as u64], ty, bytes)
}

/// The host rows of a file tensor's `bytes`: `rows` rows of `k` values
/// of type `ty`.
pub(crate) fn host_rows(ty: GgmlType, bytes: &[u8], rows: usize, k: usize) -> Vec<Vec<f32>> {
    let row_bytes = bytes.len() / rows;
    par_map(rows, |r| {
        let mut v = vec![0.0f32; k];
        dequant_row(ty, &bytes[r * row_bytes..][..row_bytes], &mut v)
            .unwrap_or_else(|e| panic!("row {r}: {e}"));
        v
    })
}

/// `m` columns of `k` activations uniform on ±2 from `seed`, multiplied
/// by `scale`, on the card, and an output of `m * rows` NaN sentinels.
pub(crate) fn columns(
    gpu: &Gpu,
    (m, k, rows): (usize, usize, usize),
    seed: u64,
    scale: f32,
) -> Result<Columns, GateError> {
    let mut rng = GateRng(seed);
    let x: Vec<f32> = (0..m * k).map(|_| rng.uni(-2.0, 2.0) * scale).collect();
    let x_dev = DeviceBuffer::from_host(gpu.stream(), &x)?;
    let y_dev = DeviceBuffer::from_host(gpu.stream(), &vec![f32::NAN; m * rows])?;
    Ok((x, x_dev, y_dev))
}
