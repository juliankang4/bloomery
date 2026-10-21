use std::collections::BTreeMap;
use std::io::Read;

use hf::quant_config::{Config, plan};
use safetensors::Dtype;
use serde_json::Value;

use super::Geometry;

pub(super) const GLM: Geometry = Geometry {
    layers: 45,
    heads: 64,
    nope: 256,
    v: 256,
    experts: 288,
};

pub(super) fn gunzip(path: &str) -> Value {
    let gz = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut text = Vec::new();
    flate2::read::GzDecoder::new(gz.as_slice())
        .read_to_end(&mut text)
        .unwrap();
    serde_json::from_slice(&text).unwrap()
}

/// The checkpoint's placed tensors with their logical shapes: an NVFP4 weight holds two values
/// a byte.
pub(super) fn logical(
    config: &Config,
    table: &[(String, Dtype, Vec<usize>)],
) -> Vec<(String, Vec<usize>)> {
    let p = plan(
        config,
        table.iter().map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
    );
    assert!(p.refused.is_empty(), "{:?}", p.refused);
    table
        .iter()
        .filter_map(|(n, _, s)| Some((n.clone(), p.placed.get(n)?.format.logical(s))))
        .collect()
}

pub(super) fn table(header: &Value) -> Vec<(String, Dtype, Vec<usize>)> {
    header
        .as_object()
        .unwrap()
        .iter()
        .filter(|(n, _)| *n != "__metadata__")
        .map(|(n, e)| {
            let shape = e["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            let d = e["dtype"].as_str().unwrap();
            let dtype = Dtype::of(d).unwrap_or_else(|| panic!("{n}: {d}"));
            (n.clone(), dtype, shape)
        })
        .collect()
}

/// The unsloth GGUF's names and ggml dims from a `tensor`, `dimensions` table.
pub(super) fn gguf_table(text: &str, keep: impl Fn(&str) -> bool) -> BTreeMap<String, Vec<usize>> {
    let cols: Vec<&str> = text.lines().next().unwrap().split('\t').collect();
    let (t, d) = (
        cols.iter().position(|c| *c == "tensor").unwrap(),
        cols.iter().position(|c| *c == "dimensions").unwrap(),
    );
    text.lines()
        .skip(1)
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .filter(|r| keep(r[t]))
        .map(|r| {
            (
                r[t].to_string(),
                r[d].split('x').map(|n| n.parse().unwrap()).collect(),
            )
        })
        .collect()
}
