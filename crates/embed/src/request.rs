//! llama-server's embedding request fields and response shapes.

use serde_json::{Map, Value, json};

use crate::{Norm, encode_base64, normalize};

/// A malformed request, or an input longer than the encoder's context.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    /// A field of the request that is malformed, named by `field`.
    #[error("{message}")]
    Field { field: String, message: String },
    #[error("input ({tokens} tokens) is larger than the max context size ({ctx} tokens)")]
    Context { tokens: usize, ctx: usize },
}

/// Parsed fields and input boundaries. Tokenization and media preparation belong to the caller.
#[derive(Debug)]
pub struct Request<'a> {
    pub key: &'static str,
    pub inputs: Vec<&'a Value>,
    pub prompt_name: Option<&'a str>,
    model: &'a str,
    oai: bool,
    base64: bool,
    norm: Norm,
    width: usize,
}

impl<'a> Request<'a> {
    /// Read the model name, input batches, prompt name, encoding, normalization and MRL width.
    /// `dims` is the model's supported widths, ascending and nonempty.
    pub fn parse(
        b: &'a Map<String, Value>,
        model: &'a str,
        dims: &[usize],
        oai: bool,
    ) -> Result<Self, Error> {
        if oai && let Some(name) = b.get("model").filter(|v| !v.is_null()) {
            let name = name
                .as_str()
                .ok_or_else(|| field("model", "model must be a string"))?;
            if name != model {
                return Err(field(
                    "model",
                    format!("model {name:?}: embeddings are served by {model:?}"),
                ));
            }
        }
        let (key, prompt, oai) = match (b.get("input"), b.get("content")) {
            (Some(v), _) => ("input", v, oai),
            (None, Some(v)) => ("content", v, false),
            (None, None) => {
                return Err(field("input", "\"input\" or \"content\" must be provided"));
            }
        };
        let base64 = match b.get("encoding_format") {
            None | Some(Value::Null) => false,
            Some(Value::String(f)) if f == "float" => false,
            Some(Value::String(f)) if f == "base64" => true,
            Some(Value::String(_)) => {
                return Err(field(
                    "encoding_format",
                    "The format to return the embeddings in. Can be either float or base64",
                ));
            }
            Some(v) => return Err(wrong_type("encoding_format", "a string", v)),
        };
        let norm = match integer(b, "embd_normalize")? {
            None => Norm::default(),
            Some(p) => i32::try_from(p)
                .map_err(|e| e.to_string())
                .and_then(Norm::of)
                .map_err(|e| field("embd_normalize", format!("embd_normalize {p}: {e}")))?,
        };
        let width = match integer(b, "dimensions")? {
            None => *dims
                .last()
                .ok_or_else(|| Error::Invalid("the embedding model has no widths".into()))?,
            Some(k) => dims
                .iter()
                .copied()
                .find(|&d| i64::try_from(d).is_ok_and(|d| d == k))
                .ok_or_else(|| {
                    let widths: Vec<String> = dims.iter().map(ToString::to_string).collect();
                    field(
                        "dimensions",
                        format!(
                            "dimensions {k}: {model} cuts its vectors to {}",
                            widths.join(", ")
                        ),
                    )
                })?,
        };
        let prompt_name = b
            .get("prompt_name")
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| field("prompt_name", "prompt_name must be a string"))
            })
            .transpose()?;
        let inputs = match prompt {
            Value::Array(a) if !a.iter().any(|x| x.is_u64() || x.is_i64()) => a.iter().collect(),
            v => vec![v],
        };
        if inputs.is_empty() {
            return Err(field(key, "\"input\" must not be empty"));
        }
        Ok(Self {
            key,
            inputs,
            prompt_name,
            model,
            oai,
            base64,
            norm,
            width,
        })
    }

    /// Cut each full pooled vector before normalization, including renormalization for MRL.
    pub fn answer(&self, pooled: &[Vec<f32>], n_tokens: usize) -> Result<Value, Error> {
        if pooled.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::Invalid(
                "the pooled vector contains a non-finite value".into(),
            ));
        }
        let vectors = pooled
            .iter()
            .map(|v| {
                let cut = v.get(..self.width).ok_or_else(|| {
                    Error::Invalid("the pooled vector is shorter than dimensions".into())
                })?;
                normalize(cut, self.norm).map_err(Error::Invalid)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // JSON numbers use the widened f32 value, as llama-server writes it.
        let json_vector =
            |v: &[f32]| -> Value { v.iter().map(|&x| Value::from(f64::from(x))).collect() };
        if !self.oai {
            return Ok(Value::Array(
                vectors
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| json!({ "embedding": [json_vector(&v)], "index": i }))
                    .collect(),
            ));
        }
        let data: Vec<Value> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if self.base64 {
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    json!({
                        "embedding": encode_base64(&bytes),
                        "encoding_format": "base64",
                        "index": i,
                        "object": "embedding",
                    })
                } else {
                    json!({ "embedding": json_vector(v), "index": i, "object": "embedding" })
                }
            })
            .collect();
        Ok(json!({
            "data": data,
            "model": self.model,
            "object": "list",
            "usage": { "prompt_tokens": n_tokens, "total_tokens": n_tokens },
        }))
    }
}

/// Refuse an empty token input or an input beyond the model's context before a pass.
pub fn check_tokens(tokens: usize, ctx: usize) -> Result<(), Error> {
    if tokens == 0 {
        return Err(Error::Invalid("Input content cannot be empty".into()));
    }
    if tokens > ctx {
        return Err(Error::Context { tokens, ctx });
    }
    Ok(())
}

/// An integer field. Absent or null is unset; integer-valued floats pass.
pub fn integer(o: &Map<String, Value>, k: &str) -> Result<Option<i64>, Error> {
    let Some(v) = o.get(k).filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    if !v.is_number() {
        return Err(wrong_type(k, "an integer", v));
    }
    if let Some(i) = v.as_i64() {
        return Ok(Some(i));
    }
    match v.as_f64() {
        Some(f) if f.fract() == 0.0 && f >= i64::MIN as f64 && f < i64::MAX as f64 => {
            Ok(Some(f as i64))
        }
        _ => Err(field(k, format!("{k} must be an integer, not {v}"))),
    }
}

/// A field's refusal naming the type it takes and the type it got.
pub fn wrong_type(k: &str, takes: &str, v: &Value) -> Error {
    field(k, format!("{k} must be {takes}, not {}", json_type(v)))
}

/// The refusal of the request field `k`.
fn field(k: &str, message: impl Into<String>) -> Error {
    Error::Field {
        field: k.to_owned(),
        message: message.into(),
    }
}

/// A JSON value's type in a field's refusal.
pub fn json_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}
