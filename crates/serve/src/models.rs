//! `GET /v1/models` (and `/models`, and Ollama's `/api/tags`, as llama-server served it): the
//! models a server serves, in llama.cpp's server's shape: `data`, OpenAI's list, and `models`
//! beside it (`tools/server/server-context.cpp` at `a4cb4c61`, `get_res_models`: its `name` and
//! `model`, the entry's other fields there being placeholders). `GET /v1/models/{model}` (and
//! `/models/{model}`) is one entry of `data`, and a model the listing does not hold a 404 naming
//! the ones it does. Every server of this crate answers through [`listing`] and [`lookup`].

use serde_json::{Value, json};

/// One model of a listing: its id, `meta` as the server knows it, and `max_model_len` when the
/// server states one.
pub(crate) struct Listed<'a> {
    pub id: &'a str,
    pub meta: Value,
    pub max_model_len: Option<usize>,
}

/// The listing of `models`, in order, each `created` at the same unix second.
pub(crate) fn listing(models: Vec<Listed<'_>>, created: u64) -> Value {
    let names: Vec<Value> = models
        .iter()
        .map(|m| json!({ "name": m.id, "model": m.id }))
        .collect();
    let data: Vec<Value> = models
        .into_iter()
        .map(|m| {
            let mut entry = json!({
                "id": m.id,
                "object": "model",
                "created": created,
                "owned_by": "bloomery",
                "meta": m.meta,
            });
            if let (Some(n), Value::Object(o)) = (m.max_model_len, &mut entry) {
                o.insert("max_model_len".to_owned(), json!(n));
            }
            entry
        })
        .collect();
    json!({
        "models": names,
        "object": "list",
        "data": data,
    })
}

/// The model a `GET /v1/models/{model}` or `/models/{model}` path names, percent-decoded once
/// with a literal `+` preserved; a named error for invalid encoding, `None` for another path.
pub(crate) fn named(path: &str) -> Option<Result<String, String>> {
    crate::http::below(path, "models").map(|id| {
        crate::http::percent_decode(&id.replace('+', "%2B"))
            .map_err(|_| format!("invalid percent-encoded model name: {id:?}"))
    })
}

/// The entry of `listing`'s `data` whose id is exactly `id`; else the 404's message, naming
/// the models the listing holds. Path callers decode through [`named`]; JSON names are literal.
pub(crate) fn lookup(listing: &Value, id: &str) -> Result<Value, String> {
    let data = listing["data"].as_array().map_or(&[][..], Vec::as_slice);
    data.iter().find(|m| m["id"] == id).cloned().ok_or_else(|| {
        let served: Vec<&str> = data.iter().filter_map(|m| m["id"].as_str()).collect();
        format!(
            "model {id:?} not found: this server serves {}",
            served.join(", ")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path decodes its model once; lookup matches the literal id, and a model the listing
    /// does not hold is refused naming the ones it does.
    #[test]
    fn one_model_is_looked_up_by_its_id() {
        let v = listing(
            vec![
                Listed {
                    id: "org/Model A",
                    meta: json!({}),
                    max_model_len: Some(16),
                },
                Listed {
                    id: "e.gguf",
                    meta: json!({}),
                    max_model_len: None,
                },
            ],
            7,
        );
        assert_eq!(named("/v1/models/e.gguf"), Some(Ok("e.gguf".to_owned())));
        assert_eq!(
            named("/models/org%2FModel%20A"),
            Some(Ok("org/Model A".to_owned()))
        );
        assert_eq!(named("/v1/models/"), None);
        assert_eq!(named("/v1/models"), None);
        assert_eq!(named("/v1models/e.gguf"), None);
        assert_eq!(named("/v1/v1/models/e.gguf"), None);
        assert_eq!(lookup(&v, "e.gguf").unwrap(), v["data"][1]);
        assert!(lookup(&v, "org%2FModel%20A").is_err());
        for path in ["org/Model A", "org%2FModel%20A", "org/Model%20A"] {
            let id = named(&format!("/models/{path}")).unwrap().unwrap();
            assert_eq!(lookup(&v, &id).unwrap(), v["data"][0], "{path}");
        }
        assert_eq!(
            lookup(&v, "other").unwrap_err(),
            "model \"other\" not found: this server serves org/Model A, e.gguf"
        );
        // Literal percent escapes need their own encoding in a path; a `+` stays a `+`.
        let ids = |ids: &[&'static str]| {
            listing(
                ids.iter()
                    .map(|&id| Listed {
                        id,
                        meta: json!({}),
                        max_model_len: None,
                    })
                    .collect(),
                7,
            )
        };
        let v = ids(&["m/a", "m%2Fa"]);
        let id = named("/v1/models/m%2Fa").unwrap().unwrap();
        assert_eq!(lookup(&v, &id).unwrap(), v["data"][0]);
        let id = named("/v1/models/m%252Fa").unwrap().unwrap();
        assert_eq!(lookup(&v, &id).unwrap(), v["data"][1]);
        assert_eq!(lookup(&v, "m%2Fa").unwrap(), v["data"][1]);
        let v = ids(&["org/Model+A"]);
        for path in ["org%2FModel+A", "org%2FModel%2BA"] {
            let id = named(&format!("/models/{path}")).unwrap().unwrap();
            assert_eq!(lookup(&v, &id).unwrap(), v["data"][0]);
            assert!(lookup(&ids(&["org/Model A"]), &id).is_err());
        }
    }

    #[test]
    fn both_lists_name_every_model_in_order() {
        let v = listing(
            vec![
                Listed {
                    id: "m.gguf",
                    meta: json!({ "n_ctx": 16 }),
                    max_model_len: Some(16),
                },
                Listed {
                    id: "e.gguf",
                    meta: json!({}),
                    max_model_len: None,
                },
            ],
            7,
        );
        assert_eq!(v["data"][0]["id"], "m.gguf");
        assert_eq!(v["data"][0]["max_model_len"], 16);
        assert_eq!(v["data"][1]["id"], "e.gguf");
        assert!(v["data"][1].get("max_model_len").is_none());
        assert_eq!(
            v["models"],
            json!([
                { "name": "m.gguf", "model": "m.gguf" },
                { "name": "e.gguf", "model": "e.gguf" },
            ])
        );
    }
}
