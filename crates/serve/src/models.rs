//! `GET /v1/models` (and `/models`): the models a server serves, in llama.cpp's server's shape:
//! `data`, OpenAI's list, and `models` beside it (`tools/server/server-context.cpp` at `a4cb4c61`,
//! `get_res_models`: its `name` and `model`, the entry's other fields there being placeholders).
//! Every server of this crate answers through [`listing`].

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

#[cfg(test)]
mod tests {
    use super::*;

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
