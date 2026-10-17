//! Named input prefixes from a model's `config_sentence_transformers.json`.

use std::collections::BTreeMap;

use crate::HfError;

/// The configured prompts, without applying `default_prompt_name`.
pub fn prompts(text: &str) -> Result<BTreeMap<String, String>, HfError> {
    let config: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| HfError::Source(format!("config_sentence_transformers.json: {e}")))?;
    let config = config.as_object().ok_or_else(|| {
        HfError::Source("config_sentence_transformers.json: expected an object".into())
    })?;
    match config.get("prompts") {
        None => Ok(BTreeMap::new()),
        Some(value) => serde_json::from_value(value.clone()).map_err(|e| {
            HfError::Source(format!("config_sentence_transformers.json prompts: {e}"))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_keep_their_bytes_and_names() {
        let got = prompts(r#"{"prompts":{"query":"task: search result | query: ","document":"title: none | text: ","empty":"","unicode":"검색: "},"default_prompt_name":"query"}"#).unwrap();
        assert_eq!(got["query"], "task: search result | query: ");
        assert_eq!(got["document"], "title: none | text: ");
        assert_eq!(got["empty"], "");
        assert_eq!(got["unicode"], "검색: ");
        assert!(prompts("{}").unwrap().is_empty());
        for bad in [
            "null",
            "[]",
            "{",
            r#"{"prompts":null}"#,
            r#"{"prompts":[]}"#,
            r#"{"prompts":{"query":7}}"#,
        ] {
            assert!(
                prompts(bad)
                    .unwrap_err()
                    .to_string()
                    .contains("config_sentence_transformers.json")
            );
        }
    }
}
