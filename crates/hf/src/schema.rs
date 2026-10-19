//! The keys a reader accepts in a checkpoint's JSON object, each with the values it allows. A key
//! the table does not name is refused by name, so a setting no reader honors cannot pass unread.

use serde_json::Value as Json;

/// What one key of an object may hold.
#[derive(Clone, Copy, Debug)]
pub enum Rule {
    /// Read and checked by the reader that owns the table.
    Read,
    /// Absent, or one of these values (a JSON array).
    Only(&'static str),
    /// Present, and one of these values (a JSON array).
    Need(&'static str),
    /// Any value of this type; the reason says why no result depends on it.
    Ignore(Type, &'static str),
}

/// The JSON type of an ignored key.
#[derive(Clone, Copy, Debug)]
pub enum Type {
    Bool,
    Number,
    String,
    Object,
}

impl Type {
    fn holds(self, v: &Json) -> bool {
        match self {
            Type::Bool => v.is_boolean(),
            Type::Number => v.is_number(),
            Type::String => v.is_string(),
            Type::Object => v.is_object(),
        }
    }
}

/// One table: each key and its rule.
pub type Fields = [(&'static str, Rule)];

/// Refuse, by `path` and key, a key of the object `value` that no table of `tables` names, and a
/// value its rule does not allow.
///
/// # Errors
///
/// `value` that is not an object, an unknown key, a value outside an `Only` or `Need` list, an
/// absent `Need` key, and an ignored key of another type.
pub fn check(value: &Json, path: &str, tables: &[&Fields]) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{path} is not an object"))?;
    let rule = |key: &str| {
        tables
            .iter()
            .flat_map(|t| t.iter())
            .find(|(k, _)| *k == key)
            .map(|&(_, r)| r)
    };
    for (key, v) in object {
        match rule(key).ok_or_else(|| format!("{path}.{key} is not a key this converter reads"))? {
            Rule::Read => {}
            Rule::Only(allowed) | Rule::Need(allowed) => allow(path, key, v, allowed)?,
            Rule::Ignore(ty, _) if ty.holds(v) => {}
            Rule::Ignore(ty, _) => return Err(format!("{path}.{key} {v} is not a {ty:?}")),
        }
    }
    for (key, rule) in tables.iter().flat_map(|t| t.iter()) {
        if matches!(rule, Rule::Need(_)) && !object.contains_key(*key) {
            return Err(format!("{path}.{key} is missing"));
        }
    }
    Ok(())
}

fn allow(path: &str, key: &str, v: &Json, allowed: &str) -> Result<(), String> {
    let allowed: Json = serde_json::from_str(allowed).expect("a schema's values are JSON");
    let list = allowed.as_array().expect("a schema's values are a list");
    if list.contains(v) {
        return Ok(());
    }
    Err(format!(
        "{path}.{key} {v} is unsupported; supported: {allowed}"
    ))
}
