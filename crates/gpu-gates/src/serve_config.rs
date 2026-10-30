//! `--config FILE`: a server start's settings in one JSON object, each read as the flag it names. A
//! key is a flag's name without its dashes (`"ctx-size": 8192` is `--ctx-size 8192`); a string or a
//! number is the flag's value, `true` gives a switch, and a list gives the flag once for each value
//! (`"api-key": ["a", "b"]`). A path in a value is read as it is on the command line, from the
//! directory the server starts in.
//!
//! The file's flags go in front of the command line's, and a flag the command line gives in any of
//! its spellings replaces the file's entry for it (`-c` replaces `ctx-size`), as does a flag that
//! sets the same thing another way: `-m`, `--model-file` and `--hf` are one model, `--embd-model`
//! and `--embd-hf` one embedding model, `--embd-mmproj` and `--embd-no-mmproj` (as `--mmproj` and
//! `--no-mmproj`) one projector, `--cpu-moe`, `-ncmoe` and `-ot` one place for the routed experts,
//! and `--place` and a flag that names cards one set of cards. What reads the arguments after that
//! reads the file's flags as the command line's: they replace llama-server's variables and the
//! levers they stand for. A line on stderr says what each entry was read as, with the value only for
//! a flag of [`SEAT_FLAGS`] that holds no key.
//!
//! Refused by name, with the file's path, before anything else is read: a file that is no JSON
//! object, a key named twice, a key spelled with its dashes, a key that asks the start to do
//! something rather than set it (`plan`, `levers` and the other answers of the command line), a
//! value of another type (`false`, `null`, an object, an empty list), a value for a switch, `true`
//! for a flag that takes a value, and `--config` given twice. A key no seat takes is the flag no
//! seat takes, refused by the seat's own check before the fetch.

use serde_json::Value;

use crate::GateError;
use crate::llama_launch::{self, Meet, SEAT_FLAGS};

const FLAG: &str = "--config";

/// The flags that name the model source.
pub const SOURCE_FLAGS: &[&str] = &["-m", "--model-file", "--hf"];

/// Flags that set the same startup setting in different ways.
const CONFIG_GROUPS: &[&[&str]] = &[
    SOURCE_FLAGS,
    &["--embd-model", "--embd-hf"],
    &["--embd-mmproj", "--embd-no-mmproj"],
];

/// `args` with the settings of the file `--config` names joined in, its lines printed on stderr
/// under `bin`; `args` as they are when no `--config` is given.
pub fn from_file(bin: &str, args: &[String]) -> Result<Vec<String>, GateError> {
    let (args, notes) = joined(args, |path| std::fs::read_to_string(path))?;
    for line in notes {
        eprintln!("{bin}: {line}");
    }
    Ok(args)
}

/// [`from_file`]'s arguments and lines, the file's text read by `read`.
pub(crate) fn joined(
    args: &[String],
    read: impl FnOnce(&str) -> std::io::Result<String>,
) -> Result<(Vec<String>, Vec<String>), GateError> {
    let (given, rest) = llama_launch::take_serve_args(args, &[FLAG]);
    let path = match given.as_slice() {
        [] => return Ok((args.to_vec(), Vec::new())),
        [_, path] => path.as_str(),
        [_] => return Err(format!("{FLAG} needs a value").into()),
        _ => {
            return Err(format!("{FLAG} is given twice: one file holds a start's settings").into());
        }
    };
    let named = |why: String| -> GateError { format!("{FLAG} {path}: {why}").into() };
    let text = read(path).map_err(|e| named(e.to_string()))?;
    let Value::Object(settings) =
        serde_json::from_str::<Value>(&text).map_err(|e| named(e.to_string()))?
    else {
        return Err(named(
            "the settings are one JSON object, each key a flag's name".into(),
        ));
    };
    serde_json::from_str::<serve::Keys>(&text).map_err(|e| named(e.to_string()))?;
    let on_the_line: Vec<&str> = llama_launch::serve_args(&rest)
        .map(|(flag, _)| flag.as_str())
        .collect();
    let file_variant = settings.get("model").is_some_and(|value| {
        let value = match value {
            Value::Array(list) => list.last(),
            value => Some(value),
        };
        value
            .and_then(Value::as_str)
            .is_some_and(|v| v.contains(':'))
    });
    let line_variant = llama_launch::serve_args(&rest)
        .filter(|(flag, _)| flag.as_str() == "--model")
        .last()
        .and_then(|(_, value)| value)
        .is_some_and(|v| v.contains(':'));
    let one = |a: &str, b: &str| {
        llama_launch::one_setting(a, b)
            || CONFIG_GROUPS
                .iter()
                .any(|g| g.contains(&a) && g.contains(&b))
            || (a == "--model" && line_variant && SOURCE_FLAGS.contains(&b))
            || (b == "--model" && file_variant && SOURCE_FLAGS.contains(&a))
    };
    let mut out = Vec::new();
    let mut notes = Vec::new();
    for (key, value) in &settings {
        let flag = format!("--{key}");
        let entry = entry(key, &flag, value).map_err(named)?;
        if let Some(given) = on_the_line.iter().find(|given| one(given, &flag)) {
            notes.push(format!(
                "{path}: {key} is set, and the command line's {given} replaces it"
            ));
            continue;
        }
        let shown = if shows_value(&flag) {
            entry.join(" ")
        } else {
            format!("{flag} (value not printed)")
        };
        notes.push(format!("{path}: {key} is read as {shown}"));
        out.extend(entry);
    }
    out.extend(rest);
    Ok((out, notes))
}

/// The arguments the file's `key` gives: its `flag` once for each value, each with its value unless
/// the flag is a switch; or why the entry is no setting. A refusal names the value's type, never
/// the value.
fn entry(key: &str, flag: &str, value: &Value) -> Result<Vec<String>, String> {
    if key.is_empty() || key.starts_with('-') {
        return Err(format!(
            "the key {key:?} is no flag's name: a key is the name without its dashes, as \"ctx-size\" is \
             --ctx-size"
        ));
    }
    if flag == FLAG || llama_launch::SERVE_ACTIONS.contains(&flag) {
        return Err(format!(
            "{key} asks the start to do something, and the file holds settings: give {flag} on the command line"
        ));
    }
    let switch = !llama_launch::serve_takes_value(flag);
    let values = match value {
        Value::Array(list) if list.is_empty() => {
            return Err(format!(
                "{key} is an empty list, which gives {flag} no value"
            ));
        }
        Value::Array(list) => list.iter().collect(),
        one => vec![one],
    };
    let mut out = Vec::new();
    for value in values {
        out.push(flag.to_owned());
        match (value, switch) {
            (Value::Bool(true), true) => {}
            (Value::String(text), false) => out.push(text.clone()),
            (Value::Number(n), false) => out.push(n.to_string()),
            (Value::Bool(true), false) => {
                return Err(format!(
                    "{key} is true, and {flag} is no switch: give it its value"
                ));
            }
            (Value::String(_) | Value::Number(_), true) => {
                return Err(format!(
                    "{key} is {}, and {flag} is a switch: true gives it, and a switch that is off is left out",
                    kind(value)
                ));
            }
            (other, _) => {
                return Err(format!(
                    "{key} is {}: a setting is a string, a number, true for a switch, or a list of them",
                    kind(other)
                ));
            }
        }
    }
    Ok(out)
}

/// Whether a line may print `flag`'s value: a flag the seat table reads that holds no key. Any other
/// key, a misspelled `api_key` among them, prints only its flag.
fn shows_value(flag: &str) -> bool {
    SEAT_FLAGS
        .iter()
        .any(|read| read.meet != Meet::AddSecret && read.spelled.contains(&flag))
}

/// `value`'s type as a refusal names it.
fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(true) => "true",
        Value::Bool(false) => "false",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::joined;

    fn line(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_owned).collect()
    }

    fn with(text: &str, args: &str) -> Result<(Vec<String>, Vec<String>), String> {
        joined(&line(args), |path| {
            assert_eq!(path, "s.json");
            Ok(text.to_owned())
        })
        .map_err(|e| e.to_string())
    }

    #[test]
    fn the_command_line_replaces_the_files_settings_and_each_entry_has_a_line() {
        let text = r#"{
            "model": "qwen38",
            "model-file": "/m/folder",
            "ctx-size": 65536,
            "parallel": 2,
            "no-mmproj": true,
            "api-key": ["k1", "k2"],
            "api_key": "s3cret",
            "device": "CUDA0",
            "alias": "served"
        }"#;
        let (args, notes) = with(
            text,
            "--hf o/r --config s.json -c 4096 -dev CUDA1 --alias --plan",
        )
        .unwrap();
        assert_eq!(
            args,
            line(
                "--model qwen38 --parallel 2 --no-mmproj --api-key k1 --api-key k2 --api_key s3cret \
                 --hf o/r -c 4096 -dev CUDA1 --alias --plan"
            )
        );
        assert_eq!(
            notes,
            [
                "s.json: model is read as --model (value not printed)",
                "s.json: model-file is set, and the command line's --hf replaces it",
                "s.json: ctx-size is set, and the command line's -c replaces it",
                "s.json: parallel is read as --parallel 2",
                "s.json: no-mmproj is read as --no-mmproj",
                "s.json: api-key is read as --api-key (value not printed)",
                "s.json: api_key is read as --api_key (value not printed)",
                "s.json: device is set, and the command line's -dev replaces it",
                "s.json: alias is set, and the command line's --alias replaces it",
            ]
        );
        let none = line("--model qwen38 --alias --config");
        assert_eq!(
            with("{}", "--model qwen38 --alias --config").unwrap(),
            (none, vec![])
        );
    }

    #[test]
    fn a_file_entry_that_is_no_setting_is_refused_by_name() {
        let refused = |text: &str, args: &str| with(text, args).expect_err(text);
        assert_eq!(refused("{}", "--config"), "--config needs a value");
        let twice = refused("{}", "--config s.json --config s.json");
        assert!(twice.starts_with("--config is given twice"), "{twice}");
        for (text, refusal) in [
            ("[]", "the settings are one JSON object"),
            ("{\"ctx-size\": 1,}", "trailing comma"),
            (
                r#"{"ctx-size": 1, "ctx-size": 2}"#,
                "ctx-size is named twice",
            ),
            (
                r#"{"--ctx-size": 1}"#,
                "the key \"--ctx-size\" is no flag's name",
            ),
            (
                r#"{"no-mmproj": false}"#,
                "no-mmproj is false: a setting is",
            ),
            (r#"{"ctx-size": null}"#, "ctx-size is null: a setting is"),
            (
                r#"{"ctx-size": {"n": 1}}"#,
                "ctx-size is an object: a setting is",
            ),
            (r#"{"ctx-size": [[1]]}"#, "ctx-size is a list: a setting is"),
            (r#"{"api-key": []}"#, "api-key is an empty list"),
            (
                r#"{"alias": true}"#,
                "alias is true, and --alias is no switch",
            ),
            (
                r#"{"no-mmproj": "x"}"#,
                "no-mmproj is a string, and --no-mmproj is a switch",
            ),
        ] {
            // An entry the command line replaces is checked too: the file is read whole.
            for args in ["--config s.json", "--config s.json -c 1 --no-mmproj"] {
                let e = refused(text, args);
                assert!(
                    e.starts_with(&format!("--config s.json: {refusal}")),
                    "{args}: {e}"
                );
            }
        }
        for key in [
            "config",
            "help",
            "version",
            "list-devices",
            "levers",
            "records-schema",
            "recommend",
            "dry-run",
            "json",
            "plan",
        ] {
            let e = refused(&format!("{{\"{key}\": true}}"), "--config s.json");
            let refusal = format!("--config s.json: {key} asks the start to do something");
            assert!(e.starts_with(&refusal), "{e}");
        }
        let e = joined(&line("--config s.json"), |_| {
            Err(std::io::Error::other("no such file"))
        })
        .unwrap_err();
        assert_eq!(e.to_string(), "--config s.json: no such file");
    }

    /// A flag that sets the file's thing another way replaces the entry, in both directions; a flag
    /// that sets another thing leaves it.
    #[test]
    fn a_flag_that_sets_the_same_thing_another_way_replaces_the_entry() {
        for (entry, given) in [
            (r#""model-file": "/m""#, "--hf o/r"),
            (r#""hf": "o/r""#, "-m /m"),
            (r#""model": "glm:Q8_0""#, "-m /x.gguf"),
            (r#""model": ["qwen38", "glm:Q8_0"]"#, "--hf o/r"),
            (r#""model-file": "/x.gguf""#, "--model glm:Q8_0"),
            (r#""hf": "o/r""#, "--model glm:Q8_0"),
            (r#""embd-model": "/e.gguf""#, "--embd-hf x/y"),
            (r#""embd-hf": "x/y""#, "--embd-model /e.gguf"),
            (r#""mmproj": "/p.gguf""#, "--no-mmproj"),
            (r#""no-mmproj": true"#, "-mm /p.gguf"),
            (r#""embd-mmproj": "/p.gguf""#, "--embd-no-mmproj"),
            (r#""embd-no-mmproj": true"#, "--embd-mmproj /p.gguf"),
            (r#""cpu-moe": true"#, "-ncmoe 3"),
            (r#""n-cpu-moe": 3"#, "-ot exps=CPU"),
            (r#""override-tensor": "exps=CPU""#, "-cmoe"),
            (r#""device": "CUDA0""#, "--place bp"),
            (r#""place": "bp""#, "-ts 1,1"),
        ] {
            let (args, notes) =
                with(&format!("{{{entry}}}"), &format!("--config s.json {given}")).unwrap();
            assert_eq!(args, line(given), "{entry}");
            let flag = given.split(' ').next().unwrap();
            assert!(
                notes[0].ends_with(&format!("the command line's {flag} replaces it")),
                "{notes:?}"
            );
        }
        for (entry, given) in [
            (r#""model": "glm""#, "-m /x.gguf"),
            (r#""model-file": "/x.gguf""#, "--model glm"),
            (r#""tensor-split": "1,1""#, "-dev CUDA0"),
            (r#""n-gpu-layers": 99"#, "--place bp"),
            (r#""embd-ctx": 8"#, "-c 8"),
            (r#""embd-mmproj": "/p.gguf""#, "--no-mmproj"),
        ] {
            let (args, _) =
                with(&format!("{{{entry}}}"), &format!("--config s.json {given}")).unwrap();
            assert_eq!(args.len(), 2 + line(given).len(), "{entry}: {args:?}");
        }
    }
}
