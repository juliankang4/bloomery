#![cfg(unix)]

use bloomery_gpu_gates::model_file::{resolve, sentence_prompts};
use hf::source::Flags;

const CONFIG: &str = "config_sentence_transformers.json";

/// The child half: the parent reruns this test inside a fixture directory holding `scenario`.
fn sentence_prompts_startup(case: &str) {
    let flags = Flags {
        hf: Some("fixture/quant:Q4_K_M".into()),
        ..Default::default()
    };
    let model = resolve(&flags, None, hf::fetch::Sources::Gguf)
        .unwrap()
        .unwrap();
    let result = sentence_prompts(&flags, &model);
    match case {
        "base" | "own" => {
            let prompts = result.unwrap();
            assert_eq!(prompts["query"], "query: ");
            assert_eq!(prompts["document"], "document: ");
        }
        "malformed" => assert!(result.unwrap_err().to_string().contains(CONFIG)),
        "no-card" | "no-base-config" => assert!(result.unwrap().is_empty()),
        _ => panic!("unknown scenario: {case}"),
    }
}

#[test]
fn cached_sentence_prompts_survive_an_offline_restart() {
    if let Ok(case) = std::fs::read_to_string("scenario") {
        return sentence_prompts_startup(&case);
    }
    let dir = std::env::temp_dir().join(format!(
        "bloomery-offline-sentence-prompts-{}",
        std::process::id()
    ));
    std::fs::create_dir(&dir).unwrap();
    let bin = dir.join("bin");
    std::fs::create_dir(&bin).unwrap();
    let curl = bin.join("curl");
    use std::os::unix::fs::PermissionsExt;
    for case in ["base", "own", "no-card", "no-base-config", "malformed"] {
        let root = dir.join(case);
        let hub = root.join("hub");
        let cache = root.join("cache");
        let config = r#"{"prompts":{"query":"query: ","document":"document: "}}"#;
        let mut files = vec![("fixture/quant", "model-Q4_K_M.gguf", "GGUF")];
        if case != "no-card" {
            files.push((
                "fixture/quant",
                "README.md",
                "---\nbase_model: fixture/base\n---\n",
            ));
        }
        match case {
            "own" => files.push(("fixture/quant", CONFIG, config)),
            "base" => files.push(("fixture/base", CONFIG, config)),
            "malformed" => files.push(("fixture/base", CONFIG, r#"{"prompts":{"query":7}}"#)),
            _ => {}
        }
        let mut listings = std::collections::BTreeMap::<&str, Vec<String>>::new();
        listings.insert("fixture/base", Vec::new());
        for (repo, name, text) in files {
            let path = hub.join(repo).join("resolve/main").join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, text).unwrap();
            let digest = hf::fetch::digest(&path, hf::Check::GitSha1, text.len() as u64).unwrap();
            listings.entry(repo).or_default().push(format!(
                r#"{{"type":"file","path":"{name}","size":{},"oid":"{digest}"}}"#,
                text.len()
            ));
        }
        for (repo, entries) in listings {
            let tree = hub.join("api/models").join(repo).join("tree");
            std::fs::create_dir_all(&tree).unwrap();
            std::fs::write(tree.join("main"), format!("[{}]", entries.join(","))).unwrap();
        }
        std::fs::write(root.join("scenario"), case).unwrap();
        for mode in ["online", "cached", "offline", "down"] {
            let script = if mode == "down" {
                "#!/bin/bash\nexit 7\n"
            } else {
                r#"#!/bin/bash
hub=$(/bin/pwd -P)/hub
args=()
for arg; do
    case "$arg" in
        https://huggingface.co/*) arg="file://$hub/${arg#https://huggingface.co/}" ;;
    esac
    args+=("$arg")
done
exec /usr/bin/curl "${args[@]}"
"#
            };
            std::fs::write(&curl, script).unwrap();
            std::fs::set_permissions(&curl, std::fs::Permissions::from_mode(0o755)).unwrap();
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "cached_sentence_prompts_survive_an_offline_restart",
                    "--nocapture",
                ])
                .current_dir(&root)
                .env("PATH", &bin)
                .env("BLOOMERY_CACHE", &cache)
                .env("HOME", &root)
                .env("HF_HUB_OFFLINE", if mode == "offline" { "1" } else { "0" })
                .env_remove("HF_TOKEN")
                .output()
                .unwrap();
            assert!(
                output.status.success()
                    && String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"),
                "{case}/{mode}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            println!("sentence-prompts {case}/{mode}: passed");
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
