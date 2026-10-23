#![cfg(unix)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;

use bloomery_gpu_gates::model_file;
use hf::source::Flags;

#[test]
fn checkpoint_support_is_checked_before_download() {
    if let Ok(case) = std::env::var("HF_SOURCE_CASE") {
        let flags = Flags {
            hf: Some("fixture/checkpoint".into()),
            ..Default::default()
        };
        let result = match case.as_str() {
            "name" => model_file::name(&flags),
            "resolve" => model_file::resolve(&flags, None).map(|_| ()),
            _ => panic!("unknown scenario: {case}"),
        };
        assert!(
            result
                .expect_err("checkpoint support is absent")
                .to_string()
                .contains("does not support safetensors checkpoints")
        );
        return;
    }

    let dir = std::env::temp_dir().join(format!("bloomery-model-sources-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let bin = dir.join("bin");
    std::fs::create_dir(&bin).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let files = [
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    let body = b"tiny fixture";
    let file = dir.join("body");
    std::fs::write(&file, body).unwrap();
    let digest = hf::fetch::digest(&file, hf::Check::GitSha1, body.len() as u64).unwrap();
    let listing = format!(
        "[{}]",
        files
            .iter()
            .map(|path| format!(
                r#"{{"type":"file","path":"{path}","size":{},"oid":"{digest}"}}"#,
                body.len()
            ))
            .collect::<Vec<_>>()
            .join(",")
    );
    let server = std::thread::spawn(move || {
        let mut requests = Vec::new();
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(10)))
                .unwrap();
            let mut request = [0; 4096];
            let n = stream.read(&mut request).unwrap();
            if n == 0 {
                break;
            }
            let request = std::str::from_utf8(&request[..n]).unwrap();
            let path = request.split_whitespace().nth(1).unwrap();
            requests.push(path.to_owned());
            let content = if path.starts_with("/api/models/") {
                listing.as_bytes()
            } else {
                body
            };
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                content.len()
            )
            .unwrap();
            stream.write_all(content).unwrap();
        }
        requests
    });
    let curl = bin.join("curl");
    std::fs::write(
        &curl,
        format!(
            r#"#!/bin/bash
args=()
for arg; do
    case "$arg" in
        https://huggingface.co/*) arg="http://{address}/${{arg#https://huggingface.co/}}" ;;
    esac
    args+=("$arg")
done
exec /usr/bin/curl "${{args[@]}}"
"#
        ),
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&curl, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut outputs = Vec::new();
    for case in ["name", "resolve"] {
        let cache = dir.join(case);
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "checkpoint_support_is_checked_before_download",
                "--nocapture",
            ])
            .env("HF_SOURCE_CASE", case)
            .env("PATH", &bin)
            .env("BLOOMERY_CACHE", &cache)
            .env("HOME", &dir)
            .env("HF_HUB_OFFLINE", "0")
            .env_remove("HF_TOKEN")
            .output()
            .unwrap();
        outputs.push((case, output));
    }
    drop(TcpStream::connect(address).unwrap());
    let requests = server.join().unwrap();
    let shards = requests.iter().filter(|p| p.ends_with(".safetensors")).count();
    println!("listing GETs: {}, shard GETs: {shards}", requests.iter().filter(|p| p.starts_with("/api/models/")).count());
    assert_eq!(shards, 0, "{requests:?}");
    assert_eq!(requests.len(), 2, "only the listings are requested");
    for (case, output) in outputs {
        assert!(
            output.status.success(),
            "{case}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!dir.join(case).join("fixture/checkpoint").exists());
    }
    std::fs::remove_dir_all(dir).unwrap();
}
