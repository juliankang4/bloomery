use super::{
    ARCH, LCPP_BUILD, MMPROJ_BF16, MMPROJ_BF16_SHA256, MODEL, MODEL_SHA256, SERVER, TEXT, TEXT_CPU,
    TOWER, TOWER_CPU,
};
use crate::RefError;
use crate::emb2::Kind;
use crate::family::Family;
use std::path::{Path, PathBuf};

const Q8_MMPROJ: &str = "/models/embeddinggemma-2/mmproj-embeddinggemma-2-Q8_0.gguf";
const Q8_MMPROJ_SHA256: &str = "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f";

/// A fresh directory for one test's set; tests of one process run in
/// parallel, so each passes its own `what`.
fn set_dir(what: &str) -> Result<PathBuf, RefError> {
    Ok(crate::clefvis::testkit::scratch(&format!("emb2-{what}")))
}

fn remove(dir: &Path) -> Result<(), RefError> {
    std::fs::remove_dir_all(dir).map_err(|e| RefError::missing(dir, e.to_string()))
}

/// The header of a `kind` set as tools/ref/emb2 writes it, one line a
/// `(key, value)` pair; the arguments of `with` replace or (with an empty
/// value) drop the line of their key.
fn header(kind: Kind, with: &[(&str, &str)]) -> Vec<(String, String)> {
    let device = format!(
        "{}\tthreads\t16\tcard\t{}",
        kind.device(),
        if kind.device() == "cpu" {
            "cpu"
        } else {
            "NVIDIA GB10"
        }
    );
    let mmproj = format!("{MMPROJ_BF16}\tsha256\t{MMPROJ_BF16_SHA256}");
    let mut h: Vec<(&str, String)> = match kind {
        Kind::Text | Kind::TextCpu => vec![
            ("model", MODEL.into()),
            ("build", LCPP_BUILD.into()),
            ("arch", ARCH.into()),
            ("emb2", "text".into()),
            ("flags", format!("text -m {MODEL} --case en_short -t 16")),
            ("device", device),
            ("model_sha256", MODEL_SHA256.into()),
        ],
        Kind::Tower | Kind::TowerCpu => vec![
            ("model", MMPROJ_BF16.into()),
            ("build", LCPP_BUILD.into()),
            ("arch", "clip".into()),
            ("emb2", "tower".into()),
            (
                "flags",
                format!("tower -m {MODEL} --mmproj {MMPROJ_BF16} -t 16"),
            ),
            ("device", device),
            ("text_model", format!("{MODEL}\tsha256\t{MODEL_SHA256}")),
            ("mmproj", mmproj),
        ],
        Kind::Server => vec![
            ("model", MODEL.into()),
            ("build", LCPP_BUILD.into()),
            ("arch", ARCH.into()),
            ("emb2", "server".into()),
            ("device", device.replace("threads\t16", "threads\t-")),
            ("model_sha256", MODEL_SHA256.into()),
            ("mmproj", mmproj),
        ],
    };
    for &(k, v) in with {
        h.retain(|&(key, _)| key != k);
        if !v.is_empty() {
            h.push((k, v.to_string()));
        }
    }
    h.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// `family`'s check of a set at `dir` with these header lines, one tensor row
/// and the trailer when `complete`.
fn check(
    dir: &Path,
    family: &Family,
    header: &[(String, String)],
    complete: bool,
) -> Result<String, RefError> {
    let mut lines = vec![
        "# dump_emb2: llama.cpp mainline EmbeddingGemma 2, raw f32, little-endian".to_string(),
    ];
    lines.extend(header.iter().map(|(k, v)| format!("# {k}\t{v}")));
    lines.push(
        "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1"
            .to_string(),
    );
    let name = if matches!(
        family.identity,
        crate::family::Identity::Emb2(Kind::Tower | Kind::TowerCpu, _)
    ) {
        "small/layer_out-0"
    } else {
        "en_short/pooled"
    };
    lines.push(format!(
        "tensor\t{name}\t0\tf32\t768\t1\t1\t1\t3072\t0\tMEAN\t1\t0\t-\t-"
    ));
    if complete {
        lines.push("# complete\t1\t0".to_string());
    }
    let path = dir.join("MANIFEST.tsv");
    std::fs::write(&path, lines.join("\n") + "\n")
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    family.check_set(dir).map(|p| p.dumped_from)
}

/// A kind of set, the family checking it, the header lines replaced and the
/// text the stale error names.
type FileCase<'a> = (Kind, &'a Family, Vec<(&'a str, &'a str)>, &'a str);

/// A kind of set, the family checking it, the header line replaced, whether
/// the trailer is written, and the text the error names.
type RefusedCase<'a> = (Kind, &'a Family, (&'a str, &'a str), bool, &'a str);

const ALL: [(Kind, &Family); 5] = [
    (Kind::Text, &TEXT),
    (Kind::TextCpu, &TEXT_CPU),
    (Kind::Tower, &TOWER),
    (Kind::TowerCpu, &TOWER_CPU),
    (Kind::Server, &SERVER),
];

/// Each family takes a set of its own kind and device, of the pinned files,
/// commit and architecture, and states the file the set names as `# model`:
/// the text file, or the projector for a tower set.
#[test]
fn each_family_takes_a_set_of_its_kind() -> Result<(), RefError> {
    let dir = set_dir("own")?;
    for (kind, family) in ALL {
        let want = if matches!(kind, Kind::Tower | Kind::TowerCpu) {
            MMPROJ_BF16
        } else {
            MODEL
        };
        assert_eq!(
            check(&dir, family, &header(kind, &[]), true)?,
            want,
            "{kind:?}"
        );
    }
    remove(&dir)
}

/// A set of another kind or device is foreign to the family and names the
/// line: the CPU twin's set at the card set's place and the other way round,
/// and the server's vectors at the text set's.
#[test]
fn a_set_of_another_kind_or_device_is_foreign() -> Result<(), RefError> {
    let dir = set_dir("kind")?;
    let cases: [(Kind, &Family, &str); 5] = [
        (Kind::TextCpu, &TEXT, "device \"cpu\""),
        (Kind::Text, &TEXT_CPU, "device \"cuda\""),
        (Kind::TowerCpu, &TOWER, "device \"cpu\""),
        (Kind::Tower, &TOWER_CPU, "device \"cuda\""),
        (Kind::Server, &TEXT, "emb2 \"server\""),
    ];
    for (kind, family, want) in cases {
        match check(&dir, family, &header(kind, &[]), true) {
            Err(e @ RefError::Foreign { .. }) => {
                assert!(e.to_string().contains(want), "{kind:?}: {e}")
            }
            r => panic!("a {kind:?} set as {}: {r:?}", family.name),
        }
    }
    remove(&dir)
}

/// A set of another file is stale and names it: the bf16 text file, a text
/// file of another sha256, the Q8_0 projector (as a tower set's `# model` and
/// as a server set's `# mmproj`), and the bf16 projector of another sha256.
#[test]
fn a_set_of_another_file_is_stale() -> Result<(), RefError> {
    let dir = set_dir("file")?;
    let bf16 = "/models/embeddinggemma-2/embeddinggemma-2-BF16.gguf";
    let other_sha = "0".repeat(64);
    let q8_line = format!("{Q8_MMPROJ}\tsha256\t{Q8_MMPROJ_SHA256}");
    let path_line = format!("{Q8_MMPROJ}\tsha256\t{MMPROJ_BF16_SHA256}");
    let sha_line = format!("{MMPROJ_BF16}\tsha256\t{other_sha}");
    let text_line = format!("{bf16}\tsha256\t{MODEL_SHA256}");
    let cases: Vec<FileCase> = vec![
        (Kind::Text, &TEXT, vec![("model", bf16)], bf16),
        (
            Kind::TextCpu,
            &TEXT_CPU,
            vec![("model_sha256", &other_sha)],
            &other_sha,
        ),
        (
            Kind::Tower,
            &TOWER,
            vec![("model", Q8_MMPROJ), ("mmproj", &q8_line)],
            Q8_MMPROJ,
        ),
        (
            Kind::TowerCpu,
            &TOWER_CPU,
            vec![("mmproj", &sha_line)],
            &other_sha,
        ),
        (Kind::Tower, &TOWER, vec![("text_model", &text_line)], bf16),
        (Kind::Server, &SERVER, vec![("mmproj", &q8_line)], Q8_MMPROJ),
        (
            Kind::Server,
            &SERVER,
            vec![("mmproj", &path_line)],
            Q8_MMPROJ,
        ),
    ];
    for (kind, family, with, want) in cases {
        match check(&dir, family, &header(kind, &with), true) {
            Err(e @ RefError::Stale { .. }) => {
                assert!(e.to_string().contains(want), "{kind:?} {with:?}: {e}")
            }
            r => panic!("{kind:?} {with:?}: {r:?}"),
        }
    }
    remove(&dir)
}

/// A set of another commit or architecture, one without its trailer, and one
/// without the line its kind is read from are refused by name.
#[test]
fn a_set_of_another_build_arch_or_shape_is_refused() -> Result<(), RefError> {
    let dir = set_dir("build")?;
    let cases: [RefusedCase; 6] = [
        (Kind::Text, &TEXT, ("build", "9cf967a0"), true, "build"),
        (Kind::Tower, &TOWER, ("arch", ARCH), true, "arch"),
        (
            Kind::Server,
            &SERVER,
            ("build", LCPP_BUILD),
            false,
            "no `# complete` trailer",
        ),
        (Kind::Text, &TEXT, ("emb2", ""), true, "no `# emb2` line"),
        (
            Kind::TowerCpu,
            &TOWER_CPU,
            ("device", "cpu"),
            true,
            "want cuda|cpu",
        ),
        (
            Kind::Server,
            &SERVER,
            ("mmproj", ""),
            true,
            "no `# mmproj` line",
        ),
    ];
    for (kind, family, with, complete, want) in cases {
        let e = check(&dir, family, &header(kind, &[with]), complete)
            .expect_err("refused")
            .to_string();
        assert!(e.contains(want), "{kind:?} {with:?} {complete}: {e}");
    }
    remove(&dir)
}

/// A complete tower set needs a captured block, on either device.
#[test]
fn a_tower_without_block_taps_is_refused() -> Result<(), RefError> {
    let dir = set_dir("block-taps")?;
    for (kind, family) in [(Kind::Tower, &TOWER), (Kind::TowerCpu, &TOWER_CPU)] {
        check(&dir, family, &header(kind, &[]), true)?;
        let path = dir.join("MANIFEST.tsv");
        let manifest =
            std::fs::read_to_string(&path).map_err(|e| RefError::missing(&path, e.to_string()))?;
        std::fs::write(
            &path,
            manifest.replace("small/layer_out-0", "small/projected"),
        )
        .map_err(|e| RefError::missing(&path, e.to_string()))?;
        let e = family
            .check_set(&dir)
            .expect_err("no block tap")
            .to_string();
        assert!(e.contains("layer_out-0"), "{kind:?}: {e}");
    }
    remove(&dir)
}
