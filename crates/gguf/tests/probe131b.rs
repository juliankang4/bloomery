//! FAIL-first probe (not part of the change): the private-type entry and the writer seed.

use gguf::write::{Layout, TensorDecl, WriteError};
use gguf::{GgmlType, Gguf, LoadError, PrivateType, Weights};

/// One tensor `q`, dims [512, 8], private id 1011, offset 0, then its 1,760 bytes.
fn tagged(path: &std::path::Path) {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(b"q");
    b.extend_from_slice(&2u32.to_le_bytes());
    for d in [512u64, 8] {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&1011u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b.extend(std::iter::repeat_n(7u8, 1760));
    std::fs::write(path, b).unwrap();
}

#[test]
fn probe_private() {
    let p = std::env::temp_dir().join(format!("probe131b-{}.gguf", std::process::id()));
    tagged(&p);
    let ok = [PrivateType { id: 1011, blck: 256, size: 110 }];
    let fits = Gguf::open_private(&p, Weights::Mapped { populate: false }, &ok).map(|g| g.tensor(0).unwrap().nbytes);
    println!("open_private size 110: {fits:?}");
    let huge = [PrivateType { id: 1011, blck: 256, size: u64::MAX / 2 }];
    let r = Gguf::open_private(&p, Weights::Mapped { populate: false }, &huge);
    let shown = r.as_ref().map(|g| g.tensor(0).unwrap().nbytes).map_err(|e| e.to_string());
    println!("open_private size u64::MAX/2: {shown:?}");
    std::fs::remove_file(&p).unwrap();
    assert_eq!(fits.ok(), Some(1760));
    assert!(
        matches!(r, Err(LoadError::TooLarge { ref name, ty, .. }) if name == "q" && ty == GgmlType::Unknown(1011)),
        "a size past u64 must be refused by name"
    );
}

#[test]
fn probe_writer() {
    let t = TensorDecl { name: "wide".to_string(), dims: vec![(1 << 62) + 1, 1], type_id: 0, nbytes: 4 };
    let r = Layout::new(&[], vec![t]);
    println!("Layout::new: {:?}", r.as_ref().map(|l| l.file_len()).map_err(|e| e.to_string()));
    assert!(matches!(r, Err(WriteError::TooLarge { ref name }) if name == "wide"));
}
