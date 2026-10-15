use super::{ARCH, FAMILIES, MODEL, PROFILE};
use crate::RefError;
use crate::emb2::tests::{check, header, remove, set_dir};
use crate::family::Identity;

#[test]
fn text_families_take_their_sets_without_a_projector() -> Result<(), RefError> {
    let dir = set_dir("v1-own")?;
    for family in FAMILIES {
        let Identity::Emb2(kind, _) = family.identity else {
            panic!("embedding family")
        };
        assert_eq!(
            check(&dir, family, &header(&PROFILE, ARCH, kind, &[]), true)?,
            MODEL,
            "{}",
            family.name
        );
    }
    remove(&dir)
}
