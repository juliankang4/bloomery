//! The pinned mainline host preprocessor's mel bytes and tower token counts.

use refset::arch::gemma_embedding2::AUDIO_MEL;
use refset::ik::{RefManifest, ref_tensor_of_in};
use vision::arch::gemma4a::{MEL_BINS, Preprocessor};

#[test]
#[ignore = "reads the pinned mainline PCM and mel set"]
fn hw_mel_matches_mainline() {
    let dir = AUDIO_MEL.path(AUDIO_MEL.sets[0]);
    let man = RefManifest::open(&dir, &AUDIO_MEL).expect("gemma4a mel reference family");
    let prep = Preprocessor::default();
    let cases: Vec<_> = man
        .tensors
        .iter()
        .filter(|r| r.name.ends_with("/pcm"))
        .collect();
    assert_eq!(cases.len(), 13, "the full synthetic PCM set");
    let mut total = 0;
    for row in cases {
        let case = row.name.strip_suffix("/pcm").unwrap();
        row.expect(case, "f32", [row.ne[0], 1, 1, 1], "NONE")
            .unwrap();
        let pcm = ref_tensor_of_in(&dir, row).expect("PCM bytes");
        let chunks = prep.prepare(&pcm).unwrap();
        let expected = man
            .tensors
            .iter()
            .filter(|r| r.name.starts_with(&format!("{case}/")) && r.name.ends_with("/mel"))
            .count();
        assert_eq!(chunks.len(), expected, "{case}: chunk count");
        for (i, chunk) in chunks.iter().enumerate() {
            let name = format!("{case}/{i}");
            let row = man.tensor(&format!("{name}/mel"), 0).unwrap();
            row.expect(
                &name,
                "f32",
                [chunk.frames as u64, MEL_BINS as u64, 1, 1],
                "NONE",
            )
            .unwrap();
            let want = ref_tensor_of_in(&dir, row).expect("mel bytes");
            assert_eq!(chunk.data.len(), want.len(), "{name}: mel length");
            let wrong = chunk
                .data
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(wrong, 0, "{name}: mel bits");
            let row = man.tensor(&format!("{name}/tokens"), 0).unwrap();
            row.expect(&name, "f32", [1, 1, 1, 1], "NONE").unwrap();
            let tokens = ref_tensor_of_in(&dir, row).unwrap();
            assert_eq!(chunk.tokens() as f32, tokens[0], "{name}: tower tokens");
            println!(
                "{name}: {} mel values bit-identical, {} tokens",
                want.len(),
                chunk.tokens()
            );
            total += 1;
        }
    }
    assert_eq!(total, 14, "every chunk, including the one after 30 seconds");
}
