//! End-to-end GPU gate for both EmbeddingGemma encoders. With `--v1` it
//! reads v1's text, CPU-twin and SentencePiece sets; without arguments it
//! reads v2's sets. Each model is held to its pinned file digest. Every
//! case's ids are checked against the tokenizer set's count and digest
//! before a pass at the model's context, with taps read back.
//!
//! A row's band is [`TWIN`] times the CPU twin's own distance from the card
//! set, and at least [`FLOOR`]: the twin is the floor any two realizations of
//! the graph lie apart (the Qwen seats' rule, `shared/glm5next_media.rs`). A
//! row is within its band only when its distance and the twin's are finite.
//!
//! 1. (t) Per case: `inp_scaled` bit for bit (the host's token rows times
//!    `√hidden`, the one f32 multiply `ggml_scale` makes); v2's `inp_per_layer`,
//!    every layer's `l_out`, `result_norm`, the token rows against `embd`
//!    and the pooled vector ([`Pass::pooled`]) against `pooled`, each within its
//!    band. V1's layer outputs are replayed from the oracle's preceding residual
//!    (layer 0 from `inp_scaled`); free-running layer errors are diagnostic rows.
//!    Its `result_norm`, token rows and pooled vector remain free-running.
//!    V2's layer outputs remain free-running. A case longer than a sliding
//!    layer's reach plus one catches a dropped window; a set with no such case fails by name. The host test
//!    `key_bounds_are_the_symmetric_mask` holds the exact key edges; every
//!    case holds the global layers.
//! 2. (d) The longest case run again is bit for bit the first run: with no
//!    taps its pooled vector, with taps every position's row.
//! 3. (i) V2's image-only and `mix_free-448x448` inputs assembled by the serving helper,
//!    pooled and normalized against the server card set and its CPU twin.
//! 4. (r) Refused by name, the load untouched after each: a pass of no ids,
//!    of one id past the load's context and of an id past the vocabulary,
//!    and a load at one position past [`CARD_CTX`].
//! 5. (h) V2's normalized vectors against official sentence-transformers,
//!    with a frozen mainline-to-HF precision budget and independent ceiling.
//!    `--hf` runs only this clause, including the exact one-sided window check.

#[cfg(feature = "gemma-embedding2")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_gemma_embedding2_e2e", gate::run())
}

#[cfg(feature = "gemma-embedding2")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[path = "shared/emb2_image.rs"]
mod image;

#[cfg(feature = "gemma-embedding2")]
mod gate {
    use std::path::Path;
    use std::time::Instant;

    use crate::e2e::{clause, elapsed, refused_by, rel, same_bits, tap};
    use crate::image::open;
    use bloomery_gpu::Gpu;
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, data_dir, ref_model_path, verdict,
    };
    use bloomery_gpu_gemma_embedding2::{Encoder, Pass, card_bytes};
    use gguf::Split;
    use model::arch::gemma_embedding2::hparams::CARD_CTX;
    use refset::arch::gemma_embedding2::{
        HF, IDS_SET, MODEL_SHA256, REL_FLOOR as FLOOR, TEXT, TEXT_CPU, TWIN,
    };
    use refset::ik::RefHeader;
    use refset::tokenizer::read_ids;

    /// One case of the set: its name and its ids.
    struct Case {
        id: String,
        ids: Vec<u32>,
    }

    fn check_case_identities(card: &RefHeader, cpu: &RefHeader) -> Result<(), GateError> {
        if !card
            .other
            .iter()
            .filter(|line| line.starts_with("# case\t"))
            .eq(cpu.other.iter().filter(|line| line.starts_with("# case\t")))
        {
            return Err("CPU twin case identities differ from the card set".into());
        }
        Ok(())
    }

    /// The cases `man`'s `# case <id> <n> <path> <sha256>` lines name, each
    /// one's ids read from [`IDS_SET`] and refused by name when their count
    /// or their file's digest is not the line's.
    fn cases(man: &RefManifest, ids_set: &str) -> Result<Vec<Case>, GateError> {
        let dir = data_dir().join(ids_set);
        let mut out = Vec::new();
        for line in &man.header.other {
            let Some(rest) = line.strip_prefix("# case\t") else {
                continue;
            };
            let fields: Vec<&str> = rest.split('\t').collect();
            let [id, n, _, sha] = fields[..] else {
                return Err(
                    format!("{}: a case line of another shape: {line:?}", TEXT.name).into(),
                );
            };
            let path = dir.join(format!("{id}.ids"));
            let want = n
                .parse::<usize>()
                .map_err(|e| format!("{}: case {id}: token count {n:?}: {e}", TEXT.name))?;
            let ids = read_ids(&path, want)?;
            let digest = gguf::digest::sha256_file(&path)?;
            if digest != sha {
                return Err(format!(
                    "{}: {} ids of digest {digest}; the set's case line says {n} of {sha}",
                    path.display(),
                    ids.len()
                )
                .into());
            }
            out.push(Case {
                id: id.to_string(),
                ids,
            });
        }
        if out.is_empty() {
            return Err(format!("{}: no case line", TEXT.name).into());
        }
        Ok(out)
    }

    /// One row held to the card set: its distance, the twin's and the band.
    struct Held {
        r: f64,
        twin: f64,
        band: f64,
    }

    impl Held {
        fn of(ours: &[f32], card: &[f32], twin: &[f32]) -> Held {
            Self::distances(rel(ours, card), rel(twin, card))
        }

        fn distances(r: f64, twin: f64) -> Held {
            Held {
                r,
                twin,
                band: (TWIN * twin).max(FLOOR),
            }
        }

        /// Within the band, both distances finite: `rel` makes a NaN or a
        /// length mismatch infinite, and an infinite twin an infinite band.
        fn ok(&self) -> bool {
            self.r.is_finite() && self.twin.is_finite() && self.r <= self.band
        }

        fn line(&self, what: &str) -> String {
            format!(
                "{what} {:.3e} (twin {:.3e}, band {:.3e})",
                self.r, self.twin, self.band
            )
        }
    }

    /// Clause (t) for one case: its pass with taps, held tap by tap. The
    /// pass's rows, for (d).
    fn case(
        enc: &mut Encoder,
        c: &Case,
        man: &RefManifest,
        cpu: &RefManifest,
        n_layer: usize,
        teacher_forced: bool,
    ) -> Result<(bool, Pass), GateError> {
        let t0 = Instant::now();
        let pass = enc.pass(&c.ids, true)?;
        let taps = pass.taps.as_ref().ok_or("a pass with taps returned none")?;
        assert_eq!(taps.l_out.len(), n_layer, "missing l_out taps");
        let row = |m: &RefManifest, name: &str| tap(m, &format!("{}/{name}", c.id));
        let held = |name: &str, ours: &[f32]| -> Result<Held, GateError> {
            Ok(Held::of(ours, &row(man, name)?, &row(cpu, name)?))
        };
        let mut ok = true;
        let mut lines = Vec::new();
        let exact = same_bits(&taps.inp_scaled, &row(man, "inp_scaled")?);
        lines.push(format!("inp_scaled bits {}", verdict(exact)));
        ok &= exact;
        if !teacher_forced {
            let h = held("inp_per_layer", &taps.inp_per_layer)?;
            lines.push(h.line("inp_per_layer"));
            ok &= h.ok();
        }
        // PIN(2026-10-16): v1 taps use oracle inputs to isolate accumulated FA rounding;
        // final outputs stay free-running and all bands stay unchanged.
        let replay = if teacher_forced {
            let mut inputs = vec![row(man, "inp_scaled")?];
            for l in 1..n_layer {
                inputs.push(row(man, &format!("l_out-{}", l - 1))?);
            }
            Some(enc.replay_layers(&c.ids, &inputs)?)
        } else {
            None
        };
        let layer_taps = replay.as_ref().unwrap_or(taps);
        assert_eq!(layer_taps.l_out.len(), n_layer, "missing replay l_out taps");
        // The worst layer by its distance over its band.
        let mut worst = (0.0f64, String::new());
        for (l, ours) in layer_taps.l_out.iter().enumerate() {
            let h = held(&format!("l_out-{l}"), ours)?;
            if teacher_forced {
                let free = held(&format!("l_out-{l}"), &taps.l_out[l])?;
                println!(
                    "(layers) {} {l}: free {:.9e}, teacher {:.9e}, twin {:.9e}, band {:.9e}",
                    c.id, free.r, h.r, h.twin, h.band
                );
            }
            if !h.ok() {
                println!(
                    "(t) {}: {} past its band {}",
                    c.id,
                    h.line(&format!("layer {l} l_out")),
                    verdict(false)
                );
                ok = false;
            }
            if h.r / h.band >= worst.0 {
                worst = (h.r / h.band, h.line(&format!("l_out worst layer {l}")));
            }
        }
        lines.push(worst.1);
        for (what, ours) in [
            ("result_norm", &taps.result_norm),
            ("embd", &taps.result_embd),
            ("pooled", &pass.pooled),
        ] {
            let h = held(what, ours)?;
            lines.push(h.line(what));
            ok &= h.ok();
        }
        println!(
            "(t) {}: {} ids, {}; {:.1} ms (runtime value) {}",
            c.id,
            c.ids.len(),
            lines.join("; "),
            t0.elapsed().as_secs_f64() * 1e3,
            verdict(ok)
        );
        Ok((ok, pass))
    }

    /// Every position's `output` row of a pass with taps.
    fn rows(pass: &Pass) -> Result<&[f32], GateError> {
        let taps = pass.taps.as_ref().ok_or("a pass with taps returned none")?;
        Ok(&taps.result_embd)
    }

    /// Clause (d): `c` run again is bit for bit `first`, a pass with taps:
    /// with no taps its pooled vector, with taps its rows.
    fn rerun(enc: &mut Encoder, c: &Case, first: &Pass) -> Result<bool, GateError> {
        let plain = enc.pass(&c.ids, false)?;
        let tapped = enc.pass(&c.ids, true)?;
        let same =
            same_bits(&plain.pooled, &first.pooled) && same_bits(rows(&tapped)?, rows(first)?);
        println!(
            "(d) {}: {} ids, a rerun's pooled vector and rows bit for bit {}",
            c.id,
            c.ids.len(),
            verdict(same)
        );
        Ok(same)
    }

    /// A case long enough for a dropped window to leave the tap bands:
    /// some row's keys stop short of the input's ends. The host test holds
    /// the exact edges. No sliding layer passes; a set with no such case fails.
    fn window(reach: &[Option<u32>], cases: &[Case]) -> bool {
        let Some(reach) = reach.iter().flatten().copied().max().map(|r| r as usize) else {
            return true;
        };
        let longest = cases.iter().map(|c| c.ids.len()).max().unwrap_or(0);
        let held = longest > reach + 1;
        println!(
            "(t) the window: the longest case holds {longest} ids, a sliding layer reaches {reach} \
             on each side {}",
            verdict(held)
        );
        held
    }

    /// Clause (r): each refusal by its words, and a pass after them the
    /// first one's rows (`first`) bit for bit.
    fn refusals(
        enc: &mut Encoder,
        c: &Case,
        first: &[f32],
        vocab: usize,
    ) -> Result<bool, GateError> {
        const PASS: &str = "Encoder::pass";
        let past = vec![c.ids[0]; enc.ctx() + 1];
        let mut bad = c.ids.clone();
        bad[0] = u32::try_from(vocab)?;
        let checks = [
            (
                "no ids",
                refused_by(&enc.pass(&[], false), PASS, "0 positions"),
            ),
            (
                "one past the context",
                refused_by(&enc.pass(&past, false), PASS, "a pass takes 1 to"),
            ),
            (
                "an id past the vocabulary",
                refused_by(&enc.pass(&bad, false), PASS, "past the vocabulary"),
            ),
        ];
        let mut ok = true;
        for (what, refused) in checks {
            println!("(r) {what}: refused by name {}", verdict(refused));
            ok &= refused;
        }
        enc.pass(&vec![c.ids[0]; enc.ctx()], false)?;
        let again = enc.pass(&c.ids, true)?;
        let same = same_bits(rows(&again)?, first);
        println!(
            "(r) {}: after the refusals the rows bit for bit {}",
            c.id,
            verdict(same)
        );
        Ok(ok && same)
    }

    fn pooled(
        key: &str,
        pooled: &[f32],
        card: &RefManifest,
        cpu: &RefManifest,
    ) -> Result<bool, GateError> {
        let norm = serve::embed::normalize(pooled, serve::embed::Norm::default())?;
        let mut ok = true;
        for (node, ours) in [("raw", pooled), ("embd", &norm)] {
            let key = format!("{key}/{node}");
            let expected = crate::image::row(card, &key, [768, 1, 1, 1])?;
            let twin = crate::image::row(cpu, &key, [768, 1, 1, 1])?;
            ok &= crate::image::compare(&key, ours, &expected, &twin);
        }
        Ok(ok)
    }

    fn images(enc: &mut Encoder, model: &Path) -> Result<bool, GateError> {
        use crate::image;
        use bloomery_gpu::media::MediaSpan;
        use bloomery_gpu_gates::bind::Vocab;
        use bloomery_gpu_vision::gemma4v::Encoder as Tower;
        use refset::arch::gemma_embedding2::{SERVER, SERVER_CPU, TOWER, TOWER_CPU, TOWER_IMAGES};
        use serve::embeddings::{InputPart, assemble};
        use vision::arch::{TokenLimits, gemma4v};

        let card = image::open(&TOWER)?;
        let cpu = image::open(&TOWER_CPU)?;
        let server = image::open(&SERVER)?;
        let server_cpu = image::open(&SERVER_CPU)?;
        let file = gguf::Gguf::open(image::projector(model)?)?;
        let hp = gemma4v::Hparams::read(&file)?;
        let limits = TokenLimits::new(70, 1120)?;
        let rule = hp.size_rule(limits)?;
        let stream = enc.context().new_stream_with_priority(0)?;
        let mut tower = Tower::load(enc.context(), &stream, &file, limits)?;
        let tok = Vocab::new(tokenizer::Tokenizer::from_gguf(model)?)?;
        let mut ok = true;
        for name in TOWER_IMAGES {
            let (rgb, digest) = image::image(&card, &cpu, name)?;
            let p = gemma4v::preprocess(&rgb, &rule)?;
            let tokens = p.plan.n_llm_h * p.plan.n_llm_w;
            let rows = tower.encode(&p)?.rows.buf().to_host_vec(&stream)?;
            for mixed in [false, true] {
                if mixed && *name != "free-448x448" {
                    continue;
                }
                let mut parts = Vec::new();
                if mixed {
                    parts.push(InputPart::Text("A picture of coloured rings and ramps: "));
                }
                parts.push(InputPart::Media {
                    begin: gemma4v::IMAGE_BEGIN,
                    end: gemma4v::IMAGE_END,
                    positions: tokens,
                    kind: "image",
                });
                let assembled = assemble(parts.into_iter().map(Ok), &tok, CARD_CTX)?;
                let ids = &assembled.ids;
                let span = assembled
                    .spans
                    .first()
                    .ok_or("image assembly returned no span")?;
                let key = format!("{}_{name}", if mixed { "mix" } else { "img" });
                // The server pair must have consumed these same bytes and position count.
                if image::image(&server, &server_cpu, &key)?.1 != digest {
                    return Err(format!("{key}: server/tower image sha256 mismatch").into());
                }
                for man in [&server, &server_cpu] {
                    let prefix = format!("# image\t{key}\t");
                    let expected = format!("prompt_tokens {}", ids.len());
                    if !man
                        .header
                        .other
                        .iter()
                        .any(|l| l.starts_with(&prefix) && l.split('\t').any(|f| f == expected))
                    {
                        return Err(format!(
                            "{key}: server position count differs from {}",
                            ids.len()
                        )
                        .into());
                    }
                }
                let pass = enc.pass_media(
                    ids,
                    &[MediaSpan {
                        at: span.clone(),
                        rows: &rows,
                    }],
                    false,
                )?;
                ok &= pooled(&key, &pass.pooled, &server, &server_cpu)?;
            }
        }
        ok &= video(enc, &mut tower, &tok, hp.patch, &stream, (&card, &cpu))?;
        Ok(ok)
    }

    fn video(
        enc: &mut Encoder,
        tower: &mut bloomery_gpu_vision::gemma4v::Encoder,
        tok: &bloomery_gpu_gates::bind::Vocab,
        patch: usize,
        stream: &std::sync::Arc<cuda_core::CudaStream>,
        images: (&RefManifest, &RefManifest),
    ) -> Result<bool, GateError> {
        use crate::image;
        use refset::arch::gemma_embedding2::{VIDEO, VIDEO_CPU, VIDEO_IMAGES};
        use vision::arch::gemma4v;

        let video = image::open(&VIDEO)?;
        let video_cpu = image::open(&VIDEO_CPU)?;
        let frames = VIDEO_IMAGES
            .lines()
            .enumerate()
            .map(|(index, name)| {
                let (rgb, digest) = image::image(&video, &video_cpu, &format!("frame_{index}"))?;
                if image::image(images.0, images.1, name)?.1 != digest {
                    return Err(format!("video frame {index}: expected {name}'s bytes").into());
                }
                Ok(rgb)
            })
            .collect::<Result<Vec<_>, GateError>>()?;
        let (indices, prepared) = gemma4v::prepare_clip(&frames, (1, 1), patch)?;
        if indices != (0..frames.len()).collect::<Vec<_>>() {
            return Err("video reference frames must stay in one-FPS order".into());
        }
        video_pass(enc, tower, tok, stream, &prepared, (&video, &video_cpu))
    }

    fn video_pass(
        enc: &mut Encoder,
        tower: &mut bloomery_gpu_vision::gemma4v::Encoder,
        tok: &bloomery_gpu_gates::bind::Vocab,
        stream: &std::sync::Arc<cuda_core::CudaStream>,
        prepared: &[vision::Prepared],
        (video, video_cpu): (&RefManifest, &RefManifest),
    ) -> Result<bool, GateError> {
        use bloomery_gpu::media::MediaSpan;
        use refset::arch::gemma_embedding2::VIDEO_PREFIX;
        use serve::embeddings::{InputPart, assemble};
        use vision::arch::gemma4v;

        let mut parts = vec![InputPart::Text(VIDEO_PREFIX)];
        parts.extend(prepared.iter().map(|p| InputPart::Media {
            begin: gemma4v::IMAGE_BEGIN,
            end: gemma4v::IMAGE_END,
            positions: p.span_len,
            kind: "image",
        }));
        let assembled = assemble(parts.into_iter().map(Ok), tok, CARD_CTX)?;
        let expected = format!(
            "# video\tvideo\tframes {}\tprompt_tokens {}",
            prepared.len(),
            assembled.ids.len()
        );
        for man in [video, video_cpu] {
            let lines = man
                .header
                .other
                .iter()
                .filter(|l| l.starts_with("# video\t"))
                .collect::<Vec<_>>();
            if lines != [&expected] {
                return Err("video frame count or position count differs from reference".into());
            }
            if man
                .header
                .other
                .iter()
                .filter(|l| l.starts_with("# image\t"))
                .count()
                != prepared.len()
            {
                return Err("video reference has an unexpected frame count".into());
            }
        }
        println!(
            "video: positions {} spans {:?}",
            assembled.ids.len(),
            assembled.spans
        );
        let rows = prepared
            .iter()
            .map(|p| Ok(tower.encode(p.grid()?)?.rows.buf().to_host_vec(stream)?))
            .collect::<Result<Vec<_>, GateError>>()?;
        let spans = assembled
            .spans
            .iter()
            .zip(&rows)
            .map(|(at, rows)| MediaSpan {
                at: at.clone(),
                rows,
            })
            .collect::<Vec<_>>();
        let pass = enc.pass_media(&assembled.ids, &spans, false)?;
        pooled("video", &pass.pooled, video, video_cpu)
    }

    fn clips(enc: &mut Encoder, model: &Path, manifest: &Path) -> Result<bool, GateError> {
        use crate::image;
        use bloomery_gpu_gates::bind::Vocab;
        use refset::arch::gemma_embedding2::{VIDEO, VIDEO_CPU, VIDEO_CPU_SET, VIDEO_SET};
        use refset::emb2::Emb2Set;
        use vision::{
            Rgb8,
            arch::{TokenLimits, gemma4v},
        };

        let root = manifest.parent().ok_or("clip manifest has no parent")?;
        let file = gguf::Gguf::open(image::projector(model)?)?;
        let hp = gemma4v::Hparams::read(&file)?;
        let stream = enc.context().new_stream_with_priority(0)?;
        let mut tower = bloomery_gpu_vision::gemma4v::Encoder::load(
            enc.context(),
            &stream,
            &file,
            TokenLimits::new(70, gemma4v::VIDEO_TOKENS)?,
        )?;
        let tok = Vocab::new(tokenizer::Tokenizer::from_gguf(model)?)?;
        let text = std::fs::read_to_string(manifest)?;
        if text.lines().count() != 3 {
            return Err("video clips require three fixture rows".into());
        }
        let mut ok = true;
        let mut names = std::collections::HashSet::new();
        for line in text.lines() {
            let f = line.split('\t').collect::<Vec<_>>();
            if f.len() != 9 {
                return Err("malformed video clip fixture row".into());
            }
            let name = f[0];
            if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
                return Err("invalid clip name".into());
            }
            if !names.insert(name) {
                return Err(format!("{name}: duplicate video clip fixture").into());
            }
            let card = Emb2Set::open(&root.join(name).join(VIDEO_SET), &VIDEO)?.man;
            let cpu = Emb2Set::open(&root.join(name).join(VIDEO_CPU_SET), &VIDEO_CPU)?.man;
            let fps = (f[1].parse()?, f[2].parse()?);
            let width: usize = f[3].parse()?;
            let height: usize = f[4].parse()?;
            let count: usize = f[5].parse()?;
            let bytes = width
                .checked_mul(height)
                .and_then(|n| n.checked_mul(3))
                .filter(|&n| n > 0)
                .ok_or("invalid clip dimensions")?;
            let total = bytes.checked_mul(count).ok_or("clip byte count overflow")?;
            if total > 512 * 1024 * 1024
                || std::fs::metadata(f[6])?.len() != total as u64
                || gguf::digest::sha256_file(Path::new(f[6]))? != f[7]
            {
                return Err(format!("{name}: clip size or sha256 mismatch").into());
            }
            let raw = std::fs::read(f[6])?;
            if raw.len() != total {
                return Err(format!("{name}: decoded clip byte count mismatch").into());
            }
            let frames = raw
                .chunks_exact(bytes)
                .map(|data| Rgb8 {
                    width,
                    height,
                    data: data.to_vec(),
                })
                .collect::<Vec<_>>();
            let (indices, prepared) = gemma4v::prepare_clip(&frames, fps, hp.patch)?;
            let expected = f[8]
                .split(',')
                .map(str::parse)
                .collect::<Result<Vec<usize>, _>>()?;
            if indices != expected {
                return Err(
                    format!("{name}: source indices {indices:?}, expected {expected:?}").into(),
                );
            }
            for (i, &source) in indices.iter().enumerate() {
                if image::image(&card, &cpu, &format!("frame_{i}"))?.0 != frames[source] {
                    return Err(
                        format!("{name}: oracle frame {i} differs from source {source}").into(),
                    );
                }
            }
            println!("{name}: source indices {indices:?} PASS");
            ok &= video_pass(enc, &mut tower, &tok, &stream, &prepared, (&card, &cpu))?;
        }
        Ok(ok)
    }

    fn hf(enc: &mut Encoder, reach: &[Option<u32>]) -> Result<bool, GateError> {
        let man = open(&HF)?;
        let window = refset::emb2::hf_window(&man)?;
        if !reach.iter().any(Option::is_some) || reach.iter().flatten().any(|&r| r != window) {
            return Err(format!("HF one-sided reach {window}, encoder reaches {reach:?}").into());
        }
        println!("(h) HF sliding_window {window}: one-sided inclusive; GGUF reaches {reach:?}");
        let llama = open(&TEXT)?;
        let cases = cases(&man, HF.sets[0])?;
        if !cases.iter().any(|c| c.ids.len() > 2 * window as usize + 1) {
            return Err("HF cases do not cross both window edges".into());
        }
        let mut ok = true;
        for c in cases {
            let expected = crate::image::row(&man, &format!("{}/embd", c.id), [768, 1, 1, 1])?;
            let pass = enc.pass(&c.ids, false)?;
            let ours = serve::embed::normalize(&pass.pooled, serve::embed::Norm::default())?;
            let text_id = c.id.strip_suffix("_hf").unwrap_or(&c.id);
            // PIN(2026-10-17): frozen unit-vector L2 budget derived from the minimum
            // Q8_0/HF cosine over 84 E3 cases on DGX Spark and RTX 5090.
            let twin = (2.0 * (1.0 - 0.999_129_044_1_f64)).sqrt();
            let mut arms = vec![("bloomery", ours)];
            if text_id == c.id {
                let reference = serve::embed::normalize(
                    &tap(&llama, &format!("{text_id}/pooled"))?,
                    serve::embed::Norm::default(),
                )?;
                arms.push(("llama.cpp", reference));
            }
            for (arm, vector) in &arms {
                let distance = rel(vector, &expected);
                // PIN(2026-10-17): 5e-2 rounds up the derived 0.04174 maximum
                // normalized Q8_0/HF distance on DGX Spark and RTX 5090.
                if !distance.is_finite() || distance > 5e-2 {
                    return Err(format!("{} {arm}/HF: rel {distance} exceeds 5e-2", c.id).into());
                }
                let cosine = crate::e2e::cosine(vector, &expected);
                let maximum = crate::e2e::max_delta(vector, &expected);
                let held = Held::distances(distance, twin);
                println!(
                    "(h) {} {arm}/HF: cosine {cosine:.9e} max_abs_diff {maximum:.9e} {}; {}",
                    c.id,
                    held.line("rel"),
                    verdict(held.ok())
                );
                ok &= held.ok();
            }
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let args: Vec<String> = std::env::args().skip(1).collect();
        let (v1, hf_only, clip_manifest) = match args.as_slice() {
            [] => (false, false, None),
            [arg] if arg == "--v1" => (true, false, None),
            [arg] if arg == "--hf" => (false, true, None),
            [flag, path] if flag == "--video-clips" => (false, false, Some(Path::new(path))),
            _ => {
                return Err(format!(
                    "expected --v1, --hf, --video-clips <manifest>, or no arguments; got {args:?}"
                )
                .into());
            }
        };
        use refset::arch::gemma_embedding as emb1;
        let (text, text_cpu, ids_set, model_sha) = if v1 {
            (
                &emb1::TEXT,
                &emb1::TEXT_CPU,
                emb1::TOKENIZER.sets[0],
                emb1::MODEL_SHA256,
            )
        } else {
            (&TEXT, &TEXT_CPU, IDS_SET, MODEL_SHA256)
        };
        let model = ref_model_path()?;
        let digest = gguf::digest::sha256_file(&model)?;
        if digest != model_sha {
            return Err(format!(
                "{}: sha256 {digest}, the sets are dumped from {model_sha}",
                model.display()
            )
            .into());
        }
        let split = Split::open(&model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let read = model::arch::spec(&split).map_err(|e| format!("{}: {e}", model.display()))?;
        let encoder = read
            .spec
            .encoder
            .as_ref()
            .ok_or("gemma-embedding2 has no encoder description")?;
        let bound = if v1 {
            model::arch::gemma_embedding::hparams::CARD_CTX
        } else {
            CARD_CTX
        };
        let ctx = bound.min(read.spec.ctx_train as usize);
        let bytes = card_bytes(&split, ctx)?;
        if let Some(manifest) = clip_manifest {
            let mut enc = Encoder::open(split, ctx, Gpu::new()?)?;
            return if clips(&mut enc, &model, manifest)? {
                Ok(())
            } else {
                Err(checks_failed())
            };
        }
        if hf_only {
            let mut enc = Encoder::open(split, ctx, Gpu::new()?)?;
            return if hf(&mut enc, &encoder.reach)? {
                Ok(())
            } else {
                Err(checks_failed())
            };
        }
        let man = open(text)?;
        let cpu = open(text_cpu)?;
        check_case_identities(&man.header, &cpu.header)?;
        let cases = cases(&man, ids_set)?;
        let gpu = Gpu::new()?;
        println!(
            "gate_gemma_embedding2_e2e: device {}: {} at {ctx} positions, card figure {bytes} B; \
             {} cases; band {TWIN} x the CPU twin, at least {FLOOR:e}",
            gpu.device_name()?,
            model.display(),
            cases.len()
        );
        let mut ok = window(&encoder.reach, &cases);
        let t = Instant::now();
        let mut enc = Encoder::open(split, ctx, gpu)?;
        elapsed("(open)", &t);
        let mut longest: Option<(usize, Pass)> = None;
        for (i, c) in cases.iter().enumerate() {
            let r =
                case(&mut enc, c, &man, &cpu, read.spec.layers.len(), v1).map(|(pass_ok, pass)| {
                    if longest
                        .as_ref()
                        .is_none_or(|(j, _)| c.ids.len() > cases[*j].ids.len())
                    {
                        longest = Some((i, pass));
                    }
                    pass_ok
                });
            ok &= clause("(t)", &c.id, r);
        }
        if let Some((i, first)) = longest {
            let c = &cases[i];
            // PIN(2026-10-24): a pass without taps no longer returns every
            // position's row; (d) holds its pooled vector and a rerun with taps
            // the rows, and (r) compares a rerun with taps.
            ok &= clause("(d)", &c.id, rerun(&mut enc, c, &first));
            ok &= clause(
                "(r)",
                "refusals",
                rows(&first).and_then(|rows| refusals(&mut enc, c, rows, read.spec.vocab as usize)),
            );
        }
        if !v1 {
            ok &= clause(
                "(h)",
                "HF sentence-transformers",
                hf(&mut enc, &encoder.reach),
            );
            ok &= clause("(i)", "image inputs", images(&mut enc, &model));
        }
        let past = Split::open(&model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let refused = refused_by(
            &Encoder::open(past, ctx + 1, Gpu::new()?),
            "Encoder::open",
            "a context of",
        );
        println!(
            "(r) a load at {} positions: refused by name {}",
            ctx + 1,
            verdict(refused)
        );
        ok &= refused;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    #[cfg(test)]
    mod tests {
        use super::{Case, RefHeader, check_case_identities, window};

        #[test]
        fn a_window_case_reaches_past_the_longest_sliding_layer() {
            let mut cases = [Case {
                id: "window".to_owned(),
                ids: vec![0; 3],
            }];
            assert!(window(&[None], &[]));
            assert!(!window(&[Some(1), None, Some(2)], &cases));
            cases[0].ids.push(0);
            assert!(window(&[Some(1), None, Some(2)], &cases));
        }

        #[test]
        fn a_cpu_case_of_another_digest_is_refused_by_name() {
            let card = RefHeader {
                other: vec![format!(
                    "# case\tcases/01\t2\tcases/01.ids\t{}",
                    "a".repeat(64)
                )],
                ..RefHeader::default()
            };
            let mut cpu = card.clone();
            check_case_identities(&card, &cpu).unwrap();
            cpu.other[0] = cpu.other[0].replace(&"a".repeat(64), &"b".repeat(64));
            let err = check_case_identities(&card, &cpu).unwrap_err().to_string();
            assert_eq!(err, "CPU twin case identities differ from the card set");
        }
    }
}
