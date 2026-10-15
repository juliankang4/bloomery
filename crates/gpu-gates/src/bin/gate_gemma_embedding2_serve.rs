//! EmbeddingGemma 2 beside Qwen3-30B-A3B-Instruct-2507 Q4_K_M: the server vectors,
//! the reserve's expert cost and unified-pool term, greedy tokens alone and during
//! embedding calls, and named refusals. One visible card keeps the side seat on the stage.
//! The plain 12 GiB plan leaves host experts; adding the reserve to the side run's
//! budget restores that split. At the unchanged budget the side run must seat fewer experts.

#[cfg(feature = "gemma-embedding2")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_gemma_embedding2_serve", gate::run())
}

#[cfg(feature = "gemma-embedding2")]
#[path = "shared/qwen3moe_place.rs"]
#[allow(
    dead_code,
    reason = "the gate uses the seat's planner, not its loaders"
)]
mod q3place;

#[cfg(feature = "gemma-embedding2")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::Duration;

    use bloomery_gpu::arch::qwen3moe::KvQ8;
    use bloomery_gpu_gates::beside::{self, Beside};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::record::{self, Log};
    use bloomery_gpu_gates::serve_client::{
        Served, check, curl, ids_of, json_of, metric, plan_run, serve_cmd, server_log, tokenized,
    };
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, data_dir, gpu_census, ref_model_path,
        ref_tensor_logical_in, sha256,
    };
    use gguf::Split;
    use model::placement::{Plan, PlanLevers, UnifiedPool};
    use refset::arch::gemma_embedding2::{
        IDS_SET, MODEL, MODEL_SHA256, REL_FLOOR, SERVER, TEXT_CPU, TWIN,
    };
    use refset::emb2::Emb2Set;
    use serde_json::{Value, json};

    use crate::q3place::PlaceQ3;

    const CTX: usize = 4096;
    const BUDGET: u64 = 12 * 1024 * 1024 * 1024;
    const N_PREDICT: usize = 512;
    const PROMPT: &str = "The capital of France is";

    struct Case {
        name: String,
        text: String,
        tokens: u64,
    }

    fn cases(man: &RefManifest) -> Result<Vec<Case>, GateError> {
        let mut cases = Vec::new();
        for line in &man.header.other {
            let Some(rest) = line.strip_prefix("# tokenize\t") else {
                continue;
            };
            let fields: Vec<_> = rest.split('\t').collect();
            let [name, "equal", count] = fields[..] else {
                return Err(format!("server set: malformed tokenization identity {line:?}").into());
            };
            let tokens = count
                .strip_prefix("prompt_tokens ")
                .ok_or_else(|| format!("server set: missing prompt_tokens in {line:?}"))?
                .parse()?;
            cases.push(Case {
                name: name.to_owned(),
                text: std::fs::read_to_string(
                    data_dir().join(IDS_SET).join(format!("{name}.txt")),
                )?,
                tokens,
            });
        }
        if cases.is_empty() {
            return Err("server set: no text cases".into());
        }
        cases.sort_by_key(|c| c.tokens);
        Ok(cases)
    }

    fn tap(man: &RefManifest, name: &str) -> Result<Vec<f32>, GateError> {
        Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, 0)?)?)
    }

    fn vector(v: &Value) -> Result<Vec<f32>, GateError> {
        v["data"][0]["embedding"]
            .as_array()
            .ok_or("no embedding array")?
            .iter()
            .map(|x| {
                let x = x.as_f64().ok_or("an embedding value is not numeric")? as f32;
                if !x.is_finite() {
                    return Err("an embedding value is not finite".into());
                }
                Ok(x)
            })
            .collect()
    }

    fn norm(v: &[f32]) -> f64 {
        v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt()
    }

    fn distances(a: &[f32], b: &[f32]) -> (f64, f64) {
        if a.len() != b.len() || a.is_empty() || a.iter().chain(b).any(|x| !x.is_finite()) {
            return (f64::INFINITY, f64::INFINITY);
        }
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let cosine = (1.0 - dot / (norm(a) * norm(b))).abs();
        let max = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
            .fold(0.0, f64::max);
        (cosine, max)
    }

    fn within(ours: &[f32], want: &[f32], raw: &[f32], twin: &[f32], normalized: bool) -> bool {
        let delta: Vec<_> = raw.iter().zip(twin).map(|(&x, &y)| x - y).collect();
        // The encoder gate's pooled floor, before normalization: the bands below derive from it,
        // not from measured serving distances.
        let relative = REL_FLOOR.max(TWIN * norm(&delta) / norm(raw));
        let absolute = relative * norm(raw);
        // Cutting can reduce the norm. Normalization magnifies an absolute error E by at most
        // 2 E / (norm - E); cosine distance between unit vectors is half their squared L2 distance.
        let cut_norm = norm(&raw[..want.len()]);
        let unit = 2.0 * absolute / (cut_norm - absolute);
        let max_band = if normalized { unit } else { absolute };
        let cosine_band = unit * unit / 2.0;
        let (cosine, max) = distances(ours, want);
        println!(
            "cosine {cosine:.6e} band {cosine_band:.6e}; max_abs {max:.6e} band {max_band:.6e} [derived bands]"
        );
        relative.is_finite()
            && cut_norm > absolute
            && cosine.is_finite()
            && max.is_finite()
            && cosine <= cosine_band
            && max <= max_band
    }

    fn embedding(url: &dyn Fn(&str) -> String, body: Value) -> Result<Value, GateError> {
        let (status, text) = curl(&url("/v1/embeddings"), Some(&body), false)?;
        json_of("/v1/embeddings", status, &text)
    }

    fn vectors(
        url: &dyn Fn(&str) -> String,
        cases: &[Case],
        server: &RefManifest,
        cpu: &RefManifest,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let name = Path::new(MODEL)
            .file_name()
            .ok_or("embedding file name")?
            .to_string_lossy();
        for (index, c) in cases.iter().enumerate() {
            let raw = tap(server, &format!("{}/raw", c.name))?;
            let twin = tap(cpu, &format!("{}/pooled", c.name))?;
            if raw.len() != 768 || twin.len() != raw.len() {
                return Err(format!("{}: reference vector widths are not 768", c.name).into());
            }
            for (width, normalized, row) in [
                (768, false, "raw"),
                (768, true, "embd"),
                (128, true, "mrl128"),
                (256, true, "mrl256"),
                (512, true, "mrl512"),
            ] {
                if index != 0 && width != 768 {
                    continue;
                }
                let body = json!({"input": c.text, "model": name,
                    "dimensions": width, "embd_normalize": if normalized { 2 } else { -1 }});
                let v = embedding(url, body)?;
                let ours = vector(&v)?;
                let want = tap(server, &format!("{}/{row}", c.name))?;
                if want.len() != width {
                    return Err(format!("{}: {row} width", c.name).into());
                }
                check(
                    &mut ok,
                    &format!("vectors_{}_{row}", c.name),
                    v["model"] == name.as_ref()
                        && v["usage"]["prompt_tokens"] == c.tokens
                        && within(&ours, &want, &raw, &twin, normalized),
                );
            }
        }
        Ok(ok)
    }

    fn cmd(budget: u64) -> Result<Command, GateError> {
        let mut cmd = serve_cmd()?;
        cmd.env(bloomery_levers::CARD_BUDGET, budget.to_string())
            .env(bloomery_levers::DRAFT, "off");
        Ok(cmd)
    }

    fn args(model: &str, side: bool) -> Vec<&str> {
        let mut args = vec![
            "-m",
            model,
            "--port",
            "0",
            "--place",
            "a",
            "--ctx",
            "4096",
            "--parallel",
            "1",
        ];
        if side {
            args.extend(["--embd-model", MODEL]);
        }
        args
    }

    fn log(path: &Path) -> Result<Log, GateError> {
        server_log(path, &[&record::PLAN38, &record::EMBEDDING_SEAT])
    }

    fn split(log: &Log) -> Result<(u64, u64), GateError> {
        let plan = log.first(&record::PLAN38)?.ok_or("missing plan record")?;
        Ok((plan.u64("card_experts")?, plan.u64("host_experts")?))
    }

    fn plan(model: &str, dir: &Path, side: bool, budget: u64) -> Result<Log, GateError> {
        std::fs::create_dir_all(dir)?;
        let err = dir.join("plan.err");
        let (status, _) = plan_run(cmd(budget)?, &args(model, side), &err)?;
        if !status.success() {
            return Err(format!("plan failed ({status}): {}", err.display()).into());
        }
        log(&err)
    }

    fn pool(model: &Path, reserve: u64) -> Result<bool, GateError> {
        let split = Split::open(model)?;
        let census = gpu_census::census()?;
        let q = PlaceQ3::qwen3_on(
            &split,
            Place::A,
            &census,
            CTX,
            KvQ8::F16,
            Beside {
                image: None,
                embedding: Some(reserve),
            },
        )?;
        let p = q.plan(
            CTX,
            &PlanLevers {
                card_budget_bytes: Some(BUDGET + reserve),
            },
        )?;
        let mut held = p.machine.clone();
        // Exercise the pool formula on discrete cards too, keeping the real plan's rows and totals.
        if held.nodes[0].unified.is_none() {
            held.nodes[0].unified = Some(UnifiedPool {
                bytes: held.host().usable_bytes,
            });
        }
        let mut plain = held.clone();
        plain.cards[0]
            .reserves
            .retain(|(name, _)| name != beside::EMBEDDING);
        let with = Plan {
            machine: &held,
            ..p.clone()
        }
        .pool_need()
        .ok_or("no pool term")?;
        let without = Plan {
            machine: &plain,
            ..p.clone()
        }
        .pool_need()
        .ok_or("no pool term")?;
        println!(
            "pool cards with {} B, without {} B, reserve {reserve} B",
            with.cards, without.cards
        );
        Ok(beside::reserve_in(p.machine).embedding == Some(reserve)
            && with.cards.checked_sub(without.cards) == Some(reserve)
            && with.host == without.host)
    }

    fn chat(url: &dyn Fn(&str) -> String, ids: &[u32]) -> Result<Vec<u32>, GateError> {
        let body = json!({"prompt": ids, "n_predict": N_PREDICT, "temperature": 0,
            "ignore_eos": true, "return_tokens": true, "cache_prompt": false});
        let (status, text) = curl(&url("/completion"), Some(&body), false)?;
        Ok(ids_of(&json_of("/completion", status, &text)?["tokens"]))
    }

    fn refusal(
        url: &dyn Fn(&str) -> String,
        body: Value,
        status: u16,
        kind: &str,
    ) -> Result<bool, GateError> {
        let (got, text) = curl(&url("/v1/embeddings"), Some(&body), false)?;
        let v: Value = serde_json::from_str(&text)?;
        Ok(got == status && v["error"]["type"] == kind && v["error"]["message"].is_string())
    }

    fn concurrent(
        url: &dyn Fn(&str) -> String,
        ids: &[u32],
        c: &Case,
        want: &[u32],
        server: &RefManifest,
        cpu: &RefManifest,
    ) -> Result<bool, GateError> {
        let completion_url = url("/completion");
        let ids = ids.to_vec();
        let raw = tap(server, &format!("{}/raw", c.name))?;
        let twin = tap(cpu, &format!("{}/pooled", c.name))?;
        let oracle = tap(server, &format!("{}/embd", c.name))?;
        let start = metric(url, "n_decode_total")?.ok_or("missing n_decode_total")?;
        std::thread::scope(|scope| {
            let generating =
                scope.spawn(|| chat(&|_| completion_url.clone(), &ids).map_err(|e| e.to_string()));
            let mut active = false;
            for _ in 0..200 {
                if generating.is_finished() {
                    break;
                }
                if metric(url, "n_decode_total")?.unwrap_or(start) > start {
                    active = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            let mut decode_steps = 0.0_f64;
            for _ in 0..32 {
                if !active || generating.is_finished() {
                    break;
                }
                let before = metric(url, "n_decode_total")?.ok_or("missing n_decode_total")?;
                let v = embedding(url, json!({"input": c.text}))?;
                let after = metric(url, "n_decode_total")?.ok_or("missing n_decode_total")?;
                if !within(&vector(&v)?, &oracle, &raw, &twin, true) {
                    return Err("concurrent embedding is outside the server vector band".into());
                }
                decode_steps = decode_steps.max(after - before);
                if decode_steps >= 2.0 {
                    break;
                }
            }
            let got = generating
                .join()
                .map_err(|_| "chat request panicked")?
                .map_err(|e| format!("chat request: {e}"))?;
            println!(
                "concurrent decode observed {decode_steps} steps during one {}-token embedding; {} greedy tokens",
                c.tokens,
                got.len()
            );
            Ok(decode_steps >= 2.0 && got == want)
        })
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        record::at_main(
            "gate_gemma_embedding2_serve",
            record::GATE_GEMMA_EMBEDDING2_SERVE,
        );
        let mut args_it = std::env::args().skip(1);
        let mut dir = PathBuf::from("target/embedding2-serve-gate");
        while let Some(a) = args_it.next() {
            if a != "--dir" {
                return Err(format!("unknown argument {a}; takes --dir DIR").into());
            }
            dir = args_it.next().ok_or("--dir needs DIR")?.into();
        }
        std::fs::create_dir_all(&dir)?;
        if gpu_census::census()?.len() != 1 {
            return Err(
                "serve gate needs one visible card, so the side seat reserves the stage card"
                    .into(),
            );
        }
        if sha256(Path::new(MODEL))? != MODEL_SHA256 {
            return Err("embedding model sha256 differs from the server set".into());
        }
        let model_path = ref_model_path()?;
        let model = model_path.to_str().ok_or("chat model path is not UTF-8")?;
        if Split::open(model)?.architecture() != Some("qwen3moe") {
            return Err("serve gate needs Qwen3-30B-A3B-Instruct-2507 Q4_K_M".into());
        }
        let server = Emb2Set::open(&SERVER.path(SERVER.sets[0]), &SERVER)?.man;
        let cpu = Emb2Set::open(&TEXT_CPU.path(TEXT_CPU.sets[0]), &TEXT_CPU)?.man;
        let cases = cases(&server)?;
        let longest = cases.last().ok_or("server set: no text cases")?;
        let plain = plan(model, &dir.join("plan-plain"), false, BUDGET)?;
        let reserved = plan(model, &dir.join("plan-reserved"), true, BUDGET)?;
        let seat = reserved
            .first(&record::EMBEDDING_SEAT)?
            .ok_or("no embedding_seat record")?;
        let reserve = seat.u64("reserve")?;
        let compensated = plan(model, &dir.join("plan-compensated"), true, BUDGET + reserve)?;
        let (base_card, base_host) = split(&plain)?;
        let (less_card, more_host) = split(&reserved)?;
        println!(
            "plain split {base_card}/{base_host}, reserved {less_card}/{more_host}, reserve {reserve} B"
        );
        let mut ok = true;
        check(
            &mut ok,
            "plan_expert_cost",
            base_card > 0
                && base_host > 0
                && reserve > 0
                && less_card < base_card
                && more_host > base_host,
        );
        check(
            &mut ok,
            "plan_same_split",
            split(&plain)? == split(&compensated)?,
        );
        check(
            &mut ok,
            "reserve_record",
            seat.word("arch")? == "gemma-embedding2"
                && reserve == seat.u64("bytes")?
                && seat.u64("ctx")? == 8192
                && reserve
                    == bloomery_gpu_gemma_embedding2::card_bytes(&Split::open(MODEL)?, 8192)?,
        );
        check(
            &mut ok,
            "pool_cards_count_reserve",
            pool(&model_path, reserve)?,
        );

        let mut baseline = Vec::new();
        for (side, budget, arm) in [(false, BUDGET, "plain"), (true, BUDGET + reserve, "side")] {
            let work = dir.join(arm);
            std::fs::create_dir_all(&work)?;
            let err = work.join("server.err");
            let mut served = Served::spawn_cmd(cmd(budget)?, &args(model, side), &work)?;
            let addr = served.address(&err, 600, Duration::from_secs(1))?;
            let url = |path: &str| format!("http://{addr}{path}");
            let ids = tokenized(&url, PROMPT)?;
            let tokens = chat(&url, &ids)?;
            std::fs::write(work.join("greedy.json"), serde_json::to_string(&tokens)?)?;
            check(
                &mut ok,
                &format!("{arm}_greedy_count"),
                tokens.len() == N_PREDICT,
            );
            check(
                &mut ok,
                &format!("{arm}_loaded_split"),
                split(&log(&err)?)? == split(&plain)?,
            );
            if !side {
                baseline = tokens;
                check(
                    &mut ok,
                    "no_side_501",
                    refusal(&url, json!({"input":"a"}), 501, "not_supported_error")?,
                );
            } else {
                check(&mut ok, "idle_greedy_same", tokens == baseline);
                ok &= vectors(&url, &cases, &server, &cpu)?;
                let (status, text) = curl(&url("/v1/models"), None, false)?;
                let models = json_of("/v1/models", status, &text)?;
                let chat_name = models["data"][0]["id"].as_str().ok_or("no chat model id")?;
                check(
                    &mut ok,
                    "embedding_model_listed",
                    models["data"][1]["id"]
                        == Path::new(MODEL)
                            .file_name()
                            .ok_or("embedding file name")?
                            .to_string_lossy()
                            .as_ref()
                        && models["data"][1]["meta"]["n_embd"] == 768
                        && models["data"][1]["meta"]["n_ctx"] == 8192,
                );
                check(
                    &mut ok,
                    "chat_model_refused",
                    refusal(
                        &url,
                        json!({"input":"a", "model": chat_name}),
                        400,
                        "invalid_request_error",
                    )?,
                );
                check(
                    &mut ok,
                    "unknown_model_refused",
                    refusal(
                        &url,
                        json!({"input":"a", "model":"not-a-served-model"}),
                        400,
                        "invalid_request_error",
                    )?,
                );
                check(
                    &mut ok,
                    "dimensions_refused",
                    refusal(
                        &url,
                        json!({"input":"a", "dimensions":129}),
                        400,
                        "invalid_request_error",
                    )?,
                );
                check(
                    &mut ok,
                    "context_refused",
                    refusal(
                        &url,
                        json!({"input":vec![1;8193]}),
                        400,
                        "exceed_context_size_error",
                    )?,
                );
                check(
                    &mut ok,
                    "concurrent_greedy_same",
                    concurrent(&url, &ids, longest, &baseline, &server, &cpu)?,
                );
            }
            println!("{arm} server stopped {}", served.stop()?);
        }
        let err = dir.join("embedding-as-chat.err");
        let (status, lines) = plan_run(cmd(BUDGET)?, &["-m", MODEL], &err)?;
        let text = lines.join("\n");
        check(
            &mut ok,
            "embedding_as_chat_refused",
            !status.success() && text.contains("embedding model") && text.contains("--embd-model"),
        );
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
