use embed::input::prompt_ids;
use embed::request::{Request, check_tokens};
use serde_json::{Value, json};

fn canonical(mut value: Value) -> Value {
    value.sort_all_objects();
    value
}

#[test]
fn llama_server_responses_are_byte_exact_at_f32_precision() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/llama-server.json")).unwrap();
    assert!(
        fixture["revision"]
            .as_str()
            .unwrap()
            .starts_with(refset::emb2::LCPP_BUILD)
    );
    assert_eq!(
        fixture["model_sha256"],
        refset::arch::gemma_embedding2::MODEL_SHA256
    );
    let pooled: Vec<f32> = fixture["pooled_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| f32::from_bits(u32::try_from(x.as_u64().unwrap()).unwrap()))
        .collect();
    let mut bytes = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let body = case["request"].as_object().unwrap();
        let model = body["model"].as_str().unwrap();
        let request = Request::parse(body, model, &[pooled.len()], true).unwrap();
        let ids = prompt_ids(
            "input",
            request.inputs[0],
            |_| panic!("a token input"),
            None,
            None,
            usize::MAX,
        )
        .unwrap();
        check_tokens(ids.len(), 128).unwrap();
        assert_eq!(json!(ids), fixture["ids"]);
        let got = request
            .answer(std::slice::from_ref(&pooled), ids.len())
            .unwrap();
        let mut want = case["response"].clone();
        if body["encoding_format"] == "float" {
            for x in want["data"][0]["embedding"].as_array_mut().unwrap() {
                *x = json!(x.as_f64().unwrap() as f32);
            }
        }
        let got = serde_json::to_vec(&canonical(got)).unwrap();
        let want = serde_json::to_vec(&canonical(want)).unwrap();
        assert_eq!(got, want, "request {body:?}");
        bytes += got.len();
    }
    println!(
        "llama-server revision={} cases={} canonical_json_bytes={bytes} difference=0 (float values round-trip through f32; base64 strings unchanged)",
        fixture["revision"].as_str().unwrap(),
        fixture["cases"].as_array().unwrap().len()
    );
}

#[test]
fn mrl_matches_the_pinned_normalizers_bytes() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/mrl.json")).unwrap();
    assert_eq!(
        fixture["set"],
        refset::arch::gemma_embedding2::SERVER_CPU_SET
    );
    assert!(
        fixture["revision"]
            .as_str()
            .unwrap()
            .starts_with(refset::emb2::LCPP_BUILD)
    );
    assert_eq!(fixture["normalizer"], "llama.cpp common_embd_normalize");
    let pooled: Vec<f32> = fixture["pooled_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| f32::from_bits(u32::try_from(x.as_u64().unwrap()).unwrap()))
        .collect();
    let vectors = fixture["vectors"].as_array().unwrap();
    let mut dims: Vec<usize> = vectors
        .iter()
        .map(|v| usize::try_from(v["width"].as_u64().unwrap()).unwrap())
        .collect();
    dims.push(pooled.len());
    for vector in vectors {
        let body =
            json!({"input": [1], "dimensions": vector["width"], "encoding_format": "base64"});
        let request = Request::parse(body.as_object().unwrap(), "encoder", &dims, true).unwrap();
        let got = request.answer(std::slice::from_ref(&pooled), 1).unwrap();
        assert_eq!(got["data"][0]["embedding"], vector["base64"]);
    }
    println!(
        "refset={} case={} widths={} difference=0 (MRL through pinned common_embd_normalize)",
        fixture["set"].as_str().unwrap(),
        fixture["case"].as_str().unwrap(),
        vectors.len()
    );
}
