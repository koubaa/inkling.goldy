//! The pinned `tokenizer.json` encodes each reference prompt to the ids
//! transformers fed the model, and decodes the reference continuation back to
//! its text. Needs `python3 tools/fetch_assets.py --only tokenizer.json`.

use ammon::HfTokenizer;
use serde_json::Value;

#[test]
fn tokenizer_matches_the_reference_ids() {
    let path = smollm2_goldy::assets_dir().join("tokenizer.json");
    let tokenizer = HfTokenizer::from_file(&path).unwrap_or_else(|e| {
        panic!("{e:#}; run `python3 tools/fetch_assets.py --only tokenizer.json` in models/smollm2")
    });
    let json: Value = serde_json::from_str(include_str!("data/reference.json")).unwrap();
    let ids = |v: &Value| -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect()
    };
    for case in json["cases"].as_array().unwrap() {
        let prompt = case["prompt"].as_str().unwrap();
        assert_eq!(
            tokenizer.encode(prompt).unwrap(),
            ids(&case["prompt_ids"]),
            "{prompt:?}"
        );
        assert_eq!(
            tokenizer.decode(&ids(&case["greedy_ids"])).unwrap(),
            case["greedy_text"].as_str().unwrap(),
            "{prompt:?}"
        );
    }
}

#[test]
fn fetch_script_pins_the_crate_revision() {
    let script = include_str!("../tools/fetch_assets.py");
    assert!(script.contains(&format!("REPO = \"{}\"", smollm2_goldy::REPO)));
    assert!(script.contains(&format!("REVISION = \"{}\"", smollm2_goldy::REVISION)));
}
