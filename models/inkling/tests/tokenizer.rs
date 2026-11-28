//! Encode checks against transformers 5.24.0 `AutoTokenizer.encode(..., add_special_tokens=False)`
//! for tiny-random/inkling-2 at `0a57c72e4e5b68aafe1d983dbb62980344a5fcd7`.

use inkling_goldy::tokenizer::InklingTokenizer;
use std::path::PathBuf;

fn tokenizer() -> InklingTokenizer {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/tokenizer.json");
    InklingTokenizer::from_file(&path).unwrap_or_else(|err| {
        panic!(
            "{err}. Fetch the pinned file with tools/fetch_tokenizer.sh (sha256 7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6)."
        )
    })
}

#[test]
fn encodes_the_same_ids_as_transformers() {
    let tok = tokenizer();
    assert_eq!(tok.encode("Hello").unwrap(), vec![32269]);
    assert_eq!(
        tok.encode("Hello, world!").unwrap(),
        vec![32269, 11, 150770, 0]
    );
    assert_eq!(
        tok.encode("the sky is blue").unwrap(),
        vec![268, 158808, 149884, 152778]
    );
    assert_eq!(tok.encode("<|begin_of_text|>").unwrap(), vec![200028]);
    assert_eq!(tok.decode(&[32269]).unwrap(), "Hello");
}

#[test]
fn decode_roundtrips_a_short_phrase() {
    let tok = tokenizer();
    let ids = tok.encode("the sky is blue").unwrap();
    assert_eq!(tok.decode(&ids).unwrap(), "the sky is blue");
}
