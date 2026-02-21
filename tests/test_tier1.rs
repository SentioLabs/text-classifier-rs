use text_classifier::features::extract_features;
use text_classifier::tier1::classify_tier1;
use text_classifier::{TextType, Tier, classify};

fn read_fixture(path: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{path}")).unwrap()
}

#[test]
fn prose_classified_as_prose() {
    let features = extract_features(&read_fixture("prose/simple.txt"));
    let result = classify_tier1(&features);
    assert_eq!(result.text_type, TextType::Prose);
    assert_eq!(result.tier, Tier::Structural);
    assert!(result.confidence >= 0.7, "confidence={}", result.confidence);
}

#[test]
fn code_classified_as_code() {
    let features = extract_features(&read_fixture("code/python.txt"));
    let result = classify_tier1(&features);
    assert_eq!(result.text_type, TextType::Code);
    assert_eq!(result.tier, Tier::Structural);
    assert!(result.confidence >= 0.7, "confidence={}", result.confidence);
}

#[test]
fn table_classified_as_tabular() {
    let features = extract_features(&read_fixture("tabular/pipe_table.txt"));
    let result = classify_tier1(&features);
    assert_eq!(result.text_type, TextType::Tabular);
    assert_eq!(result.tier, Tier::Structural);
    assert!(result.confidence >= 0.7, "confidence={}", result.confidence);
}

#[test]
fn pdf_dump_classified_as_pdf_dump() {
    let features = extract_features(&read_fixture("pdf_dump/ocr_garbage.txt"));
    let result = classify_tier1(&features);
    assert_eq!(result.text_type, TextType::PdfDump);
    assert_eq!(result.tier, Tier::Structural);
    assert!(result.confidence >= 0.7, "confidence={}", result.confidence);
}

#[test]
fn short_text_classified_as_skip() {
    // Short text is caught by classify() before reaching tier1
    let result = classify("hello world");
    assert_eq!(result.text_type, TextType::Skip);
    assert_eq!(result.confidence, 1.0);
}

#[test]
fn empty_text_classified_as_skip() {
    let result = classify("");
    assert_eq!(result.text_type, TextType::Skip);
}
