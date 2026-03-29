use text_classifier::features::extract_features;
use text_classifier::tier1::{classify_tier1, thresholds};
use text_classifier::types::{ContentSubType, TextCategory};
use text_classifier::{TextType, Tier, classify};

fn read_fixture(path: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{path}")).unwrap()
}

#[test]
fn prose_classified_as_prose() {
    let features = extract_features(&read_fixture("prose/simple.txt"));
    let result = classify_tier1(&features);
    assert_eq!(result.category, TextType::Prose);
    assert_eq!(result.tier, Tier::Structural);
    assert!(result.confidence >= 0.7, "confidence={}", result.confidence);
}

#[test]
fn code_classified_as_code() {
    let features = extract_features(&read_fixture("code/python.txt"));
    let result = classify_tier1(&features);
    assert_eq!(result.category, TextType::Code);
    assert_eq!(result.tier, Tier::Structural);
    assert!(result.confidence >= 0.7, "confidence={}", result.confidence);
}

#[test]
fn table_classified_as_structured() {
    let features = extract_features(&read_fixture("tabular/pipe_table.txt"));
    let result = classify_tier1(&features);
    assert_eq!(result.category, TextType::Structured);
    assert_eq!(result.tier, Tier::Structural);
    assert!(result.confidence >= 0.6, "confidence={}", result.confidence);
}

#[test]
fn pdf_dump_classified_as_artifact() {
    let features = extract_features(&read_fixture("pdf_dump/ocr_garbage.txt"));
    let result = classify_tier1(&features);
    assert_eq!(result.category, TextType::Artifact);
    assert_eq!(result.tier, Tier::Structural);
    assert!(
        result.confidence >= thresholds::ARTIFACT,
        "confidence={}",
        result.confidence
    );
}

#[test]
fn short_text_classified_as_skip() {
    // Short text is caught by classify() before reaching tier1
    let result = classify("hello world");
    assert_eq!(result.category, TextType::Skip);
    assert_eq!(result.confidence, 1.0);
}

#[test]
fn empty_text_classified_as_skip() {
    let result = classify("");
    assert_eq!(result.category, TextType::Skip);
}

#[test]
fn test_structured_csv() {
    let features = extract_features(&read_fixture("structured/csv_data.txt"));
    let result = classify_tier1(&features);
    assert_eq!(
        result.category,
        TextCategory::Structured,
        "CSV data should be Structured, got {:?} (reason: {})",
        result.category,
        result.reason
    );
}

#[test]
fn test_structured_json() {
    let features = extract_features(&read_fixture("structured/json_object.txt"));
    let result = classify_tier1(&features);
    assert_eq!(
        result.category,
        TextCategory::Structured,
        "JSON data should be Structured, got {:?} (reason: {})",
        result.category,
        result.reason
    );
}

#[test]
fn test_structured_key_value() {
    let features = extract_features(&read_fixture("structured/key_value.txt"));
    let result = classify_tier1(&features);
    // Key-value files with paths/URLs have high symbol_ratio → Code is reasonable
    assert!(
        result.category == TextCategory::Structured || result.category == TextCategory::Code,
        "Key-value data should be Structured or Code, got {:?} (reason: {})",
        result.category,
        result.reason
    );
}

#[test]
fn test_structured_log_lines() {
    let features = extract_features(&read_fixture("structured/log_lines.txt"));
    let result = classify_tier1(&features);
    assert_eq!(
        result.category,
        TextCategory::Structured,
        "Log lines should be Structured, got {:?} (reason: {})",
        result.category,
        result.reason
    );
}

#[test]
fn test_artifact_boilerplate() {
    let features = extract_features(&read_fixture("artifact/boilerplate.txt"));
    let result = classify_tier1(&features);
    assert_eq!(
        result.category,
        TextCategory::Artifact,
        "Boilerplate should be Artifact, got {:?} (reason: {})",
        result.category,
        result.reason
    );
}

#[test]
fn test_sub_type_csv() {
    let features = extract_features(&read_fixture("structured/csv_data.txt"));
    let result = classify_tier1(&features);
    assert_eq!(
        result.sub_type,
        Some(ContentSubType::Csv),
        "CSV data should have sub_type Csv, got {:?}",
        result.sub_type
    );
}

#[test]
fn test_sub_type_json() {
    let features = extract_features(&read_fixture("structured/json_object.txt"));
    let result = classify_tier1(&features);
    assert_eq!(
        result.sub_type,
        Some(ContentSubType::Json),
        "JSON data should have sub_type Json, got {:?}",
        result.sub_type
    );
}

#[test]
fn test_yaml_classified_as_code() {
    let features = extract_features(&read_fixture("code/yaml_nested.txt"));
    let result = classify_tier1(&features);
    assert_eq!(
        result.category,
        TextCategory::Code,
        "YAML should be Code (NOT Structured), got {:?} (reason: {})",
        result.category,
        result.reason
    );
}
