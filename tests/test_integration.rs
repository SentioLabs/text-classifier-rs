use text_classifier::Classifier;
use text_classifier::TextType;
use text_classifier::Tier;

/// Verify that Classifier::classify() uses per-type thresholds for Tier 1 acceptance.
/// A prose result at confidence >= 0.65 (PROSE threshold) should be accepted at Tier 1
/// without falling through to Tier 2.
#[test]
fn classifier_uses_per_type_thresholds_prose() {
    let clf = Classifier::new();
    // This is a standard prose fixture — Tier 1 should accept it at
    // the prose threshold (0.65) rather than the old single threshold (0.7).
    let result = clf.classify(&read_fixture("prose/simple.txt"));
    assert_eq!(result.category, TextType::Prose);
    assert!(result.confidence >= 0.65);
    assert_eq!(result.tier, Tier::Structural);
}

/// Verify that structured data uses the STRUCTURED threshold (0.60).
#[test]
fn classifier_uses_per_type_thresholds_structured() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("tabular/pipe_table.txt"));
    assert_eq!(result.category, TextType::Structured);
    assert!(result.confidence >= 0.60);
    assert_eq!(result.tier, Tier::Structural);
}

fn read_fixture(path: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{path}")).unwrap()
}

#[test]
fn end_to_end_prose() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("prose/simple.txt"));
    assert_eq!(result.category, TextType::Prose);
    assert!(result.confidence >= 0.7);
}

#[test]
fn end_to_end_python_code() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/python.txt"));
    assert_eq!(result.category, TextType::Code);
    assert!(result.confidence >= 0.7);
}

#[test]
fn end_to_end_html_code() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/html.txt"));
    assert_eq!(result.category, TextType::Code);
    assert!(result.confidence >= 0.7);
}

#[test]
fn end_to_end_pipe_table() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("tabular/pipe_table.txt"));
    assert_eq!(result.category, TextType::Structured);
    assert!(result.confidence >= 0.6);
}

#[test]
fn end_to_end_tsv_data() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("tabular/tsv_data.txt"));
    assert_eq!(result.category, TextType::Structured);
    assert!(result.confidence >= 0.6);
}

#[test]
fn end_to_end_pdf_dump() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("pdf_dump/ocr_garbage.txt"));
    assert_eq!(result.category, TextType::Artifact);
    assert!(result.confidence >= 0.7);
}

#[test]
fn short_text_is_skip() {
    let clf = Classifier::new();
    let result = clf.classify("hello");
    assert_eq!(result.category, TextType::Skip);
    assert_eq!(result.confidence, 1.0);
}

#[test]
fn empty_text_is_skip() {
    let clf = Classifier::new();
    let result = clf.classify("");
    assert_eq!(result.category, TextType::Skip);
}

#[test]
fn batch_classification_works() {
    let clf = Classifier::new();
    let texts = vec![
        "The quick brown fox jumps over the lazy dog. This is a sentence with proper punctuation. It has multiple sentences to establish a prose pattern.",
        "def foo(): pass",
        "hello",
    ];
    let results = clf.classify_batch(&texts);
    assert_eq!(results.len(), 3);
    // Third should be skip (too short)
    assert_eq!(results[2].category, TextType::Skip);
}

#[test]
fn end_to_end_javascript_code() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/javascript.txt"));
    assert_eq!(result.category, TextType::Code);
    assert!(result.confidence >= 0.7);
}

#[test]
fn end_to_end_rust_code() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/rust.txt"));
    assert_eq!(result.category, TextType::Code);
    assert!(result.confidence >= 0.7);
}

#[test]
fn end_to_end_sql_code() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/sql.txt"));
    assert_eq!(result.category, TextType::Code);
    assert!(result.confidence >= 0.7);
}

#[test]
fn end_to_end_shell_code() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/shell.txt"));
    assert_eq!(result.category, TextType::Code);
    assert!(result.confidence >= 0.7);
}

#[test]
fn end_to_end_yaml_config() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/yaml_config.txt"));
    assert_eq!(result.category, TextType::Code);
    assert!(result.confidence >= 0.7);
}

#[test]
fn end_to_end_minified_js() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/minified_js.txt"));
    assert_eq!(result.category, TextType::Code);
    assert!(result.confidence >= 0.7);
}

#[test]
fn features_extraction_returns_all_fields() {
    let clf = Classifier::new();
    let f = clf.extract_features(&read_fixture("prose/simple.txt"));
    // Verify all fields are populated (non-NaN, non-negative)
    assert!(f.line_length_cv >= 0.0);
    assert!(f.char_entropy >= 0.0);
    assert!(f.leading_whitespace_ratio >= 0.0);
    assert!(f.tab_density >= 0.0);
    assert!(f.sentence_punctuation_rate >= 0.0);
    assert!(f.paragraph_break_rate >= 0.0);
    assert!(f.alpha_ratio >= 0.0);
    assert!(f.line_uniqueness >= 0.0);
    assert!(f.short_line_ratio >= 0.0);
    assert!(f.symbol_ratio >= 0.0);
    assert!(f.line_count > 0);
}
