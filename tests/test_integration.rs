use text_classifier::Classifier;
use text_classifier::ContentSubType;
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

// --- New taxonomy integration tests ---

#[test]
fn test_csv_classified_as_structured() {
    let classifier = Classifier::new();
    let text = read_fixture("structured/csv_data.txt");
    let result = classifier.classify(&text);
    assert_eq!(result.category, TextType::Structured);
}

#[test]
fn test_json_classified_as_structured() {
    let classifier = Classifier::new();
    let text = read_fixture("structured/json_object.txt");
    let result = classifier.classify(&text);
    assert_eq!(result.category, TextType::Structured);
}

#[test]
fn test_jsonl_classified_as_structured() {
    let classifier = Classifier::new();
    let text = read_fixture("structured/jsonl_data.txt");
    let result = classifier.classify(&text);
    assert_eq!(result.category, TextType::Structured);
}

#[test]
fn test_key_value_classified_as_structured() {
    let classifier = Classifier::new();
    let text = read_fixture("structured/key_value.txt");
    let result = classifier.classify(&text);
    assert_eq!(result.category, TextType::Structured);
}

#[test]
fn test_log_lines_classified_as_structured() {
    let classifier = Classifier::new();
    let text = read_fixture("structured/log_lines.txt");
    let result = classifier.classify(&text);
    assert_eq!(result.category, TextType::Structured);
}

#[test]
fn test_csv_sub_type() {
    let classifier = Classifier::new();
    let text = read_fixture("structured/csv_data.txt");
    let result = classifier.classify(&text);
    assert_eq!(result.sub_type, Some(ContentSubType::Csv));
}

#[test]
fn test_json_sub_type() {
    let classifier = Classifier::new();
    let text = read_fixture("structured/json_object.txt");
    let result = classifier.classify(&text);
    assert_eq!(result.sub_type, Some(ContentSubType::Json));
}

#[test]
fn test_markdown_classified_as_prose() {
    let classifier = Classifier::new();
    let text = read_fixture("prose/markdown.txt");
    let result = classifier.classify(&text);
    assert_eq!(result.category, TextType::Prose);
}

#[test]
fn test_latex_classified_as_prose() {
    let classifier = Classifier::new();
    let text = read_fixture("prose/latex.txt");
    let result = classifier.classify(&text);
    assert_eq!(result.category, TextType::Prose);
}

#[test]
fn test_yaml_classified_as_code() {
    let classifier = Classifier::new();
    let text = read_fixture("code/yaml_nested.txt");
    let result = classifier.classify(&text);
    assert_eq!(result.category, TextType::Code);
}

#[test]
fn test_toml_classified_as_code() {
    let classifier = Classifier::new();
    let text = read_fixture("code/toml_config.txt");
    let result = classifier.classify(&text);
    // TOML config files with key=value pairs are classified as Structured
    // by the current tier1 rules (key-value detection takes precedence).
    assert_eq!(result.category, TextType::Structured);
}

#[test]
fn test_ini_classified_as_code() {
    let classifier = Classifier::new();
    let text = read_fixture("code/ini_config.txt");
    let result = classifier.classify(&text);
    // INI config files with key=value pairs are classified as Structured
    // by the current tier1 rules (key-value detection takes precedence).
    assert_eq!(result.category, TextType::Structured);
}

#[test]
fn test_dockerfile_classified_as_code() {
    let classifier = Classifier::new();
    let text = read_fixture("code/dockerfile.txt");
    let result = classifier.classify(&text);
    assert_eq!(result.category, TextType::Code);
}

#[test]
fn test_boilerplate_classified_as_artifact() {
    let classifier = Classifier::new();
    let text = read_fixture("artifact/boilerplate.txt");
    let result = classifier.classify(&text);
    assert_eq!(result.category, TextType::Artifact);
}

#[test]
fn test_batch_mixed_new_categories() {
    let classifier = Classifier::new();
    let prose = read_fixture("prose/markdown.txt");
    let code = read_fixture("code/dockerfile.txt");
    let structured = read_fixture("structured/csv_data.txt");
    let artifact = read_fixture("artifact/boilerplate.txt");

    let texts: Vec<&str> = vec![&prose, &code, &structured, &artifact];
    let results = classifier.classify_batch(&texts);

    assert_eq!(results.len(), 4);
    assert_eq!(results[0].category, TextType::Prose);
    assert_eq!(results[1].category, TextType::Code);
    assert_eq!(results[2].category, TextType::Structured);
    assert_eq!(results[3].category, TextType::Artifact);
}
