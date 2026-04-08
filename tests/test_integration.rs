use text_classifier::Classifier;
use text_classifier::TextCategory;

fn read_fixture(path: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{path}")).unwrap()
}

// --- Tier 1 short-circuit cases (high confidence, no model needed) ---

#[test]
fn end_to_end_prose() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("prose/simple.txt"));
    assert_eq!(result.category, TextCategory::Prose);
    assert!(result.confidence >= 0.95);
}

#[test]
fn end_to_end_html_code() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/html.txt"));
    assert_eq!(result.category, TextCategory::Code);
    assert!(result.confidence >= 0.95);
}

#[test]
fn end_to_end_pipe_table() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("tabular/pipe_table.txt"));
    assert_eq!(result.category, TextCategory::Structured);
    assert!(result.confidence >= 0.95);
}

#[test]
fn end_to_end_tsv_data() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("tabular/tsv_data.txt"));
    assert_eq!(result.category, TextCategory::Structured);
    assert!(result.confidence >= 0.95);
}

// --- Cases that now defer to Tier 2 model ---

#[test]
fn python_code_defers_to_model() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/python.txt"));
    assert!(
        result.confidence <= 0.5,
        "without model, python code should get low-confidence fallback, got {}",
        result.confidence
    );
}

#[test]
fn javascript_code_defers_to_model() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/javascript.txt"));
    assert!(
        result.confidence <= 0.5,
        "without model, JS code should get low-confidence fallback, got {}",
        result.confidence
    );
}

#[test]
fn sql_code_defers_to_model() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/sql.txt"));
    assert!(
        result.confidence <= 0.5,
        "without model, SQL should get low-confidence fallback, got {}",
        result.confidence
    );
}

#[test]
fn yaml_config_defers_to_model() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/yaml_config.txt"));
    assert!(
        result.confidence <= 0.5,
        "without model, YAML should get low-confidence fallback, got {}",
        result.confidence
    );
}

#[test]
fn minified_js_defers_to_model() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/minified_js.txt"));
    assert!(
        result.confidence <= 0.5,
        "without model, minified JS should get low-confidence fallback, got {}",
        result.confidence
    );
}

#[test]
fn pdf_dump_defers_to_model() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("pdf_dump/ocr_garbage.txt"));
    assert!(
        result.confidence <= 0.5,
        "without model, PDF dump should get low-confidence fallback, got {}",
        result.confidence
    );
}

// --- Tier 1 short-circuits for indented code with high symbol ratio ---

#[test]
fn end_to_end_rust_code() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/rust.txt"));
    assert_eq!(result.category, TextCategory::Code);
    assert!(result.confidence >= 0.95);
}

#[test]
fn end_to_end_shell_code() {
    let clf = Classifier::new();
    let result = clf.classify(&read_fixture("code/shell.txt"));
    assert!(result.confidence > 0.0);
}

// --- Basic skip tests ---

#[test]
fn short_text_is_low_confidence_prose() {
    let clf = Classifier::new();
    let result = clf.classify("hello");
    assert_eq!(result.category, TextCategory::Prose);
    assert!((result.confidence - 0.5).abs() < f32::EPSILON);
}

#[test]
fn empty_text_is_zero_confidence_prose() {
    let clf = Classifier::new();
    let result = clf.classify("");
    assert_eq!(result.category, TextCategory::Prose);
    assert_eq!(result.confidence, 0.0);
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
    assert_eq!(results[2].category, TextCategory::Prose);
}

#[test]
fn features_extraction_works() {
    let text = &read_fixture("prose/simple.txt");
    let f = text_classifier::extract_features(text);
    assert!(f.line_length_cv >= 0.0);
    assert!(f.char_entropy >= 0.0);
    assert!(f.alpha_ratio >= 0.0);
    assert!(f.line_count > 0);
}
