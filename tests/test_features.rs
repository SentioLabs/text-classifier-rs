use text_classifier::features::extract_features;

fn read_fixture(path: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{path}")).unwrap()
}

#[test]
fn prose_has_high_sentence_punctuation() {
    let features = extract_features(&read_fixture("prose/simple.txt"));
    assert!(
        features.sentence_punctuation_rate > 0.03,
        "prose sentence_punctuation_rate={} should be > 0.03",
        features.sentence_punctuation_rate
    );
}

#[test]
fn prose_has_high_alpha_ratio() {
    let features = extract_features(&read_fixture("prose/simple.txt"));
    assert!(
        features.alpha_ratio > 0.70,
        "prose alpha_ratio={} should be > 0.70",
        features.alpha_ratio
    );
}

#[test]
fn prose_has_high_line_length_cv() {
    let features = extract_features(&read_fixture("prose/simple.txt"));
    assert!(
        features.line_length_cv > 0.3,
        "prose line_length_cv={} should be > 0.3",
        features.line_length_cv
    );
}

#[test]
fn code_has_high_leading_whitespace() {
    let features = extract_features(&read_fixture("code/python.txt"));
    assert!(
        features.leading_whitespace_ratio > 0.3,
        "code leading_whitespace_ratio={} should be > 0.3",
        features.leading_whitespace_ratio
    );
}

#[test]
fn code_has_high_symbol_ratio() {
    let features = extract_features(&read_fixture("code/python.txt"));
    assert!(
        features.symbol_ratio > 0.05,
        "code symbol_ratio={} should be > 0.05",
        features.symbol_ratio
    );
}

#[test]
fn code_has_low_sentence_punctuation() {
    let features = extract_features(&read_fixture("code/python.txt"));
    assert!(
        features.sentence_punctuation_rate < 0.02,
        "code sentence_punctuation_rate={} should be < 0.02",
        features.sentence_punctuation_rate
    );
}

#[test]
fn tabular_has_low_line_length_cv() {
    let features = extract_features(&read_fixture("tabular/pipe_table.txt"));
    assert!(
        features.line_length_cv < 0.3,
        "tabular line_length_cv={} should be < 0.3",
        features.line_length_cv
    );
}

#[test]
fn tabular_has_low_sentence_punctuation() {
    let features = extract_features(&read_fixture("tabular/pipe_table.txt"));
    assert!(
        features.sentence_punctuation_rate < 0.01,
        "tabular sentence_punctuation_rate={} should be < 0.01",
        features.sentence_punctuation_rate
    );
}

#[test]
fn pdf_dump_has_high_short_line_ratio() {
    let features = extract_features(&read_fixture("pdf_dump/ocr_garbage.txt"));
    assert!(
        features.short_line_ratio > 0.5,
        "pdf_dump short_line_ratio={} should be > 0.5",
        features.short_line_ratio
    );
}

#[test]
fn empty_text_returns_zero_features() {
    let features = extract_features("");
    assert_eq!(features.line_length_cv, 0.0);
    assert_eq!(features.char_entropy, 0.0);
    assert_eq!(features.sentence_punctuation_rate, 0.0);
}
