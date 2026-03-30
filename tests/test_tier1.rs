use text_classifier::features::extract_features;
use text_classifier::tier1::classify_tier1;
use text_classifier::types::FeatureVector;
use text_classifier::{TextCategory, Tier, classify};

fn read_fixture(path: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{path}")).unwrap()
}

// --- High-confidence short-circuit tests ---

#[test]
fn pure_prose_high_confidence() {
    let features = extract_features(&read_fixture("prose/simple.txt"));
    let result = classify_tier1(&features);
    assert_eq!(result.category, TextCategory::Prose);
    assert_eq!(result.tier, Tier::Structural);
    assert!(
        result.confidence >= 0.95,
        "prose confidence should be >= 0.95, got {}",
        result.confidence
    );
}

#[test]
fn tsv_data_high_confidence() {
    let features = extract_features(&read_fixture("tabular/tsv_data.txt"));
    let result = classify_tier1(&features);
    assert_eq!(result.category, TextCategory::Structured);
    assert_eq!(result.tier, Tier::Structural);
    assert!(
        result.confidence >= 0.95,
        "TSV confidence should be >= 0.95, got {}",
        result.confidence
    );
}

#[test]
fn html_code_high_confidence() {
    let features = extract_features(&read_fixture("code/html.txt"));
    let result = classify_tier1(&features);
    assert_eq!(result.category, TextCategory::Code);
    assert_eq!(result.tier, Tier::Structural);
    assert!(
        result.confidence >= 0.95,
        "HTML confidence should be >= 0.95, got {}",
        result.confidence
    );
}

// --- Ambiguous text falls through to model ---

#[test]
fn ambiguous_text_low_confidence() {
    let features = FeatureVector {
        line_length_cv: 0.4,
        char_entropy: 4.0,
        leading_whitespace_ratio: 0.15,
        tab_density: 0.01,
        sentence_punctuation_rate: 0.03,
        paragraph_break_rate: 0.05,
        alpha_ratio: 0.70,
        line_uniqueness: 0.8,
        short_line_ratio: 0.2,
        symbol_ratio: 0.04,
        delimiter_consistency: 0.0,
        json_brace_depth: 0.0,
        key_value_ratio: 0.0,
        xml_tag_ratio: 0.0,
        log_line_ratio: 0.0,
        comment_ratio: 0.0,
        numeric_field_ratio: 0.0,
        repetitive_structure_score: 0.0,
        line_count: 10,
    };
    let result = classify_tier1(&features);
    assert!(
        result.confidence < 0.60,
        "ambiguous text confidence should be < 0.60, got {}",
        result.confidence
    );
}

// --- Default fallback behavior ---

#[test]
fn fallback_returns_low_confidence_skip() {
    let features = FeatureVector {
        line_length_cv: 0.5,
        char_entropy: 3.5,
        leading_whitespace_ratio: 0.05,
        tab_density: 0.0,
        sentence_punctuation_rate: 0.01,
        paragraph_break_rate: 0.02,
        alpha_ratio: 0.50,
        line_uniqueness: 0.9,
        short_line_ratio: 0.3,
        symbol_ratio: 0.05,
        delimiter_consistency: 0.0,
        json_brace_depth: 0.0,
        key_value_ratio: 0.0,
        xml_tag_ratio: 0.0,
        log_line_ratio: 0.0,
        comment_ratio: 0.0,
        numeric_field_ratio: 0.0,
        repetitive_structure_score: 0.0,
        line_count: 8,
    };
    let result = classify_tier1(&features);
    assert_eq!(result.category, TextCategory::Skip);
    assert!(
        result.confidence <= 0.40,
        "fallback confidence should be <= 0.40, got {}",
        result.confidence
    );
}

// --- Existing basic tests updated for new thresholds ---

#[test]
fn short_text_classified_as_skip() {
    let result = classify("hello world");
    assert_eq!(result.category, TextCategory::Skip);
    assert_eq!(result.confidence, 1.0);
}

#[test]
fn empty_text_classified_as_skip() {
    let result = classify("");
    assert_eq!(result.category, TextCategory::Skip);
}

#[test]
fn empty_features_classified_as_skip() {
    let features = FeatureVector::zeroed();
    let result = classify_tier1(&features);
    assert_eq!(result.category, TextCategory::Skip);
    assert_eq!(result.confidence, 1.0);
}
