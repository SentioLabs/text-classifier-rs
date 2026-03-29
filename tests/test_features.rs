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
    assert_eq!(features.delimiter_consistency, 0.0);
    assert_eq!(features.json_brace_depth, 0.0);
    assert_eq!(features.key_value_ratio, 0.0);
    assert_eq!(features.xml_tag_ratio, 0.0);
    assert_eq!(features.log_line_ratio, 0.0);
    assert_eq!(features.comment_ratio, 0.0);
    assert_eq!(features.numeric_field_ratio, 0.0);
    assert_eq!(features.repetitive_structure_score, 0.0);
}

#[test]
fn test_delimiter_consistency_csv() {
    let text = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,SF\nDave,40,CHI\n";
    let features = extract_features(text);
    assert!(
        features.delimiter_consistency > 0.8,
        "CSV delimiter_consistency={} should be > 0.8",
        features.delimiter_consistency
    );
}

#[test]
fn test_delimiter_consistency_prose() {
    let text = "The quick brown fox jumps over the lazy dog.\nShe sells seashells by the seashore.\nPeter Piper picked a peck of pickled peppers.\nHow much wood would a woodchuck chuck.\n";
    let features = extract_features(text);
    assert!(
        features.delimiter_consistency < 0.3,
        "prose delimiter_consistency={} should be < 0.3",
        features.delimiter_consistency
    );
}

#[test]
fn test_json_brace_depth() {
    let text = r#"{"name": "Alice", "age": 30, "address": {"city": "NYC", "state": "NY"}}"#;
    let features = extract_features(text);
    assert!(
        features.json_brace_depth > 0.05,
        "JSON json_brace_depth={} should be > 0.05",
        features.json_brace_depth
    );
}

#[test]
fn test_key_value_ratio_yaml() {
    let text = "name: Alice\nage: 30\ncity: NYC\nstate: NY\ncountry: USA\n";
    let features = extract_features(text);
    assert!(
        features.key_value_ratio > 0.5,
        "YAML key_value_ratio={} should be > 0.5",
        features.key_value_ratio
    );
}

#[test]
fn test_xml_tag_ratio_html() {
    let text =
        "<html>\n<head>\n<title>Test</title>\n</head>\n<body>\n<p>Hello</p>\n</body>\n</html>\n";
    let features = extract_features(text);
    assert!(
        features.xml_tag_ratio > 0.3,
        "HTML xml_tag_ratio={} should be > 0.3",
        features.xml_tag_ratio
    );
}

#[test]
fn test_log_line_ratio() {
    let text = "2024-01-15 10:30:00 INFO Starting service\n2024-01-15 10:30:01 DEBUG Connecting to DB\n2024-01-15 10:30:02 INFO Service ready\n2024-01-15 10:30:03 WARN High memory usage\n";
    let features = extract_features(text);
    assert!(
        features.log_line_ratio > 0.5,
        "log log_line_ratio={} should be > 0.5",
        features.log_line_ratio
    );
}

#[test]
fn test_comment_ratio_code() {
    let text = "# This is a comment\n# Another comment\ndef foo():\n    # inline comment\n    return 42\n# end\n";
    let features = extract_features(text);
    assert!(
        features.comment_ratio > 0.2,
        "code comment_ratio={} should be > 0.2",
        features.comment_ratio
    );
}

#[test]
fn test_numeric_field_ratio_data() {
    let text = "10 20 30 40\n50 60 70 80\n90 100 110 120\n";
    let features = extract_features(text);
    assert!(
        features.numeric_field_ratio > 0.3,
        "numeric data numeric_field_ratio={} should be > 0.3",
        features.numeric_field_ratio
    );
}

#[test]
fn test_repetitive_structure_score() {
    let text = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n";
    let features = extract_features(text);
    assert!(
        features.repetitive_structure_score > 0.7,
        "CSV repetitive_structure_score={} should be > 0.7",
        features.repetitive_structure_score
    );
}
