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

// ---------------------------------------------------------------------------
// Tests for 10 new features
// ---------------------------------------------------------------------------

#[test]
fn test_hyphenated_line_break_ratio_positive() {
    let text = "This is a hyphen-\nated line break.\nAnother hyphen-\nated break here.\n";
    let features = extract_features(text);
    assert!(
        features.hyphenated_line_break_ratio > 0.0,
        "hyphenated_line_break_ratio={} should be > 0.0",
        features.hyphenated_line_break_ratio
    );
}

#[test]
fn test_hyphenated_line_break_ratio_zero_for_normal_text() {
    let text = "Normal line one.\nNormal line two.\nNormal line three.\n";
    let features = extract_features(text);
    assert!(
        features.hyphenated_line_break_ratio < 0.01,
        "hyphenated_line_break_ratio={} should be ~0.0",
        features.hyphenated_line_break_ratio
    );
}

#[test]
fn test_hyphenated_line_break_ratio_empty() {
    let features = extract_features("");
    assert_eq!(features.hyphenated_line_break_ratio, 0.0);
}

#[test]
fn test_short_repeated_line_ratio_positive() {
    let text = "Header\nSome content here that is longer.\nHeader\nMore content.\nHeader\nFooter\nFooter\n";
    let features = extract_features(text);
    assert!(
        features.short_repeated_line_ratio > 0.0,
        "short_repeated_line_ratio={} should be > 0.0",
        features.short_repeated_line_ratio
    );
}

#[test]
fn test_short_repeated_line_ratio_zero_for_unique_lines() {
    let text = "Alpha\nBravo\nCharlie\nDelta\nEcho\n";
    let features = extract_features(text);
    assert_eq!(
        features.short_repeated_line_ratio, 0.0,
        "short_repeated_line_ratio={} should be 0.0 for unique lines",
        features.short_repeated_line_ratio
    );
}

#[test]
fn test_short_repeated_line_ratio_empty() {
    let features = extract_features("");
    assert_eq!(features.short_repeated_line_ratio, 0.0);
}

#[test]
fn test_page_number_density_positive() {
    let text = "1\nSome text here.\n2\nMore text here.\n3\nPage 4\nPage 5 of 10\n";
    let features = extract_features(text);
    assert!(
        features.page_number_density > 0.0,
        "page_number_density={} should be > 0.0",
        features.page_number_density
    );
}

#[test]
fn test_page_number_density_zero_for_prose() {
    let text = "The quick brown fox jumps over the lazy dog.\nShe sells seashells by the seashore.\nPeter Piper picked a peck of pickled peppers.\n";
    let features = extract_features(text);
    assert_eq!(
        features.page_number_density, 0.0,
        "page_number_density={} should be 0.0 for prose",
        features.page_number_density
    );
}

#[test]
fn test_page_number_density_empty() {
    let features = extract_features("");
    assert_eq!(features.page_number_density, 0.0);
}

#[test]
fn test_page_number_density_fraction_format() {
    let text = "Some text\n3 / 10\nMore text\n5/20\n";
    let features = extract_features(text);
    assert!(
        features.page_number_density > 0.0,
        "page_number_density={} should detect fraction format",
        features.page_number_density
    );
}

#[test]
fn test_label_value_line_ratio_positive() {
    let text = "Name: John Doe\nAge: 30\nCity: New York\nState: NY\nCountry: USA\n";
    let features = extract_features(text);
    assert!(
        features.label_value_line_ratio > 0.0,
        "label_value_line_ratio={} should be > 0.0",
        features.label_value_line_ratio
    );
}

#[test]
fn test_label_value_line_ratio_zero_for_code() {
    let text = "def foo():\n    return 42\n\ndef bar():\n    return 99\n";
    let features = extract_features(text);
    assert!(
        features.label_value_line_ratio < 0.1,
        "label_value_line_ratio={} should be low for code",
        features.label_value_line_ratio
    );
}

#[test]
fn test_label_value_line_ratio_empty() {
    let features = extract_features("");
    assert_eq!(features.label_value_line_ratio, 0.0);
}

#[test]
fn test_table_fragment_score_positive() {
    let text = "name,age,city\nAlice,30,NYC\nBob,25,LA\n";
    let features = extract_features(text);
    assert!(
        features.table_fragment_score > 0.0,
        "table_fragment_score={} should be > 0.0",
        features.table_fragment_score
    );
}

#[test]
fn test_table_fragment_score_multi_space_columns() {
    let text = "Name       Age    City\nAlice      30     NYC\nBob        25     LA\n";
    let features = extract_features(text);
    assert!(
        features.table_fragment_score > 0.0,
        "table_fragment_score={} should detect multi-space columns",
        features.table_fragment_score
    );
}

#[test]
fn test_table_fragment_score_zero_for_prose() {
    let text = "The quick brown fox.\nShe sells seashells.\n";
    let features = extract_features(text);
    assert!(
        features.table_fragment_score < 0.01,
        "table_fragment_score={} should be ~0.0 for prose",
        features.table_fragment_score
    );
}

#[test]
fn test_table_fragment_score_empty() {
    let features = extract_features("");
    assert_eq!(features.table_fragment_score, 0.0);
}

#[test]
fn test_uppercase_header_ratio_positive() {
    let text = "INTRODUCTION\nSome body text here.\nMETHODOLOGY\nMore body text.\nCONCLUSION\n";
    let features = extract_features(text);
    assert!(
        features.uppercase_header_ratio > 0.0,
        "uppercase_header_ratio={} should be > 0.0",
        features.uppercase_header_ratio
    );
}

#[test]
fn test_uppercase_header_ratio_zero_for_lowercase() {
    let text = "this is all lowercase.\nanother lowercase line.\nyet another one.\n";
    let features = extract_features(text);
    assert_eq!(
        features.uppercase_header_ratio, 0.0,
        "uppercase_header_ratio={} should be 0.0 for lowercase text",
        features.uppercase_header_ratio
    );
}

#[test]
fn test_uppercase_header_ratio_empty() {
    let features = extract_features("");
    assert_eq!(features.uppercase_header_ratio, 0.0);
}

#[test]
fn test_uppercase_header_excludes_sentences_ending_with_period() {
    let text = "THIS IS A SENTENCE.\nANOTHER SENTENCE!\nYET ANOTHER?\n";
    let features = extract_features(text);
    assert_eq!(
        features.uppercase_header_ratio, 0.0,
        "uppercase_header_ratio={} should be 0.0 for lines ending with .!?",
        features.uppercase_header_ratio
    );
}

#[test]
fn test_dictionary_word_ratio_positive() {
    let text = "the quick brown fox jumps over the lazy dog";
    let features = extract_features(text);
    assert!(
        features.dictionary_word_ratio > 0.5,
        "dictionary_word_ratio={} should be > 0.5 for English prose",
        features.dictionary_word_ratio
    );
}

#[test]
fn test_dictionary_word_ratio_low_for_gibberish() {
    let text = "xyzzy plugh qwerty asdfgh zxcvbn mnbvcx";
    let features = extract_features(text);
    assert!(
        features.dictionary_word_ratio < 0.3,
        "dictionary_word_ratio={} should be low for gibberish",
        features.dictionary_word_ratio
    );
}

#[test]
fn test_dictionary_word_ratio_empty() {
    let features = extract_features("");
    assert_eq!(features.dictionary_word_ratio, 0.0);
}

#[test]
fn test_encoding_error_ratio_positive() {
    let text = "Some text with \u{FFFD} replacement chars \u{FFFD} and more \u{FFFD}";
    let features = extract_features(text);
    assert!(
        features.encoding_error_ratio > 0.0,
        "encoding_error_ratio={} should be > 0.0 for text with replacement chars",
        features.encoding_error_ratio
    );
}

#[test]
fn test_encoding_error_ratio_mojibake() {
    let text = "Caf\u{00c3}\u{00a9} is a nice place. Temperature is 20\u{00c2}\u{00b0}C.";
    let features = extract_features(text);
    assert!(
        features.encoding_error_ratio > 0.0,
        "encoding_error_ratio={} should detect mojibake",
        features.encoding_error_ratio
    );
}

#[test]
fn test_encoding_error_ratio_zero_for_clean_text() {
    let text = "This is perfectly clean English text with no encoding errors at all.";
    let features = extract_features(text);
    assert_eq!(
        features.encoding_error_ratio, 0.0,
        "encoding_error_ratio={} should be 0.0 for clean text",
        features.encoding_error_ratio
    );
}

#[test]
fn test_encoding_error_ratio_empty() {
    let features = extract_features("");
    assert_eq!(features.encoding_error_ratio, 0.0);
}

#[test]
fn test_repeated_ngram_ratio_positive() {
    let text = "the cat sat the cat sat the cat sat on the mat";
    let features = extract_features(text);
    assert!(
        features.repeated_ngram_ratio > 0.0,
        "repeated_ngram_ratio={} should be > 0.0 for repetitive text",
        features.repeated_ngram_ratio
    );
}

#[test]
fn test_repeated_ngram_ratio_zero_for_unique_text() {
    let text = "alpha bravo charlie delta echo foxtrot golf hotel india";
    let features = extract_features(text);
    assert_eq!(
        features.repeated_ngram_ratio, 0.0,
        "repeated_ngram_ratio={} should be 0.0 for all unique ngrams",
        features.repeated_ngram_ratio
    );
}

#[test]
fn test_repeated_ngram_ratio_empty() {
    let features = extract_features("");
    assert_eq!(features.repeated_ngram_ratio, 0.0);
}

#[test]
fn test_repeated_ngram_ratio_short_text() {
    let text = "just two";
    let features = extract_features(text);
    assert_eq!(
        features.repeated_ngram_ratio, 0.0,
        "repeated_ngram_ratio={} should be 0.0 for < 3 words",
        features.repeated_ngram_ratio
    );
}

#[test]
fn test_sentence_coherence_score_positive() {
    let text = "This is a sentence.\nAnother good sentence!\nIs this a question?\n";
    let features = extract_features(text);
    assert!(
        features.sentence_coherence_score > 0.5,
        "sentence_coherence_score={} should be > 0.5 for proper sentences",
        features.sentence_coherence_score
    );
}

#[test]
fn test_sentence_coherence_score_zero_for_fragments() {
    let text = "no caps here\nalso no caps\nstill nothing\n";
    let features = extract_features(text);
    assert_eq!(
        features.sentence_coherence_score, 0.0,
        "sentence_coherence_score={} should be 0.0 for fragments",
        features.sentence_coherence_score
    );
}

#[test]
fn test_sentence_coherence_score_empty() {
    let features = extract_features("");
    assert_eq!(features.sentence_coherence_score, 0.0);
}

#[test]
fn test_empty_text_returns_zero_for_new_features() {
    let features = extract_features("");
    assert_eq!(features.hyphenated_line_break_ratio, 0.0);
    assert_eq!(features.short_repeated_line_ratio, 0.0);
    assert_eq!(features.page_number_density, 0.0);
    assert_eq!(features.label_value_line_ratio, 0.0);
    assert_eq!(features.table_fragment_score, 0.0);
    assert_eq!(features.uppercase_header_ratio, 0.0);
    assert_eq!(features.dictionary_word_ratio, 0.0);
    assert_eq!(features.encoding_error_ratio, 0.0);
    assert_eq!(features.repeated_ngram_ratio, 0.0);
    assert_eq!(features.sentence_coherence_score, 0.0);
}

#[test]
fn test_single_line_no_panic() {
    let features = extract_features("Single line of text.");
    assert!(features.hyphenated_line_break_ratio >= 0.0);
    assert!(features.short_repeated_line_ratio >= 0.0);
    assert!(features.page_number_density >= 0.0);
    assert!(features.label_value_line_ratio >= 0.0);
    assert!(features.table_fragment_score >= 0.0);
    assert!(features.uppercase_header_ratio >= 0.0);
    assert!(features.dictionary_word_ratio >= 0.0);
    assert!(features.encoding_error_ratio >= 0.0);
    assert!(features.repeated_ngram_ratio >= 0.0);
    assert!(features.sentence_coherence_score >= 0.0);
}

#[test]
fn test_section_header_ratio_ini() {
    let text = "[section]\nkey=value\n[other]\nkey2=value2";
    let features = extract_features(text);
    assert!(
        (features.section_header_ratio - 0.5).abs() < 0.01,
        "section_header_ratio={} should be ~0.5 for INI with 2/4 section headers",
        features.section_header_ratio
    );
}

#[test]
fn test_section_header_ratio_key_value() {
    let text = "key=value\nkey2=value2";
    let features = extract_features(text);
    assert_eq!(
        features.section_header_ratio, 0.0,
        "section_header_ratio={} should be 0.0 for plain key-value text",
        features.section_header_ratio
    );
}

#[test]
fn test_json_lines_ratio_jsonl() {
    let text = "{\"a\":1}\n{\"b\":2}\n{\"c\":3}";
    let features = extract_features(text);
    assert!(
        (features.json_lines_ratio - 1.0).abs() < 0.01,
        "json_lines_ratio={} should be ~1.0 for pure JSONL",
        features.json_lines_ratio
    );
}

#[test]
fn test_json_lines_ratio_code() {
    let text = "function foo() {\n  return 1;\n}";
    let features = extract_features(text);
    assert_eq!(
        features.json_lines_ratio, 0.0,
        "json_lines_ratio={} should be 0.0 for code",
        features.json_lines_ratio
    );
}
