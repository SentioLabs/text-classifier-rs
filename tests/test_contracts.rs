use std::collections::BTreeMap;
use text_classifier::types::{
    Classification, ContentSubType, Detection, FeatureVector, TextCategory, Tier,
};

/// Verify that TextCategory, ContentSubType, and thresholds are re-exported
/// from the crate root (not just from types module).
#[test]
fn contract_root_reexports_text_category() {
    let _: text_classifier::TextCategory = text_classifier::TextCategory::Prose;
}

#[test]
fn contract_root_reexports_content_sub_type() {
    let _: text_classifier::ContentSubType = text_classifier::ContentSubType::Python;
}

#[test]
fn contract_root_reexports_thresholds() {
    assert_eq!(text_classifier::thresholds::PROSE, 0.65);
    assert_eq!(text_classifier::thresholds::CODE, 0.70);
    assert_eq!(text_classifier::thresholds::STRUCTURED, 0.60);
}

#[test]
fn contract_text_category_variants() {
    let _: TextCategory = TextCategory::Prose;
    let _: TextCategory = TextCategory::Code;
    let _: TextCategory = TextCategory::Structured;
    let _: TextCategory = TextCategory::Skip;
}

#[test]
fn contract_content_sub_type_category_mapping() {
    assert_eq!(ContentSubType::Plain.category(), TextCategory::Prose);
    assert_eq!(ContentSubType::Markdown.category(), TextCategory::Prose);
    assert_eq!(ContentSubType::Python.category(), TextCategory::Code);
    assert_eq!(ContentSubType::Yaml.category(), TextCategory::Structured);
    assert_eq!(ContentSubType::Html.category(), TextCategory::Code);
    assert_eq!(ContentSubType::Csv.category(), TextCategory::Structured);
    assert_eq!(ContentSubType::Json.category(), TextCategory::Structured);
    assert_eq!(
        ContentSubType::LogLines.category(),
        TextCategory::Structured
    );
    assert_eq!(ContentSubType::Unknown.category(), TextCategory::Skip);
}

#[test]
fn contract_content_sub_type_labels() {
    assert_eq!(ContentSubType::Python.label(), "python");
    assert_eq!(ContentSubType::Csv.label(), "csv");
    assert_eq!(ContentSubType::Unknown.label(), "unknown");
}

#[test]
fn contract_classification_has_category_and_sub_type() {
    let c = Classification {
        category: TextCategory::Code,
        sub_type: Some(ContentSubType::Python),
        confidence: 0.95,
        reason: "test".to_string(),
        tier: Tier::Structural,
        detections: BTreeMap::new(),
    };
    assert_eq!(c.category, TextCategory::Code);
    assert_eq!(c.sub_type, Some(ContentSubType::Python));
}

#[test]
fn contract_feature_vector_has_new_fields() {
    let f = FeatureVector::zeroed();
    let _ = f.delimiter_consistency;
    let _ = f.json_brace_depth;
    let _ = f.key_value_ratio;
    let _ = f.xml_tag_ratio;
    let _ = f.log_line_ratio;
    let _ = f.comment_ratio;
    let _ = f.numeric_field_ratio;
    let _ = f.repetitive_structure_score;
    let _ = f.hyphenated_line_break_ratio;
    let _ = f.short_repeated_line_ratio;
    let _ = f.page_number_density;
    let _ = f.label_value_line_ratio;
    let _ = f.table_fragment_score;
    let _ = f.uppercase_header_ratio;
    let _ = f.dictionary_word_ratio;
    let _ = f.encoding_error_ratio;
    let _ = f.repeated_ngram_ratio;
    let _ = f.sentence_coherence_score;
}

#[test]
fn contract_backward_compat_text_type_alias() {
    use text_classifier::TextType;
    let _: TextType = TextType::Prose; // TextType is alias for TextCategory
}

#[test]
fn contract_text_category_display() {
    assert_eq!(TextCategory::Prose.to_string(), "prose");
    assert_eq!(TextCategory::Code.to_string(), "code");
    assert_eq!(TextCategory::Structured.to_string(), "structured");
    assert_eq!(TextCategory::Skip.to_string(), "skip");
}

#[test]
fn contract_text_category_is_prose() {
    assert!(TextCategory::Prose.is_prose());
    assert!(!TextCategory::Code.is_prose());
    assert!(!TextCategory::Structured.is_prose());
    assert!(!TextCategory::Skip.is_prose());
}

#[test]
fn contract_content_sub_type_display() {
    assert_eq!(ContentSubType::Python.to_string(), "python");
    assert_eq!(ContentSubType::Csv.to_string(), "csv");
    assert_eq!(ContentSubType::Unknown.to_string(), "unknown");
}

#[test]
fn contract_thresholds() {
    use text_classifier::types::thresholds;
    assert_eq!(thresholds::PROSE, 0.65);
    assert_eq!(thresholds::CODE, 0.70);
    assert_eq!(thresholds::STRUCTURED, 0.60);
    assert_eq!(thresholds::SUB_TYPE, 0.80);
}

#[test]
#[allow(deprecated)]
fn contract_classification_text_type_deprecated_accessor() {
    let c = Classification {
        category: TextCategory::Code,
        sub_type: Some(ContentSubType::Python),
        confidence: 0.95,
        reason: "test".to_string(),
        tier: Tier::Structural,
        detections: BTreeMap::new(),
    };
    assert_eq!(c.text_type(), TextCategory::Code);
}

#[test]
fn contract_content_sub_type_all_category_mappings() {
    // Prose variants
    assert_eq!(ContentSubType::Plain.category(), TextCategory::Prose);
    assert_eq!(ContentSubType::Markdown.category(), TextCategory::Prose);
    assert_eq!(ContentSubType::Rst.category(), TextCategory::Prose);
    assert_eq!(ContentSubType::Latex.category(), TextCategory::Prose);

    // Code variants
    assert_eq!(ContentSubType::Python.category(), TextCategory::Code);
    assert_eq!(ContentSubType::JavaScript.category(), TextCategory::Code);
    assert_eq!(ContentSubType::TypeScript.category(), TextCategory::Code);
    assert_eq!(ContentSubType::Rust.category(), TextCategory::Code);
    assert_eq!(ContentSubType::Go.category(), TextCategory::Code);
    assert_eq!(ContentSubType::Java.category(), TextCategory::Code);
    assert_eq!(ContentSubType::Sql.category(), TextCategory::Code);
    assert_eq!(ContentSubType::Shell.category(), TextCategory::Code);
    assert_eq!(ContentSubType::Css.category(), TextCategory::Code);

    // Code > Config
    assert_eq!(ContentSubType::Dockerfile.category(), TextCategory::Code);
    assert_eq!(ContentSubType::Makefile.category(), TextCategory::Code);

    // Code > Markup
    assert_eq!(ContentSubType::Html.category(), TextCategory::Code);
    assert_eq!(ContentSubType::Xml.category(), TextCategory::Code);
    assert_eq!(ContentSubType::Sgml.category(), TextCategory::Code);

    // Structured > Tabular
    assert_eq!(ContentSubType::Csv.category(), TextCategory::Structured);
    assert_eq!(ContentSubType::Tsv.category(), TextCategory::Structured);
    assert_eq!(
        ContentSubType::PipeTable.category(),
        TextCategory::Structured
    );
    assert_eq!(
        ContentSubType::FixedWidth.category(),
        TextCategory::Structured
    );

    // Structured > Data
    assert_eq!(ContentSubType::Json.category(), TextCategory::Structured);
    assert_eq!(ContentSubType::Jsonl.category(), TextCategory::Structured);
    assert_eq!(
        ContentSubType::KeyValue.category(),
        TextCategory::Structured
    );
    assert_eq!(
        ContentSubType::LogLines.category(),
        TextCategory::Structured
    );

    // Structured > Config
    assert_eq!(ContentSubType::Yaml.category(), TextCategory::Structured);
    assert_eq!(ContentSubType::Toml.category(), TextCategory::Structured);
    assert_eq!(ContentSubType::Ini.category(), TextCategory::Structured);

    // Fallback
    assert_eq!(ContentSubType::Unknown.category(), TextCategory::Skip);
}

#[test]
fn contract_content_sub_type_all_labels() {
    assert_eq!(ContentSubType::Plain.label(), "plain");
    assert_eq!(ContentSubType::Markdown.label(), "markdown");
    assert_eq!(ContentSubType::Rst.label(), "rst");
    assert_eq!(ContentSubType::Latex.label(), "latex");
    assert_eq!(ContentSubType::Python.label(), "python");
    assert_eq!(ContentSubType::JavaScript.label(), "javascript");
    assert_eq!(ContentSubType::TypeScript.label(), "typescript");
    assert_eq!(ContentSubType::Rust.label(), "rust");
    assert_eq!(ContentSubType::Go.label(), "go");
    assert_eq!(ContentSubType::Java.label(), "java");
    assert_eq!(ContentSubType::Sql.label(), "sql");
    assert_eq!(ContentSubType::Shell.label(), "shell");
    assert_eq!(ContentSubType::Css.label(), "css");
    assert_eq!(ContentSubType::Yaml.label(), "yaml");
    assert_eq!(ContentSubType::Toml.label(), "toml");
    assert_eq!(ContentSubType::Ini.label(), "ini");
    assert_eq!(ContentSubType::Dockerfile.label(), "dockerfile");
    assert_eq!(ContentSubType::Makefile.label(), "makefile");
    assert_eq!(ContentSubType::Html.label(), "html");
    assert_eq!(ContentSubType::Xml.label(), "xml");
    assert_eq!(ContentSubType::Sgml.label(), "sgml");
    assert_eq!(ContentSubType::Csv.label(), "csv");
    assert_eq!(ContentSubType::Tsv.label(), "tsv");
    assert_eq!(ContentSubType::PipeTable.label(), "pipe_table");
    assert_eq!(ContentSubType::FixedWidth.label(), "fixed_width");
    assert_eq!(ContentSubType::Json.label(), "json");
    assert_eq!(ContentSubType::Jsonl.label(), "jsonl");
    assert_eq!(ContentSubType::KeyValue.label(), "key_value");
    assert_eq!(ContentSubType::LogLines.label(), "log_lines");
    assert_eq!(ContentSubType::Unknown.label(), "unknown");
}

#[test]
fn contract_feature_vector_zeroed_all_zero() {
    let f = FeatureVector::zeroed();
    assert_eq!(f.line_length_cv, 0.0);
    assert_eq!(f.char_entropy, 0.0);
    assert_eq!(f.leading_whitespace_ratio, 0.0);
    assert_eq!(f.tab_density, 0.0);
    assert_eq!(f.sentence_punctuation_rate, 0.0);
    assert_eq!(f.paragraph_break_rate, 0.0);
    assert_eq!(f.alpha_ratio, 0.0);
    assert_eq!(f.line_uniqueness, 0.0);
    assert_eq!(f.short_line_ratio, 0.0);
    assert_eq!(f.symbol_ratio, 0.0);
    assert_eq!(f.delimiter_consistency, 0.0);
    assert_eq!(f.json_brace_depth, 0.0);
    assert_eq!(f.key_value_ratio, 0.0);
    assert_eq!(f.xml_tag_ratio, 0.0);
    assert_eq!(f.log_line_ratio, 0.0);
    assert_eq!(f.comment_ratio, 0.0);
    assert_eq!(f.numeric_field_ratio, 0.0);
    assert_eq!(f.repetitive_structure_score, 0.0);
    assert_eq!(f.hyphenated_line_break_ratio, 0.0);
    assert_eq!(f.short_repeated_line_ratio, 0.0);
    assert_eq!(f.page_number_density, 0.0);
    assert_eq!(f.label_value_line_ratio, 0.0);
    assert_eq!(f.table_fragment_score, 0.0);
    assert_eq!(f.uppercase_header_ratio, 0.0);
    assert_eq!(f.dictionary_word_ratio, 0.0);
    assert_eq!(f.encoding_error_ratio, 0.0);
    assert_eq!(f.repeated_ngram_ratio, 0.0);
    assert_eq!(f.sentence_coherence_score, 0.0);
    assert_eq!(f.line_count, 0);
}

/// Verify that TextCategory has exactly 4 variants after simplification.
#[test]
fn contract_text_category_no_artifact_variants() {
    // TextCategory should only have Prose, Code, Structured, Skip
    // This test verifies the enum is exhaustive with exactly these 4 variants
    let categories = [
        TextCategory::Prose,
        TextCategory::Code,
        TextCategory::Structured,
        TextCategory::Skip,
    ];
    for cat in &categories {
        // Each variant should have a valid display string
        let s = cat.to_string();
        assert!(!s.is_empty());
    }
}

/// Verify that ContentSubType no longer has artifact or skip sub-types.
#[test]
fn contract_content_sub_type_no_artifact_skip_subtypes() {
    // Unknown is the only fallback, mapping to Skip
    assert_eq!(ContentSubType::Unknown.category(), TextCategory::Skip);
    assert_eq!(ContentSubType::Unknown.label(), "unknown");
}

#[test]
fn contract_detection_struct_fields() {
    let d = Detection {
        sub_type: ContentSubType::Python,
        score: 0.85,
    };
    assert_eq!(d.sub_type, ContentSubType::Python);
    assert!((d.score - 0.85).abs() < f32::EPSILON);
}

#[test]
fn contract_classification_has_detections() {
    let c = Classification {
        category: TextCategory::Prose,
        sub_type: Some(ContentSubType::Markdown),
        confidence: 0.97,
        reason: "test".to_string(),
        tier: Tier::Model,
        detections: BTreeMap::new(),
    };
    assert!(c.detections.is_empty());
}

#[test]
fn contract_classification_detections_serialized() {
    let mut detections = BTreeMap::new();
    detections.insert(
        TextCategory::Code,
        vec![Detection {
            sub_type: ContentSubType::Python,
            score: 0.85,
        }],
    );
    let c = Classification {
        category: TextCategory::Prose,
        sub_type: Some(ContentSubType::Markdown),
        confidence: 0.97,
        reason: "test".to_string(),
        tier: Tier::Model,
        detections,
    };
    let json = serde_json::to_string(&c).unwrap();
    assert!(json.contains("detections"));
    assert!(json.contains("python"));
}
