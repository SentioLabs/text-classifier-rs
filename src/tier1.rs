use crate::types::{Classification, ContentSubType, FeatureVector, TextCategory, Tier};

pub use crate::types::thresholds;

/// Minimum confidence to accept a Tier 1 classification.
/// Still used by lib.rs for the Tier 1 → Tier 2 handoff.
pub const MIN_CONFIDENCE: f32 = 0.7;

/// Classify text using Tier 1 structural features with a two-pass approach.
///
/// Pass 1: Detect the broad `TextCategory` using priority-ordered rules.
/// Pass 2: Refine into a `ContentSubType` when confidence is sufficient.
///
/// Returns a Classification. Low-confidence results indicate ambiguous input —
/// `Classifier::classify()` uses this as a signal to fall through to Tier 2.
pub fn classify_tier1(features: &FeatureVector) -> Classification {
    // Pass 1: category detection in priority order
    // skip → structured → code → artifact → prose → fallback
    let mut result = None;

    // Skip: empty or no content
    if features.alpha_ratio == 0.0 && features.sentence_punctuation_rate == 0.0 {
        return Classification {
            category: TextCategory::Skip,
            sub_type: None,
            confidence: 1.0,
            reason: "empty or no content".to_string(),
            tier: Tier::Structural,
        };
    }

    // Try each category in priority order, accept first with sufficient confidence
    let candidates: [Option<Classification>; 4] = [
        try_structured(features),
        try_code(features),
        try_artifact(features),
        try_prose(features),
    ];

    for candidate in candidates.into_iter().flatten() {
        let threshold = match candidate.category {
            TextCategory::Prose => thresholds::PROSE,
            TextCategory::Code => thresholds::CODE,
            TextCategory::Structured => thresholds::STRUCTURED,
            TextCategory::Artifact => thresholds::ARTIFACT,
            _ => MIN_CONFIDENCE,
        };
        if candidate.confidence >= threshold {
            result = Some(candidate);
            break;
        }
    }

    // Pass 2: refine sub-type if category was detected
    match result {
        Some(mut classification) => {
            classification.sub_type = refine_sub_type(classification.category, features);
            classification
        }
        None => fallback_classification(features),
    }
}

/// Check if text is too short for reliable feature extraction.
/// Requires both few whitespace-delimited tokens AND few characters.
/// The character check prevents CJK/Thai text (which lacks spaces between
/// words) from being wrongly skipped.
pub fn is_too_short(text: &str) -> bool {
    text.split_whitespace().count() < 5 && text.chars().count() < 20
}

fn try_structured(f: &FeatureVector) -> Option<Classification> {
    // Guard: high XML tag ratio means markup → Code
    if f.xml_tag_ratio > 0.3 {
        return None;
    }

    // Guard: key-value with significant indentation means config (YAML/TOML) → Code
    // But NOT if json_brace_depth is present (JSON has both kv and indentation)
    if f.key_value_ratio > 0.5 && f.leading_whitespace_ratio > 0.3 && f.json_brace_depth < 0.02 {
        return None;
    }

    // Guard: high key-value ratio with comments suggests config files (.env, INI, TOML) → Code
    // Config files have comments (# lines) that pure structured data doesn't
    if f.key_value_ratio > 0.5 && f.comment_ratio > 0.05 {
        return None;
    }

    // Guard: very high key-value ratio with symbols → config file with URLs/paths, not plain data
    // .env files have KEY=value with connection strings containing :// @ etc.
    if f.key_value_ratio > 0.7 && f.symbol_ratio > 0.08 && f.sentence_punctuation_rate < 0.01 {
        return None;
    }

    // Delimiter-consistent tabular data (CSV, TSV, pipe-delimited)
    if f.delimiter_consistency > 0.6 && f.line_count >= 3 {
        let confidence = 0.6 + 0.4 * f.delimiter_consistency;
        return Some(Classification {
            category: TextCategory::Structured,
            sub_type: None,
            confidence: confidence.min(1.0),
            reason: format!(
                "consistent delimiters (consistency={:.2}), {} lines",
                f.delimiter_consistency, f.line_count
            ),
            tier: Tier::Structural,
        });
    }

    // Guard: high symbol ratio with indentation is code with braces, not JSON data
    // (real JSON has symbol_ratio ~0.04, code with braces has ~0.07+)
    if f.symbol_ratio > 0.06 && f.leading_whitespace_ratio > 0.3 {
        return None;
    }

    // Guard: very high symbol ratio suggests minified code, not structured data
    if f.symbol_ratio > 0.12 {
        return None;
    }

    // JSON: brace/bracket content with symbols
    if f.json_brace_depth > 0.02 && f.symbol_ratio > 0.03 {
        let confidence = 0.6 + 0.4 * (f.json_brace_depth / 0.10).min(1.0);
        return Some(Classification {
            category: TextCategory::Structured,
            sub_type: None,
            confidence: confidence.min(1.0),
            reason: format!(
                "JSON structure (brace_depth={:.3}, sym={:.3})",
                f.json_brace_depth, f.symbol_ratio
            ),
            tier: Tier::Structural,
        });
    }

    // Key-value pairs without sentence structure (require multiple lines)
    if f.key_value_ratio > 0.5 && f.sentence_punctuation_rate < 0.02 && f.line_count >= 3 {
        let confidence = 0.6 + 0.3 * f.key_value_ratio;
        return Some(Classification {
            category: TextCategory::Structured,
            sub_type: None,
            confidence: confidence.min(1.0),
            reason: format!("key-value pattern (kv_ratio={:.2})", f.key_value_ratio),
            tier: Tier::Structural,
        });
    }

    // Log lines with timestamp patterns
    if f.log_line_ratio > 0.4 {
        let confidence = 0.6 + 0.4 * f.log_line_ratio;
        return Some(Classification {
            category: TextCategory::Structured,
            sub_type: None,
            confidence: confidence.min(1.0),
            reason: format!("log line pattern (log_ratio={:.2})", f.log_line_ratio),
            tier: Tier::Structural,
        });
    }

    None
}

fn try_code(f: &FeatureVector) -> Option<Classification> {
    // Path: XML/HTML markup
    if f.xml_tag_ratio > 0.3 {
        let confidence = (0.6 + 0.4 * f.xml_tag_ratio).min(1.0);
        return Some(Classification {
            category: TextCategory::Code,
            sub_type: None,
            confidence,
            reason: format!("markup tags (xml_ratio={:.2})", f.xml_tag_ratio),
            tier: Tier::Structural,
        });
    }

    // Path: Config files (YAML, TOML) — key-value with indentation
    if f.key_value_ratio > 0.5 && f.leading_whitespace_ratio > 0.3 {
        let confidence = (0.6 + 0.4 * f.leading_whitespace_ratio).min(1.0);
        return Some(Classification {
            category: TextCategory::Code,
            sub_type: None,
            confidence,
            reason: format!(
                "config/markup pattern (kv={:.2}, ws={:.2})",
                f.key_value_ratio, f.leading_whitespace_ratio
            ),
            tier: Tier::Structural,
        });
    }

    // Path: Flat config files (.env, INI without sections) — key-value with comments
    if f.key_value_ratio > 0.5 && f.comment_ratio > 0.05 && f.sentence_punctuation_rate < 0.02 {
        let confidence = (0.6 + 0.3 * f.key_value_ratio).min(1.0);
        return Some(Classification {
            category: TextCategory::Code,
            sub_type: None,
            confidence,
            reason: format!(
                "config pattern (kv={:.2}, comments={:.2})",
                f.key_value_ratio, f.comment_ratio
            ),
            tier: Tier::Structural,
        });
    }

    // Path: Very high key-value ratio with symbols → config file with URLs/paths
    if f.key_value_ratio > 0.7 && f.symbol_ratio > 0.08 && f.sentence_punctuation_rate < 0.01 {
        let confidence = (0.6 + 0.3 * f.key_value_ratio).min(1.0);
        return Some(Classification {
            category: TextCategory::Code,
            sub_type: None,
            confidence,
            reason: format!("config pattern (kv={:.2}, sym={:.2})", f.key_value_ratio, f.symbol_ratio),
            tier: Tier::Structural,
        });
    }

    // Path A: Indented code (Python, JS, Rust, HTML, etc.)
    if f.leading_whitespace_ratio > 0.3
        && f.symbol_ratio > 0.05
        && f.sentence_punctuation_rate < 0.02
    {
        let confidence = (0.6 + 0.4 * f.leading_whitespace_ratio).min(1.0);
        return Some(Classification {
            category: TextCategory::Code,
            sub_type: None,
            confidence,
            reason: format!(
                "indentation pattern (ws={:.2}), high symbols (sym={:.2})",
                f.leading_whitespace_ratio, f.symbol_ratio
            ),
            tier: Tier::Structural,
        });
    }

    // Path B: Config-like (heavy indentation, no prose signals)
    if f.leading_whitespace_ratio > 0.5 && f.sentence_punctuation_rate < 0.01 {
        let confidence = (0.6 + 0.4 * f.leading_whitespace_ratio).min(1.0);
        return Some(Classification {
            category: TextCategory::Code,
            sub_type: None,
            confidence,
            reason: format!(
                "config/markup pattern (ws={:.2}), no sentence structure",
                f.leading_whitespace_ratio
            ),
            tier: Tier::Structural,
        });
    }

    // Path C: Dense symbols (minified code)
    if f.symbol_ratio > 0.12 && f.sentence_punctuation_rate < 0.01 {
        let confidence = (0.6 + 0.3 * f.symbol_ratio / 0.25).min(1.0);
        return Some(Classification {
            category: TextCategory::Code,
            sub_type: None,
            confidence,
            reason: format!(
                "dense symbols (sym={:.2}), likely minified code",
                f.symbol_ratio
            ),
            tier: Tier::Structural,
        });
    }

    // Path D: Flat code with moderate symbols
    if f.symbol_ratio > 0.06
        && f.sentence_punctuation_rate < 0.01
        && f.leading_whitespace_ratio > 0.10
    {
        let confidence = (0.6 + 0.3 * f.symbol_ratio / 0.15).min(1.0);
        return Some(Classification {
            category: TextCategory::Code,
            sub_type: None,
            confidence,
            reason: format!(
                "code symbols (sym={:.2}), no sentence structure",
                f.symbol_ratio
            ),
            tier: Tier::Structural,
        });
    }

    None
}

fn try_artifact(f: &FeatureVector) -> Option<Classification> {
    // Artifact: very high short-line ratio with low alpha content (excludes config
    // files which are mostly alphabetic key-value pairs), or moderate short-line
    // ratio with low line uniqueness (repeated fragments typical of OCR dumps),
    // or very low line uniqueness on its own (boilerplate).
    if (f.short_line_ratio > 0.8 && f.alpha_ratio < 0.75)
        || (f.short_line_ratio > 0.5 && f.line_uniqueness < 0.5)
        || (f.line_uniqueness < 0.3 && f.line_count >= 5)
    {
        let confidence = if f.line_uniqueness < 0.3 {
            // Boilerplate/repetitive content — confidence based on how repetitive
            (0.7 + 0.3 * (1.0 - f.line_uniqueness)).min(1.0)
        } else {
            (0.6 + 0.4 * f.short_line_ratio).min(1.0)
        };
        Some(Classification {
            category: TextCategory::Artifact,
            sub_type: None,
            confidence,
            reason: format!(
                "artifact content (short_line={:.2}, uniqueness={:.2})",
                f.short_line_ratio, f.line_uniqueness
            ),
            tier: Tier::Structural,
        })
    } else {
        None
    }
}

fn try_prose(f: &FeatureVector) -> Option<Classification> {
    // Prose: sentence structure + alphanumeric content + variable line lengths
    if f.sentence_punctuation_rate > 0.03 && f.alpha_ratio > 0.70 && f.line_length_cv > 0.3 {
        let punct_score = (f.sentence_punctuation_rate / 0.08).min(1.0);
        let confidence = 0.6 + 0.4 * punct_score;
        Some(Classification {
            category: TextCategory::Prose,
            sub_type: None,
            confidence: confidence.min(1.0),
            reason: format!(
                "sentence structure (punct={:.3}), high alpha (ratio={:.2})",
                f.sentence_punctuation_rate, f.alpha_ratio
            ),
            tier: Tier::Structural,
        })
    } else {
        None
    }
}

/// Refine a detected category into a more specific content sub-type.
fn refine_sub_type(category: TextCategory, f: &FeatureVector) -> Option<ContentSubType> {
    match category {
        TextCategory::Structured => {
            if f.delimiter_consistency > 0.6 && f.tab_density > 0.03 {
                Some(ContentSubType::Tsv)
            } else if f.delimiter_consistency > 0.6 {
                Some(ContentSubType::Csv)
            } else if f.json_brace_depth > 0.02 {
                Some(ContentSubType::Json)
            } else if f.key_value_ratio > 0.5 {
                Some(ContentSubType::KeyValue)
            } else if f.log_line_ratio > 0.4 {
                Some(ContentSubType::LogLines)
            } else {
                None
            }
        }
        TextCategory::Code => {
            if f.xml_tag_ratio > 0.3 {
                Some(ContentSubType::Html)
            } else if f.key_value_ratio > 0.5 && f.leading_whitespace_ratio > 0.3 {
                Some(ContentSubType::Yaml)
            } else {
                None
            }
        }
        TextCategory::Artifact => {
            if f.line_uniqueness < 0.3 {
                Some(ContentSubType::Boilerplate)
            } else {
                Some(ContentSubType::PdfDump)
            }
        }
        _ => None,
    }
}

/// Fallback when no rule triggers at sufficient confidence.
/// Returns a low-confidence guess that signals Tier 2 should decide.
fn fallback_classification(f: &FeatureVector) -> Classification {
    if f.sentence_punctuation_rate > 0.02 && f.alpha_ratio > 0.55 {
        Classification {
            category: TextCategory::Prose,
            sub_type: None,
            confidence: 0.5,
            reason: "ambiguous — moderate sentence structure".to_string(),
            tier: Tier::Structural,
        }
    } else {
        Classification {
            category: TextCategory::Skip,
            sub_type: None,
            confidence: 0.5,
            reason: "ambiguous — insufficient prose signals".to_string(),
            tier: Tier::Structural,
        }
    }
}
