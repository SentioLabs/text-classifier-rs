use crate::types::{Classification, FeatureVector, TextType, Tier};

/// Minimum confidence to accept a Tier 1 classification.
const MIN_CONFIDENCE: f32 = 0.7;

/// Classify text using Tier 1 structural features.
///
/// Returns a Classification. If confidence < 0.7, the caller should
/// fall through to Tier 2 (model-based classification).
pub fn classify_tier1(features: &FeatureVector) -> Classification {
    // Rule 1: Too short — fewer than ~5 words means all features are unreliable
    // (We detect this via zero/near-zero sentence punctuation + very low alpha content)
    if features.alpha_ratio == 0.0 && features.sentence_punctuation_rate == 0.0 {
        return Classification {
            text_type: TextType::Skip,
            confidence: 1.0,
            reason: "empty or no content".to_string(),
            tier: Tier::Structural,
        };
    }

    // Try each rule in priority order, collect the best candidate
    let candidates = [
        try_tabular(features),
        try_code(features),
        try_pdf_dump(features),
        try_translatable(features),
    ];

    // Return the first candidate with confidence >= threshold
    for c in candidates.iter().flatten() {
        if c.confidence >= MIN_CONFIDENCE {
            return c.clone();
        }
    }

    // No rule triggered at high confidence — ambiguous
    // Return best guess at low confidence for Tier 2 fallback
    fallback_classification(features)
}

/// Check if the caller passed in features for very short text.
/// This is called by the public `classify()` function before `classify_tier1()`.
pub fn is_too_short(text: &str) -> bool {
    text.split_whitespace().count() < 5
}

fn try_tabular(f: &FeatureVector) -> Option<Classification> {
    // Tabular: uniform line lengths AND (tabs or no sentence structure)
    // Exclude high-symbol content (e.g. minified code has cv=0 but symbol_ratio > 0.10)
    if f.line_length_cv < 0.15
        && f.symbol_ratio < 0.10
        && (f.tab_density > 0.03 || f.sentence_punctuation_rate < 0.01)
    {
        let confidence = 0.7 + 0.3 * (1.0 - f.line_length_cv / 0.15);
        Some(Classification {
            text_type: TextType::Tabular,
            confidence: confidence.min(1.0),
            reason: format!(
                "uniform line lengths (cv={:.2}), low sentence structure",
                f.line_length_cv
            ),
            tier: Tier::Structural,
        })
    } else {
        None
    }
}

fn try_code(f: &FeatureVector) -> Option<Classification> {
    // Path A: Indented code (Python, JS, Rust, HTML, etc.)
    let indented = f.leading_whitespace_ratio > 0.3
        && f.symbol_ratio > 0.05
        && f.sentence_punctuation_rate < 0.02;

    // Path B: Flat code with moderate symbols (SQL, Go, etc.)
    // Requires some indentation (> 0.10) to avoid matching pipe tables (ws=0)
    let flat = f.symbol_ratio > 0.06
        && f.sentence_punctuation_rate < 0.01
        && f.leading_whitespace_ratio > 0.10;

    // Path C: Config-like (YAML, TOML) — heavy indentation, no prose signals
    // Structural chars like : and - are excluded from symbol_ratio, so we
    // rely on indentation alone when it's very strong
    let config_like = f.leading_whitespace_ratio > 0.5 && f.sentence_punctuation_rate < 0.01;

    // Path D: Dense symbols (minified code) — single long line packed with symbols
    let dense = f.symbol_ratio > 0.12 && f.sentence_punctuation_rate < 0.01;

    if indented {
        let confidence = 0.6 + 0.4 * f.leading_whitespace_ratio;
        Some(Classification {
            text_type: TextType::Code,
            confidence: confidence.min(1.0),
            reason: format!(
                "indentation pattern (ws={:.2}), high symbols (sym={:.2})",
                f.leading_whitespace_ratio, f.symbol_ratio
            ),
            tier: Tier::Structural,
        })
    } else if config_like {
        let confidence = 0.6 + 0.4 * f.leading_whitespace_ratio;
        Some(Classification {
            text_type: TextType::Code,
            confidence: confidence.min(1.0),
            reason: format!(
                "config/markup pattern (ws={:.2}), no sentence structure",
                f.leading_whitespace_ratio
            ),
            tier: Tier::Structural,
        })
    } else if dense {
        let confidence = (0.6 + 0.3 * f.symbol_ratio / 0.25).min(1.0);
        Some(Classification {
            text_type: TextType::Code,
            confidence,
            reason: format!(
                "dense symbols (sym={:.2}), likely minified code",
                f.symbol_ratio
            ),
            tier: Tier::Structural,
        })
    } else if flat {
        let confidence = (0.6 + 0.3 * f.symbol_ratio / 0.15).min(1.0);
        Some(Classification {
            text_type: TextType::Code,
            confidence,
            reason: format!(
                "code symbols (sym={:.2}), no sentence structure",
                f.symbol_ratio
            ),
            tier: Tier::Structural,
        })
    } else {
        None
    }
}

fn try_pdf_dump(f: &FeatureVector) -> Option<Classification> {
    // PdfDump: very high short-line ratio alone (>0.8), or moderate short-line ratio
    // with either high symbols or low line uniqueness. OCR char-by-char dumps have
    // low entropy (not high), so we don't use entropy as a signal.
    if f.short_line_ratio > 0.8 || (f.short_line_ratio > 0.5 && f.line_uniqueness < 0.5) {
        let confidence = 0.6 + 0.4 * f.short_line_ratio;
        Some(Classification {
            text_type: TextType::PdfDump,
            confidence: confidence.min(1.0),
            reason: format!(
                "short lines (ratio={:.2}), garbled content",
                f.short_line_ratio
            ),
            tier: Tier::Structural,
        })
    } else {
        None
    }
}

fn try_translatable(f: &FeatureVector) -> Option<Classification> {
    // Translatable: sentence structure + alphanumeric content + variable line lengths
    if f.sentence_punctuation_rate > 0.03 && f.alpha_ratio > 0.70 && f.line_length_cv > 0.3 {
        // Scale confidence by how strong the sentence signal is (cap at 0.08)
        let punct_score = (f.sentence_punctuation_rate / 0.08).min(1.0);
        let confidence = 0.6 + 0.4 * punct_score;
        Some(Classification {
            text_type: TextType::Translatable,
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

/// Fallback when no rule triggers at >= 0.7 confidence.
/// Returns a low-confidence guess that signals Tier 2 should decide.
fn fallback_classification(f: &FeatureVector) -> Classification {
    // Lean toward translatable if there's some sentence structure
    if f.sentence_punctuation_rate > 0.02 && f.alpha_ratio > 0.55 {
        Classification {
            text_type: TextType::Translatable,
            confidence: 0.5,
            reason: "ambiguous — moderate sentence structure".to_string(),
            tier: Tier::Structural,
        }
    } else {
        Classification {
            text_type: TextType::Skip,
            confidence: 0.5,
            reason: "ambiguous — insufficient prose signals".to_string(),
            tier: Tier::Structural,
        }
    }
}
