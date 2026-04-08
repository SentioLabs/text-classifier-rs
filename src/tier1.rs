use crate::types::{Classification, FeatureVector, TextCategory, Tier};
use std::collections::BTreeMap;

/// Minimum confidence to accept a Tier 1 classification.
/// Set high so only unambiguous short-circuits are accepted;
/// everything else falls through to the model (Tier 2).
pub const MIN_CONFIDENCE: f32 = 0.95;

/// Classify text using Tier 1 structural short-circuits.
///
/// Only returns high-confidence (>= 0.95) results for unambiguous cases.
/// All other inputs get a low-confidence fallback (0.40) that signals
/// `Classifier::classify()` to hand off to Tier 2 (model-based classification).
pub fn classify_tier1(features: &FeatureVector) -> Classification {
    // Short-circuit: empty / no content
    if features.alpha_ratio == 0.0 && features.sentence_punctuation_rate == 0.0 {
        return Classification {
            category: TextCategory::Prose,
            sub_type: None,
            confidence: 0.0,
            reason: "empty or no content".to_string(),
            tier: Tier::Structural,
            detections: BTreeMap::new(),
            sub_type_scores: BTreeMap::new(),
        };
    }

    // Short-circuit 1: Pure CSV/TSV — very uniform line lengths with tabs
    if features.line_count >= 5
        && features.line_length_cv < 0.15
        && features.symbol_ratio < 0.10
        && (features.tab_density > 0.03 || features.sentence_punctuation_rate < 0.01)
    {
        return Classification {
            category: TextCategory::Structured,
            sub_type: None,
            confidence: 0.98,
            reason: format!(
                "uniform line lengths (cv={:.2}), tabular structure",
                features.line_length_cv
            ),
            tier: Tier::Structural,
            detections: BTreeMap::new(),
            sub_type_scores: BTreeMap::new(),
        };
    }

    // Short-circuit 2: HTML/XML — heavy indentation + high symbol ratio (angle brackets etc.)
    if features.leading_whitespace_ratio > 0.3
        && features.symbol_ratio > 0.08
        && features.sentence_punctuation_rate < 0.02
    {
        return Classification {
            category: TextCategory::Code,
            sub_type: None,
            confidence: 0.98,
            reason: format!(
                "markup pattern (ws={:.2}, sym={:.2})",
                features.leading_whitespace_ratio, features.symbol_ratio
            ),
            tier: Tier::Structural,
            detections: BTreeMap::new(),
            sub_type_scores: BTreeMap::new(),
        };
    }

    // Short-circuit 3: Pure prose — strong sentence punctuation, high alpha, low symbols/indentation
    if features.sentence_punctuation_rate > 0.06
        && features.alpha_ratio > 0.85
        && features.symbol_ratio < 0.03
        && features.leading_whitespace_ratio < 0.1
    {
        return Classification {
            category: TextCategory::Prose,
            sub_type: None,
            confidence: 0.95,
            reason: format!(
                "sentence structure (punct={:.3}), high alpha (ratio={:.2})",
                features.sentence_punctuation_rate, features.alpha_ratio
            ),
            tier: Tier::Structural,
            detections: BTreeMap::new(),
            sub_type_scores: BTreeMap::new(),
        };
    }

    // Default fallback: low confidence to trigger Tier 2
    Classification {
        category: TextCategory::Prose,
        sub_type: None,
        confidence: 0.40,
        reason: "ambiguous — deferring to model".to_string(),
        tier: Tier::Structural,
        detections: BTreeMap::new(),
        sub_type_scores: BTreeMap::new(),
    }
}

/// Check if text is too short for reliable feature extraction.
/// Requires both few whitespace-delimited tokens AND few characters.
/// The character check prevents CJK/Thai text (which lacks spaces between
/// words) from being wrongly skipped.
pub fn is_too_short(text: &str) -> bool {
    text.split_whitespace().count() < 5 && text.chars().count() < 20
}
