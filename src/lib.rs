pub mod features;
#[cfg(feature = "python")]
mod python;
pub mod tier1;
pub mod tier2;
pub mod types;

pub use features::extract_features;
pub use tier2::ModelClassifier;
pub use types::{
    Classification, ContentSubType, FeatureVector, TextCategory, TextType, Tier, thresholds,
};

use std::collections::BTreeMap;

/// Classify a text string using Tier 1 structural features only.
///
/// Convenience function for simple usage without a model.
/// For short texts (< 5 words), returns low-confidence Prose.
pub fn classify(text: &str) -> Classification {
    if text.trim().is_empty() {
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
    if tier1::is_too_short(text) {
        return Classification {
            category: TextCategory::Prose,
            sub_type: None,
            confidence: 0.5,
            reason: "too short for reliable classification".to_string(),
            tier: Tier::Structural,
            detections: BTreeMap::new(),
            sub_type_scores: BTreeMap::new(),
        };
    }

    let features = features::extract_features(text);
    tier1::classify_tier1(&features)
}

/// Main classifier combining Tier 1 (structural) and Tier 2 (model).
pub struct Classifier {
    model: ModelClassifier,
}

impl Classifier {
    /// Create a classifier using Tier 1 only (no model).
    pub fn new() -> Self {
        Self {
            model: ModelClassifier::without_model(),
        }
    }

    /// Create a classifier with the embedded ONNX model for Tier 2.
    ///
    /// When compiled with `--features onnx-model`, loads the model
    /// baked into the binary at compile time. Without the feature,
    /// this is equivalent to `new()`.
    pub fn with_embedded_model() -> Self {
        Self {
            model: ModelClassifier::new(),
        }
    }

    /// Create a classifier with a fasttext model for Tier 2.
    pub fn with_model(model_path: &str) -> Result<Self, String> {
        Ok(Self {
            model: ModelClassifier::with_model(model_path)?,
        })
    }

    /// Classify a single text string.
    ///
    /// When the ONNX model is loaded, all non-trivial samples go through
    /// the model (model-first). When no model is available, falls back
    /// to Tier 1 rule-based classification.
    pub fn classify(&self, text: &str) -> Classification {
        if text.trim().is_empty() {
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
        if tier1::is_too_short(text) {
            return Classification {
                category: TextCategory::Prose,
                sub_type: None,
                confidence: 0.5,
                reason: "too short for reliable classification".to_string(),
                tier: Tier::Structural,
                detections: BTreeMap::new(),
                sub_type_scores: BTreeMap::new(),
            };
        }

        let features = features::extract_features(text);

        if self.model.has_model() {
            return self.model.classify(&features);
        }

        // No model loaded — use Tier 1 rules
        tier1::classify_tier1(&features)
    }

    /// Classify multiple texts in parallel using rayon.
    pub fn classify_batch(&self, texts: &[&str]) -> Vec<Classification> {
        use rayon::prelude::*;
        texts.par_iter().map(|t| self.classify(t)).collect()
    }

    /// Extract raw structural features (for debugging/analysis).
    pub fn extract_features(&self, text: &str) -> FeatureVector {
        features::extract_features(text)
    }
}

impl Default for Classifier {
    fn default() -> Self {
        Self::new()
    }
}
