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
    min_score: Option<f32>,
}

impl Classifier {
    /// Create a classifier using Tier 1 only (no model).
    pub fn new() -> Self {
        Self {
            model: ModelClassifier::without_model(),
            min_score: None,
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
            min_score: None,
        }
    }

    /// Create a classifier with a fasttext model for Tier 2.
    pub fn with_model(model_path: &str) -> Result<Self, String> {
        Ok(Self {
            model: ModelClassifier::with_model(model_path)?,
            min_score: None,
        })
    }

    /// Set a minimum score threshold for `sub_type_scores` and `detections`.
    ///
    /// Scores below this threshold are dropped from the output.
    /// Default is `None` (no filtering — all scores included).
    pub fn min_score(mut self, threshold: f32) -> Self {
        self.min_score = Some(threshold);
        self
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

        let mut result = if self.model.has_model() {
            self.model.classify(&features)
        } else {
            tier1::classify_tier1(&features)
        };

        if let Some(threshold) = self.min_score {
            filter_scores(&mut result.sub_type_scores, threshold);
            filter_scores(&mut result.detections, threshold);
        }

        result
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

fn filter_scores(scores: &mut BTreeMap<TextCategory, Vec<types::Detection>>, threshold: f32) {
    scores.retain(|_, detections| {
        detections.retain(|d| d.score >= threshold);
        !detections.is_empty()
    });
}

impl Default for Classifier {
    fn default() -> Self {
        Self::new()
    }
}
