pub mod features;
#[cfg(feature = "python")]
mod python;
pub mod tier1;
pub mod tier2;
pub mod types;

pub use features::extract_features;
pub use tier2::ModelClassifier;
pub use types::thresholds;
pub use types::{Classification, ContentSubType, FeatureVector, TextCategory, TextType, Tier};

/// Classify a text string using Tier 1 structural features only.
///
/// Convenience function for simple usage without a model.
/// For short texts (< 5 words), returns Skip immediately.
pub fn classify(text: &str) -> Classification {
    if text.trim().is_empty() || tier1::is_too_short(text) {
        return Classification {
            category: TextType::Skip,
            sub_type: None,
            confidence: 1.0,
            reason: "too short".to_string(),
            tier: Tier::Structural,
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
    /// Create a classifier, auto-loading the embedded ONNX model if available.
    pub fn new() -> Self {
        Self {
            model: ModelClassifier::new(),
        }
    }

    /// Classify a single text string.
    ///
    /// 1. Short text (< 5 words) -> Skip
    /// 2. Tier 1 structural features -> if confidence >= per-type threshold, return
    /// 3. Tier 2 model (if loaded) -> return model decision
    /// 4. No model -> return low-confidence fallback
    pub fn classify(&self, text: &str) -> Classification {
        if text.trim().is_empty() || tier1::is_too_short(text) {
            return Classification {
                category: TextType::Skip,
                sub_type: None,
                confidence: 1.0,
                reason: "too short".to_string(),
                tier: Tier::Structural,
            };
        }

        let features = features::extract_features(text);
        let tier1_result = tier1::classify_tier1(&features);

        // If Tier 1 is confident (per-type threshold), use it
        let threshold = match tier1_result.category {
            TextCategory::Prose => thresholds::PROSE,
            TextCategory::Code => thresholds::CODE,
            TextCategory::Structured => thresholds::STRUCTURED,
            TextCategory::Artifact => thresholds::ARTIFACT,
            TextCategory::Tabular => thresholds::STRUCTURED,
            TextCategory::PdfDump => thresholds::ARTIFACT,
            TextCategory::Skip => 0.0, // Skip always accepted
        };
        if tier1_result.confidence >= threshold {
            return tier1_result;
        }

        // Otherwise, fall through to Tier 2
        self.model.classify(&features)
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
