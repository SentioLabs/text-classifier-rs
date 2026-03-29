use crate::types::{Classification, FeatureVector, TextCategory, Tier};

#[cfg(feature = "model")]
use fasttext::FastText;

/// Maximum words to pass to the fasttext model.
#[cfg(feature = "model")]
const MAX_MODEL_WORDS: usize = 1000;

/// Tier 2 model-based classifier.
///
/// Wraps an optional fasttext model. When no model is loaded,
/// falls back to a simple heuristic on the structural features.
pub struct ModelClassifier {
    #[cfg(feature = "model")]
    model: Option<FastText>,

    #[cfg(not(feature = "model"))]
    _phantom: (),
}

impl ModelClassifier {
    /// Create a classifier without a model (fallback mode).
    pub fn without_model() -> Self {
        Self {
            #[cfg(feature = "model")]
            model: None,
            #[cfg(not(feature = "model"))]
            _phantom: (),
        }
    }

    /// Load a fasttext model from a .bin file.
    #[cfg(feature = "model")]
    pub fn with_model(model_path: &str) -> Result<Self, String> {
        let mut model = FastText::new();
        model
            .load_model(model_path)
            .map_err(|e| format!("Failed to load model: {e}"))?;
        Ok(Self { model: Some(model) })
    }

    #[cfg(not(feature = "model"))]
    pub fn with_model(_model_path: &str) -> Result<Self, String> {
        Err("text-classifier was compiled without 'model' feature. \
             Rebuild with: cargo build --features model"
            .to_string())
    }

    pub fn has_model(&self) -> bool {
        #[cfg(feature = "model")]
        {
            self.model.is_some()
        }
        #[cfg(not(feature = "model"))]
        {
            false
        }
    }

    /// Classify text using the model, or fall back to feature-based heuristic.
    pub fn classify(&self, _text: &str, features: &FeatureVector) -> Classification {
        #[cfg(feature = "model")]
        if let Some(model) = &self.model {
            return self.classify_with_model(model, _text);
        }

        // Fallback when no model is loaded
        self.classify_fallback(features)
    }

    #[cfg(feature = "model")]
    fn classify_with_model(&self, model: &FastText, text: &str) -> Classification {
        // Prepare input: first N words, single line
        let words: Vec<&str> = text.split_whitespace().take(MAX_MODEL_WORDS).collect();
        let input = words.join(" ");

        let predictions = match model.predict(&input, 1, 0.0) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Model prediction failed: {e}");
                return Classification {
                    category: TextCategory::Skip,
                    sub_type: None,
                    confidence: 0.3,
                    reason: "model prediction failed".to_string(),
                    tier: Tier::Model,
                };
            }
        };

        if let Some(prediction) = predictions.first() {
            let category = label_to_category(&prediction.label);
            Classification {
                category: category,
                sub_type: None,
                confidence: prediction.prob as f32,
                reason: format!("model prediction: {}", prediction.label),
                tier: Tier::Model,
            }
        } else {
            Classification {
                category: TextCategory::Skip,
                sub_type: None,
                confidence: 0.3,
                reason: "model returned no predictions".to_string(),
                tier: Tier::Model,
            }
        }
    }

    fn classify_fallback(&self, features: &FeatureVector) -> Classification {
        if features.sentence_punctuation_rate > 0.02 && features.alpha_ratio > 0.55 {
            Classification {
                category: TextCategory::Prose,
                sub_type: None,
                confidence: 0.5,
                reason: "no model — fallback: moderate prose signals".to_string(),
                tier: Tier::Structural,
            }
        } else {
            Classification {
                category: TextCategory::Skip,
                sub_type: None,
                confidence: 0.5,
                reason: "no model — fallback: insufficient prose signals".to_string(),
                tier: Tier::Structural,
            }
        }
    }
}

#[cfg(feature = "model")]
fn label_to_category(label: &str) -> TextCategory {
    // fasttext labels are prefixed with __label__
    let clean = label.strip_prefix("__label__").unwrap_or(label);
    match clean.to_lowercase().as_str() {
        "prose" => TextCategory::Prose,
        "code" => TextCategory::Code,
        "tabular" => TextCategory::Structured,
        "pdf_dump" | "pdfdump" => TextCategory::Artifact,
        _ => TextCategory::Skip,
    }
}
