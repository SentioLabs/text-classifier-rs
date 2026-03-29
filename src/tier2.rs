#[cfg(any(feature = "onnx-model", test))]
use crate::types::ContentSubType;
use crate::types::{Classification, FeatureVector, TextCategory, Tier};

#[cfg(feature = "onnx-model")]
use std::collections::HashMap;
#[cfg(feature = "onnx-model")]
use std::sync::Mutex;

#[cfg(feature = "onnx-model")]
static MODEL_BYTES: &[u8] = include_bytes!("model.onnx");
#[cfg(feature = "onnx-model")]
static CONFIG_JSON: &str = include_str!("model_config.json");

#[cfg(feature = "onnx-model")]
#[derive(serde::Deserialize)]
struct ModelConfig {
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    category_map: HashMap<String, usize>,
    sub_type_map: HashMap<String, usize>,
}

/// Tier 2 model-based classifier.
///
/// When the `onnx-model` feature is enabled, loads an embedded ONNX model
/// and runs inference on structural features. Falls back to a simple
/// heuristic when no model is available.
pub struct ModelClassifier {
    #[cfg(feature = "onnx-model")]
    session: Option<Mutex<ort::session::Session>>,
    #[cfg(feature = "onnx-model")]
    config: Option<ModelConfig>,
    #[cfg(feature = "onnx-model")]
    inv_category_map: Option<HashMap<usize, String>>,
    #[cfg(feature = "onnx-model")]
    inv_sub_type_map: Option<HashMap<usize, String>>,

    #[cfg(not(feature = "onnx-model"))]
    _phantom: (),
}

impl Default for ModelClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelClassifier {
    /// Create a classifier, loading the embedded ONNX model if available.
    #[cfg(feature = "onnx-model")]
    pub fn new() -> Self {
        match Self::try_load() {
            Ok(classifier) => classifier,
            Err(e) => {
                eprintln!("Warning: failed to load ONNX model: {e}");
                Self {
                    session: None,
                    config: None,
                    inv_category_map: None,
                    inv_sub_type_map: None,
                }
            }
        }
    }

    #[cfg(not(feature = "onnx-model"))]
    pub fn new() -> Self {
        Self { _phantom: () }
    }

    #[cfg(feature = "onnx-model")]
    fn try_load() -> Result<Self, Box<dyn std::error::Error>> {
        let config: ModelConfig = serde_json::from_str(CONFIG_JSON)?;

        let inv_category_map: HashMap<usize, String> = config
            .category_map
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        let inv_sub_type_map: HashMap<usize, String> = config
            .sub_type_map
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        let session = ort::session::Session::builder()?.commit_from_memory(MODEL_BYTES)?;

        Ok(Self {
            session: Some(Mutex::new(session)),
            config: Some(config),
            inv_category_map: Some(inv_category_map),
            inv_sub_type_map: Some(inv_sub_type_map),
        })
    }

    /// Create a classifier without a model (fallback mode).
    pub fn without_model() -> Self {
        #[cfg(feature = "onnx-model")]
        {
            Self {
                session: None,
                config: None,
                inv_category_map: None,
                inv_sub_type_map: None,
            }
        }
        #[cfg(not(feature = "onnx-model"))]
        {
            Self { _phantom: () }
        }
    }

    /// Backward-compatible constructor that accepts a model path.
    ///
    /// When the `onnx-model` feature is enabled, ignores the path and loads
    /// the embedded model instead. Without the feature, returns an error.
    #[cfg(feature = "onnx-model")]
    pub fn with_model(_model_path: &str) -> Result<Self, String> {
        Ok(Self::new())
    }

    #[cfg(not(feature = "onnx-model"))]
    pub fn with_model(_model_path: &str) -> Result<Self, String> {
        Err(
            "text-classifier was compiled without 'onnx-model' feature. \
             Rebuild with: cargo build --features onnx-model"
                .to_string(),
        )
    }

    /// Returns true if an ONNX model session is loaded.
    pub fn has_model(&self) -> bool {
        #[cfg(feature = "onnx-model")]
        {
            self.session.is_some()
        }
        #[cfg(not(feature = "onnx-model"))]
        {
            false
        }
    }

    /// Classify using the model, or fall back to feature-based heuristic.
    pub fn classify(&self, features: &FeatureVector) -> Classification {
        #[cfg(feature = "onnx-model")]
        if self.session.is_some() {
            match self.classify_onnx(features) {
                Ok(result) => return result,
                Err(e) => {
                    eprintln!("ONNX inference failed: {e}");
                }
            }
        }

        self.classify_fallback(features)
    }

    #[cfg(feature = "onnx-model")]
    fn classify_onnx(
        &self,
        features: &FeatureVector,
    ) -> Result<Classification, Box<dyn std::error::Error>> {
        let config = self.config.as_ref().unwrap();
        let inv_cat = self.inv_category_map.as_ref().unwrap();
        let inv_sub = self.inv_sub_type_map.as_ref().unwrap();

        let raw = feature_vector_to_array(features);

        // Z-score standardize
        let standardized: Vec<f32> = raw
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let std_dev = config.feature_std[i];
                if std_dev == 0.0 {
                    0.0
                } else {
                    (val - config.feature_mean[i]) / std_dev
                }
            })
            .collect();

        let input =
            ort::value::Tensor::from_array((vec![1i64, 18], standardized.into_boxed_slice()))?;
        let mut session = self.session.as_ref().unwrap().lock().map_err(|e| {
            Box::<dyn std::error::Error>::from(format!("session lock poisoned: {e}"))
        })?;
        let outputs = session.run(ort::inputs!["features" => input])?;

        let (_cat_shape, cat_logits) = outputs[0].try_extract_tensor::<f32>()?;
        let (_sub_shape, sub_logits) = outputs[1].try_extract_tensor::<f32>()?;

        let cat_probs = softmax(cat_logits);
        let sub_probs = softmax(sub_logits);

        let (cat_idx, cat_conf) = argmax(&cat_probs);
        let (sub_idx, _sub_conf) = argmax(&sub_probs);

        let cat_label = inv_cat.get(&cat_idx).map(|s| s.as_str()).unwrap_or("skip");
        let sub_label = inv_sub
            .get(&sub_idx)
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        let category = parse_category(cat_label);
        let sub_type = parse_sub_type(sub_label);

        Ok(Classification {
            category,
            sub_type: Some(sub_type),
            confidence: cat_conf,
            reason: format!("onnx model: category={cat_label}, sub_type={sub_label}"),
            tier: Tier::Model,
        })
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

/// Extract 18 f32 features from a FeatureVector in model-expected order.
#[cfg(any(feature = "onnx-model", test))]
fn feature_vector_to_array(f: &FeatureVector) -> [f32; 18] {
    [
        f.line_length_cv,
        f.char_entropy,
        f.leading_whitespace_ratio,
        f.tab_density,
        f.sentence_punctuation_rate,
        f.paragraph_break_rate,
        f.alpha_ratio,
        f.line_uniqueness,
        f.short_line_ratio,
        f.symbol_ratio,
        f.delimiter_consistency,
        f.json_brace_depth,
        f.key_value_ratio,
        f.xml_tag_ratio,
        f.log_line_ratio,
        f.comment_ratio,
        f.numeric_field_ratio,
        f.repetitive_structure_score,
    ]
}

#[cfg(any(feature = "onnx-model", test))]
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(any(feature = "onnx-model", test))]
fn argmax(values: &[f32]) -> (usize, f32) {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0))
}

#[cfg(any(feature = "onnx-model", test))]
fn parse_category(s: &str) -> TextCategory {
    match s {
        "prose" => TextCategory::Prose,
        "code" => TextCategory::Code,
        "structured" => TextCategory::Structured,
        "artifact" => TextCategory::Artifact,
        _ => TextCategory::Skip,
    }
}

#[cfg(any(feature = "onnx-model", test))]
fn parse_sub_type(s: &str) -> ContentSubType {
    match s {
        "plain" => ContentSubType::Plain,
        "markdown" => ContentSubType::Markdown,
        "rst" => ContentSubType::Rst,
        "latex" => ContentSubType::Latex,
        "python" => ContentSubType::Python,
        "javascript" | "minified_js" => ContentSubType::JavaScript,
        "typescript" => ContentSubType::TypeScript,
        "rust" => ContentSubType::Rust,
        "go" => ContentSubType::Go,
        "java" => ContentSubType::Java,
        "sql" => ContentSubType::Sql,
        "shell" => ContentSubType::Shell,
        "css" => ContentSubType::Css,
        "yaml" | "yaml_config" | "yaml_nested" => ContentSubType::Yaml,
        "toml" | "toml_config" => ContentSubType::Toml,
        "ini" | "ini_config" => ContentSubType::Ini,
        "dockerfile" => ContentSubType::Dockerfile,
        "makefile" => ContentSubType::Makefile,
        "html" => ContentSubType::Html,
        "xml" | "sgml" => ContentSubType::Xml,
        "csv" | "csv_data" => ContentSubType::Csv,
        "tsv" | "tsv_data" => ContentSubType::Tsv,
        "pipe_table" => ContentSubType::PipeTable,
        "fixed_width" => ContentSubType::FixedWidth,
        "json" | "json_object" | "simple" => ContentSubType::Json,
        "jsonl" | "jsonl_data" => ContentSubType::Jsonl,
        "key_value" => ContentSubType::KeyValue,
        "log_lines" => ContentSubType::LogLines,
        "pdf_dump" => ContentSubType::PdfDump,
        "ocr_garbage" => ContentSubType::OcrGarbage,
        "boilerplate" => ContentSubType::Boilerplate,
        "too_short" => ContentSubType::TooShort,
        "empty" => ContentSubType::Empty,
        "ambiguous" => ContentSubType::Ambiguous,
        _ => ContentSubType::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FeatureVector;

    fn prose_features() -> FeatureVector {
        let mut f = FeatureVector::zeroed();
        f.sentence_punctuation_rate = 0.05;
        f.alpha_ratio = 0.85;
        f
    }

    fn skip_features() -> FeatureVector {
        let mut f = FeatureVector::zeroed();
        f.sentence_punctuation_rate = 0.0;
        f.alpha_ratio = 0.1;
        f
    }

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1.0");
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_single() {
        let probs = softmax(&[5.0]);
        assert!((probs[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_equal() {
        let probs = softmax(&[1.0, 1.0, 1.0]);
        for p in &probs {
            assert!((*p - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_large_values() {
        // Should not overflow thanks to max subtraction
        let probs = softmax(&[1000.0, 1001.0, 1002.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3]).0, 1);
        assert_eq!(argmax(&[0.9, 0.05, 0.05]).0, 0);
    }

    #[test]
    fn test_parse_category() {
        assert_eq!(parse_category("prose"), TextCategory::Prose);
        assert_eq!(parse_category("code"), TextCategory::Code);
        assert_eq!(parse_category("structured"), TextCategory::Structured);
        assert_eq!(parse_category("artifact"), TextCategory::Artifact);
        assert_eq!(parse_category("skip"), TextCategory::Skip);
        assert_eq!(parse_category("unknown_value"), TextCategory::Skip);
    }

    #[test]
    fn test_parse_sub_type_known_labels() {
        assert_eq!(parse_sub_type("python"), ContentSubType::Python);
        assert_eq!(parse_sub_type("markdown"), ContentSubType::Markdown);
        assert_eq!(parse_sub_type("csv"), ContentSubType::Csv);
        assert_eq!(parse_sub_type("csv_data"), ContentSubType::Csv);
        assert_eq!(parse_sub_type("yaml_config"), ContentSubType::Yaml);
        assert_eq!(parse_sub_type("json_object"), ContentSubType::Json);
        assert_eq!(parse_sub_type("minified_js"), ContentSubType::JavaScript);
        assert_eq!(parse_sub_type("boilerplate"), ContentSubType::Boilerplate);
    }

    #[test]
    fn test_parse_sub_type_unknown() {
        assert_eq!(parse_sub_type("nonexistent"), ContentSubType::Unknown);
    }

    #[test]
    fn test_feature_vector_to_array_order() {
        let mut f = FeatureVector::zeroed();
        f.line_length_cv = 1.0;
        f.char_entropy = 2.0;
        f.repetitive_structure_score = 18.0;
        let arr = feature_vector_to_array(&f);
        assert_eq!(arr[0], 1.0);
        assert_eq!(arr[1], 2.0);
        assert_eq!(arr[17], 18.0);
        assert_eq!(arr.len(), 18);
    }

    #[test]
    fn test_fallback_prose() {
        let classifier = ModelClassifier::without_model();
        let features = prose_features();
        let result = classifier.classify(&features);
        assert_eq!(result.category, TextCategory::Prose);
        assert!((result.confidence - 0.5).abs() < f32::EPSILON);
        assert_eq!(result.tier, Tier::Structural);
    }

    #[test]
    fn test_fallback_skip() {
        let classifier = ModelClassifier::without_model();
        let features = skip_features();
        let result = classifier.classify(&features);
        assert_eq!(result.category, TextCategory::Skip);
        assert!((result.confidence - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_without_model_has_no_model() {
        let classifier = ModelClassifier::without_model();
        assert!(!classifier.has_model());
    }

    #[cfg(feature = "onnx-model")]
    #[test]
    fn test_new_loads_model() {
        let classifier = ModelClassifier::new();
        assert!(classifier.has_model());
    }

    #[cfg(feature = "onnx-model")]
    #[test]
    fn test_onnx_inference_returns_valid_result() {
        let classifier = ModelClassifier::new();
        assert!(classifier.has_model());

        let features = prose_features();
        let result = classifier.classify(&features);

        // Should return a model-tier classification
        assert_eq!(result.tier, Tier::Model);
        assert!(result.confidence > 0.0);
        assert!(result.confidence <= 1.0);
        assert!(result.sub_type.is_some());
        assert!(result.reason.starts_with("onnx model:"));
    }

    #[cfg(feature = "onnx-model")]
    #[test]
    fn test_onnx_with_model_backward_compat() {
        let classifier = ModelClassifier::with_model("ignored_path").unwrap();
        assert!(classifier.has_model());
    }

    #[cfg(not(feature = "onnx-model"))]
    #[test]
    fn test_new_without_feature_has_no_model() {
        let classifier = ModelClassifier::new();
        assert!(!classifier.has_model());
    }
}
