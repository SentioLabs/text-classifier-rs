#[cfg(any(feature = "onnx-model", test))]
use crate::types::ContentSubType;
#[cfg(any(feature = "onnx-model", test))]
use crate::types::Detection;
use crate::types::{
    Classification, DEFAULT_DETECTION_THRESHOLD, FeatureVector, TextCategory, Tier,
};
use std::collections::BTreeMap;

#[cfg(any(feature = "onnx-model", test))]
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
    detection_map: Option<HashMap<String, usize>>,
    detection_threshold: Option<f32>,
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
    #[cfg(feature = "onnx-model")]
    inv_detection_map: Option<HashMap<usize, String>>,
    #[cfg(feature = "onnx-model")]
    detection_threshold: f32,

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
                    inv_detection_map: None,
                    detection_threshold: DEFAULT_DETECTION_THRESHOLD,
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

        let inv_detection_map = config
            .detection_map
            .as_ref()
            .map(|dm| dm.iter().map(|(k, v)| (*v, k.clone())).collect());

        let session = ort::session::Session::builder()?.commit_from_memory(MODEL_BYTES)?;

        let detection_threshold = config
            .detection_threshold
            .unwrap_or(DEFAULT_DETECTION_THRESHOLD);

        Ok(Self {
            session: Some(Mutex::new(session)),
            config: Some(config),
            inv_category_map: Some(inv_category_map),
            inv_sub_type_map: Some(inv_sub_type_map),
            inv_detection_map,
            detection_threshold,
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
                inv_detection_map: None,
                detection_threshold: DEFAULT_DETECTION_THRESHOLD,
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

        let input = ort::value::Tensor::from_array((
            vec![1i64, NUM_FEATURES as i64],
            standardized.into_boxed_slice(),
        ))?;
        let mut session = self.session.as_ref().unwrap().lock().map_err(|e| {
            Box::<dyn std::error::Error>::from(format!("session lock poisoned: {e}"))
        })?;
        let outputs = session.run(ort::inputs!["features" => input])?;

        let (_cat_shape, cat_logits) = outputs[0].try_extract_tensor::<f32>()?;
        let (_sub_shape, sub_logits) = outputs[1].try_extract_tensor::<f32>()?;

        let cat_probs = softmax(cat_logits);
        let sub_probs = softmax(sub_logits);
        let det_logits = if outputs.len() >= 3 {
            Some(outputs[2].try_extract_tensor::<f32>()?.1)
        } else {
            None
        };

        Ok(build_classification(
            &cat_probs,
            &sub_probs,
            det_logits,
            inv_cat,
            inv_sub,
            self.inv_detection_map.as_ref(),
            self.detection_threshold,
        ))
    }

    fn classify_fallback(&self, features: &FeatureVector) -> Classification {
        if features.sentence_punctuation_rate > 0.02 && features.alpha_ratio > 0.55 {
            Classification {
                category: TextCategory::Prose,
                sub_type: None,
                confidence: 0.5,
                reason: "no model — fallback: moderate prose signals".to_string(),
                tier: Tier::Structural,
                detections: BTreeMap::new(),
            }
        } else {
            Classification {
                category: TextCategory::Skip,
                sub_type: None,
                confidence: 0.5,
                reason: "no model — fallback: insufficient prose signals".to_string(),
                tier: Tier::Structural,
                detections: BTreeMap::new(),
            }
        }
    }
}

/// Number of features in the feature vector (must match model input size).
#[cfg(any(feature = "onnx-model", test))]
const NUM_FEATURES: usize = 40;

/// Extract features from a FeatureVector in model-expected order.
#[cfg(any(feature = "onnx-model", test))]
fn feature_vector_to_array(f: &FeatureVector) -> [f32; NUM_FEATURES] {
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
        f.hyphenated_line_break_ratio,
        f.short_repeated_line_ratio,
        f.page_number_density,
        f.label_value_line_ratio,
        f.table_fragment_score,
        f.uppercase_header_ratio,
        f.dictionary_word_ratio,
        f.encoding_error_ratio,
        f.repeated_ngram_ratio,
        f.sentence_coherence_score,
        // New features (v2)
        f.avg_words_per_line,
        f.operator_density,
        f.inline_markup_count,
        f.indentation_consistency,
        f.markup_heading_ratio,
        f.code_fence_density,
        f.prose_paragraph_ratio,
        f.semicolon_line_ending_ratio,
        f.list_item_ratio,
        f.parenthesis_density,
        f.section_header_ratio,
        f.json_lines_ratio,
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

#[cfg(test)]
fn parse_category(s: &str) -> TextCategory {
    match s {
        "prose" => TextCategory::Prose,
        "code" => TextCategory::Code,
        "structured" => TextCategory::Structured,
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
        _ => ContentSubType::Unknown,
    }
}

#[cfg(any(feature = "onnx-model", test))]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(any(feature = "onnx-model", test))]
fn sigmoid_vec(logits: &[f32]) -> Vec<f32> {
    logits.iter().map(|&x| sigmoid(x)).collect()
}

#[cfg(any(feature = "onnx-model", test))]
fn build_detections(
    scores: &[f32],
    inv_map: &HashMap<usize, String>,
    threshold: f32,
) -> BTreeMap<TextCategory, Vec<Detection>> {
    let mut result = BTreeMap::new();
    for (idx, &score) in scores.iter().enumerate() {
        if score < threshold {
            continue;
        }
        let label = inv_map.get(&idx).map(|s| s.as_str()).unwrap_or("unknown");
        let sub_type = parse_sub_type(label);
        let category = sub_type.category();
        result
            .entry(category)
            .or_insert_with(Vec::new)
            .push(Detection { sub_type, score });
    }
    for detections in result.values_mut() {
        detections.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    result
}

#[cfg(any(feature = "onnx-model", test))]
fn build_classification(
    _cat_probs: &[f32],
    sub_probs: &[f32],
    det_logits: Option<&[f32]>,
    _inv_cat: &HashMap<usize, String>,
    inv_sub: &HashMap<usize, String>,
    inv_det: Option<&HashMap<usize, String>>,
    detection_threshold: f32,
) -> Classification {
    let (sub_idx, _sub_conf) = argmax(sub_probs);

    let sub_label = inv_sub
        .get(&sub_idx)
        .map(|s| s.as_str())
        .unwrap_or("unknown");

    // Marginalize category from sub-type probabilities: sum sub_type probs
    // grouped by their canonical category.
    let mut cat_accum: HashMap<TextCategory, f32> = HashMap::new();
    for (idx, &prob) in sub_probs.iter().enumerate() {
        let label = inv_sub.get(&idx).map(|s| s.as_str()).unwrap_or("unknown");
        let sub_type = parse_sub_type(label);
        let category = sub_type.category();
        *cat_accum.entry(category).or_insert(0.0) += prob;
    }

    let (marginalized_cat, marginalized_conf) = cat_accum
        .into_iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((TextCategory::Skip, 0.0));

    let detections = match (det_logits, inv_det) {
        (Some(logits), Some(inv_det)) => {
            let det_scores = sigmoid_vec(logits);
            build_detections(&det_scores, inv_det, detection_threshold)
        }
        _ => BTreeMap::new(),
    };

    Classification {
        category: marginalized_cat,
        sub_type: Some(parse_sub_type(sub_label)),
        confidence: marginalized_conf,
        reason: format!(
            "onnx model (marginalized): category={marginalized_cat}, sub_type={sub_label}"
        ),
        tier: Tier::Model,
        detections,
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
        assert_eq!(parse_category("skip"), TextCategory::Skip);
        assert_eq!(parse_category("artifact"), TextCategory::Skip); // removed category falls to default
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
        assert_eq!(parse_sub_type("boilerplate"), ContentSubType::Unknown); // removed sub-type falls to default
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
        assert_eq!(arr.len(), NUM_FEATURES);
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
        assert!(result.reason.starts_with("onnx model"));
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

    #[test]
    fn test_sigmoid_zero() {
        let result = sigmoid(0.0);
        assert!((result - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5");
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let result = sigmoid(10.0);
        assert!(result > 0.999, "sigmoid(10) should be close to 1.0");
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let result = sigmoid(-10.0);
        assert!(result < 0.001, "sigmoid(-10) should be close to 0.0");
    }

    #[test]
    fn test_sigmoid_symmetry() {
        let pos = sigmoid(2.0);
        let neg = sigmoid(-2.0);
        assert!(
            (pos + neg - 1.0).abs() < 1e-6,
            "sigmoid(x) + sigmoid(-x) should equal 1.0"
        );
    }

    #[test]
    fn test_sigmoid_vec_basic() {
        let logits = vec![0.0, 10.0, -10.0];
        let results = sigmoid_vec(&logits);
        assert_eq!(results.len(), 3);
        assert!((results[0] - 0.5).abs() < 1e-6);
        assert!(results[1] > 0.999);
        assert!(results[2] < 0.001);
    }

    #[test]
    fn test_sigmoid_vec_empty() {
        let results = sigmoid_vec(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_build_detections_above_threshold() {
        use std::collections::HashMap;

        let mut inv_map = HashMap::new();
        inv_map.insert(0, "python".to_string());
        inv_map.insert(1, "markdown".to_string());

        let scores = vec![0.9, 0.8];
        let result = build_detections(&scores, &inv_map, 0.5);

        // python -> Code, markdown -> Prose
        assert!(result.contains_key(&TextCategory::Code));
        assert!(result.contains_key(&TextCategory::Prose));
        assert_eq!(result[&TextCategory::Code].len(), 1);
        assert_eq!(
            result[&TextCategory::Code][0].sub_type,
            ContentSubType::Python
        );
        assert!((result[&TextCategory::Code][0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_build_detections_below_threshold() {
        use std::collections::HashMap;

        let mut inv_map = HashMap::new();
        inv_map.insert(0, "python".to_string());

        let scores = vec![0.3];
        let result = build_detections(&scores, &inv_map, 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_build_detections_sorted_by_score_descending() {
        use std::collections::HashMap;

        let mut inv_map = HashMap::new();
        inv_map.insert(0, "python".to_string());
        inv_map.insert(1, "rust".to_string());
        inv_map.insert(2, "javascript".to_string());

        // All code subtypes, different scores
        let scores = vec![0.7, 0.9, 0.8];
        let result = build_detections(&scores, &inv_map, 0.5);

        let code_detections = &result[&TextCategory::Code];
        assert_eq!(code_detections.len(), 3);
        assert_eq!(code_detections[0].sub_type, ContentSubType::Rust);
        assert_eq!(code_detections[1].sub_type, ContentSubType::JavaScript);
        assert_eq!(code_detections[2].sub_type, ContentSubType::Python);
    }

    #[test]
    fn test_build_detections_unknown_label() {
        use std::collections::HashMap;

        let inv_map: HashMap<usize, String> = HashMap::new();
        // index 0 has no entry in map
        let scores = vec![0.9];
        let result = build_detections(&scores, &inv_map, 0.5);

        // "unknown" label maps to ContentSubType::Unknown, category Skip
        assert!(result.contains_key(&TextCategory::Skip));
        assert_eq!(
            result[&TextCategory::Skip][0].sub_type,
            ContentSubType::Unknown
        );
    }

    #[test]
    fn test_build_detections_empty_scores() {
        use std::collections::HashMap;
        let inv_map: HashMap<usize, String> = HashMap::new();
        let result = build_detections(&[], &inv_map, 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_marginalization_overrides_category_head() {
        use std::collections::HashMap;

        // cat_probs: code is highest at 0.60
        let cat_probs = vec![0.15, 0.60, 0.20, 0.05]; // prose, code, structured, skip
        let mut inv_cat = HashMap::new();
        inv_cat.insert(0, "prose".to_string());
        inv_cat.insert(1, "code".to_string());
        inv_cat.insert(2, "structured".to_string());
        inv_cat.insert(3, "skip".to_string());

        // sub_probs: ini(0.35) + key_value(0.30) + toml(0.15) = 0.80 for structured
        // remaining 0.20 spread across code sub-types
        let mut inv_sub = HashMap::new();
        inv_sub.insert(0, "ini".to_string());
        inv_sub.insert(1, "key_value".to_string());
        inv_sub.insert(2, "toml".to_string());
        inv_sub.insert(3, "python".to_string());
        inv_sub.insert(4, "rust".to_string());

        let sub_probs = vec![0.35, 0.30, 0.15, 0.12, 0.08];

        let classification = build_classification(
            &cat_probs,
            &sub_probs,
            None,
            &inv_cat,
            &inv_sub,
            None,
            DEFAULT_DETECTION_THRESHOLD,
        );

        // Marginalization should pick structured (0.80) over code (0.20)
        assert_eq!(
            classification.category,
            TextCategory::Structured,
            "marginalized category should be Structured, not Code"
        );
        assert!(
            (classification.confidence - 0.80).abs() < 0.01,
            "confidence should be ~0.80, got {}",
            classification.confidence
        );
        assert!(
            classification.reason.contains("marginalized"),
            "reason should mention marginalized, got: {}",
            classification.reason
        );
    }

    #[test]
    fn test_marginalization_unknown_sub_types_fall_to_skip() {
        use std::collections::HashMap;

        let cat_probs = vec![0.5, 0.5];
        let mut inv_cat = HashMap::new();
        inv_cat.insert(0, "prose".to_string());
        inv_cat.insert(1, "code".to_string());

        // inv_sub is empty — no indices map to known sub-types
        let inv_sub: HashMap<usize, String> = HashMap::new();
        let sub_probs = vec![0.6, 0.4];

        let classification = build_classification(
            &cat_probs,
            &sub_probs,
            None,
            &inv_cat,
            &inv_sub,
            None,
            DEFAULT_DETECTION_THRESHOLD,
        );

        // Unknown sub-types map to ContentSubType::Unknown -> TextCategory::Skip
        assert_eq!(
            classification.category,
            TextCategory::Skip,
            "unknown sub-types should result in Skip category"
        );
    }

    #[test]
    fn test_detection_threshold_from_config() {
        use std::collections::HashMap;

        // Detection logits that produce sigmoid scores between 0.3 and 0.5:
        // sigmoid(-0.4) ≈ 0.40, sigmoid(-0.8) ≈ 0.31
        // These should be included at threshold 0.3 but excluded at 0.5.
        let det_logits = vec![-0.4, -0.8]; // sigmoid ≈ [0.40, 0.31]

        let cat_probs = vec![0.1, 0.8, 0.1];
        let sub_probs = vec![0.1, 0.9];

        let mut inv_cat = HashMap::new();
        inv_cat.insert(0, "prose".to_string());
        inv_cat.insert(1, "code".to_string());
        inv_cat.insert(2, "structured".to_string());

        let mut inv_sub = HashMap::new();
        inv_sub.insert(0, "markdown".to_string());
        inv_sub.insert(1, "python".to_string());

        let mut inv_det = HashMap::new();
        inv_det.insert(0, "python".to_string());
        inv_det.insert(1, "markdown".to_string());

        // With the default threshold of 0.3, both detections should be included
        let default_threshold = crate::types::DEFAULT_DETECTION_THRESHOLD;
        assert!(
            (default_threshold - 0.3).abs() < f32::EPSILON,
            "DEFAULT_DETECTION_THRESHOLD should be 0.3"
        );

        let classification_low = build_classification(
            &cat_probs,
            &sub_probs,
            Some(&det_logits),
            &inv_cat,
            &inv_sub,
            Some(&inv_det),
            default_threshold,
        );
        // At 0.3 threshold, both scores (0.40, 0.30) should pass
        assert!(
            !classification_low.detections.is_empty(),
            "detections should not be empty at threshold 0.3"
        );
        let total_detections_low: usize = classification_low
            .detections
            .values()
            .map(|v| v.len())
            .sum();
        assert_eq!(
            total_detections_low, 2,
            "both detections should be included at threshold 0.3"
        );

        // With threshold 0.5, both should be excluded (scores are 0.40 and 0.30)
        let classification_high = build_classification(
            &cat_probs,
            &sub_probs,
            Some(&det_logits),
            &inv_cat,
            &inv_sub,
            Some(&inv_det),
            0.5,
        );
        assert!(
            classification_high.detections.is_empty(),
            "detections should be empty at threshold 0.5, got {:?}",
            classification_high.detections
        );
    }

    #[test]
    fn test_build_classification_includes_detections() {
        use std::collections::HashMap;

        let cat_probs = vec![0.1, 0.8, 0.1];
        let sub_probs = vec![0.1, 0.9];
        let det_logits = vec![2.0, 1.0];

        let mut inv_cat = HashMap::new();
        inv_cat.insert(0, "prose".to_string());
        inv_cat.insert(1, "code".to_string());
        inv_cat.insert(2, "structured".to_string());

        let mut inv_sub = HashMap::new();
        inv_sub.insert(0, "markdown".to_string());
        inv_sub.insert(1, "python".to_string());

        let mut inv_det = HashMap::new();
        inv_det.insert(0, "python".to_string());
        inv_det.insert(1, "markdown".to_string());

        let classification = build_classification(
            &cat_probs,
            &sub_probs,
            Some(&det_logits),
            &inv_cat,
            &inv_sub,
            Some(&inv_det),
            DEFAULT_DETECTION_THRESHOLD,
        );

        assert_eq!(classification.category, TextCategory::Code);
        assert_eq!(classification.sub_type, Some(ContentSubType::Python));
        assert_eq!(classification.tier, Tier::Model);
        assert_eq!(classification.detections[&TextCategory::Code].len(), 1);
        assert_eq!(
            classification.detections[&TextCategory::Code][0].sub_type,
            ContentSubType::Python
        );
        assert_eq!(classification.detections[&TextCategory::Prose].len(), 1);
        assert_eq!(
            classification.detections[&TextCategory::Prose][0].sub_type,
            ContentSubType::Markdown
        );
    }
}
