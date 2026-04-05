use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::Classifier as RustClassifier;
use crate::types::Classification;

#[pyclass(name = "Classification", from_py_object)]
#[derive(Clone)]
struct PyClassification {
    #[pyo3(get)]
    category: String,
    #[pyo3(get)]
    sub_type: Option<String>,
    #[pyo3(get)]
    text_type: String,
    #[pyo3(get)]
    confidence: f32,
    #[pyo3(get)]
    reason: String,
    #[pyo3(get)]
    tier: String,
    #[pyo3(get)]
    detections: String,
}

impl From<Classification> for PyClassification {
    fn from(c: Classification) -> Self {
        let detections = serde_json::to_string(&c.detections).unwrap_or_default();
        PyClassification {
            category: c.category.to_string(),
            sub_type: c.sub_type.map(|s| s.label().to_string()),
            text_type: c.category.to_string(),
            confidence: c.confidence,
            reason: c.reason,
            tier: c.tier.to_string(),
            detections,
        }
    }
}

#[pymethods]
impl PyClassification {
    fn __repr__(&self) -> String {
        let detection_count = serde_json::from_str::<serde_json::Value>(&self.detections)
            .ok()
            .and_then(|v| v.as_object().map(|obj| {
                obj.values()
                    .filter_map(|arr| arr.as_array().map(|a| a.len()))
                    .sum::<usize>()
            }))
            .unwrap_or(0);
        format!(
            "Classification(category='{}', sub_type={}, confidence={:.2}, reason='{}', tier='{}', {} detections)",
            self.category,
            self.sub_type
                .as_deref()
                .map_or("None".to_string(), |s| format!("'{s}'")),
            self.confidence,
            self.reason,
            self.tier,
            detection_count
        )
    }
}

#[pyclass(name = "Classifier")]
struct PyClassifier {
    inner: RustClassifier,
}

#[pymethods]
impl PyClassifier {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyClassifier {
            inner: RustClassifier::new(),
        })
    }

    fn classify(&self, text: &str) -> PyClassification {
        self.inner.classify(text).into()
    }

    fn classify_batch(&self, texts: Vec<String>) -> Vec<PyClassification> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner
            .classify_batch(&refs)
            .into_iter()
            .map(PyClassification::from)
            .collect()
    }

    fn extract_features<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
        let f = self.inner.extract_features(text);
        let dict = PyDict::new(py);
        dict.set_item("line_length_cv", f.line_length_cv)?;
        dict.set_item("char_entropy", f.char_entropy)?;
        dict.set_item("leading_whitespace_ratio", f.leading_whitespace_ratio)?;
        dict.set_item("tab_density", f.tab_density)?;
        dict.set_item("sentence_punctuation_rate", f.sentence_punctuation_rate)?;
        dict.set_item("paragraph_break_rate", f.paragraph_break_rate)?;
        dict.set_item("alpha_ratio", f.alpha_ratio)?;
        dict.set_item("line_uniqueness", f.line_uniqueness)?;
        dict.set_item("short_line_ratio", f.short_line_ratio)?;
        dict.set_item("symbol_ratio", f.symbol_ratio)?;
        dict.set_item("delimiter_consistency", f.delimiter_consistency)?;
        dict.set_item("json_brace_depth", f.json_brace_depth)?;
        dict.set_item("key_value_ratio", f.key_value_ratio)?;
        dict.set_item("xml_tag_ratio", f.xml_tag_ratio)?;
        dict.set_item("log_line_ratio", f.log_line_ratio)?;
        dict.set_item("comment_ratio", f.comment_ratio)?;
        dict.set_item("numeric_field_ratio", f.numeric_field_ratio)?;
        dict.set_item("repetitive_structure_score", f.repetitive_structure_score)?;
        dict.set_item("line_count", f.line_count)?;
        Ok(dict)
    }
}

#[pyfunction]
fn classify(text: &str) -> PyClassification {
    let classifier = RustClassifier::new();
    classifier.classify(text).into()
}

#[pymodule]
fn text_classifier(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClassifier>()?;
    m.add_class::<PyClassification>()?;
    m.add_function(wrap_pyfunction!(classify, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Classification, Detection, ContentSubType, TextCategory, Tier};
    use std::collections::BTreeMap;

    fn make_classification_with_detections() -> Classification {
        let mut detections = BTreeMap::new();
        detections.insert(
            TextCategory::Code,
            vec![
                Detection { sub_type: ContentSubType::Python, score: 0.85 },
                Detection { sub_type: ContentSubType::Shell, score: 0.10 },
            ],
        );
        Classification {
            category: TextCategory::Code,
            sub_type: Some(ContentSubType::Python),
            confidence: 0.85,
            reason: "code detected".to_string(),
            tier: Tier::Structural,
            detections,
        }
    }

    #[test]
    fn test_py_classification_has_detections_field() {
        let c = make_classification_with_detections();
        let py_c = PyClassification::from(c);
        // The detections field should be a JSON string
        assert!(!py_c.detections.is_empty());
    }

    #[test]
    fn test_py_classification_detections_is_valid_json() {
        let c = make_classification_with_detections();
        let py_c = PyClassification::from(c);
        let parsed: serde_json::Value = serde_json::from_str(&py_c.detections)
            .expect("detections should be valid JSON");
        assert!(parsed.is_object());
        // Should contain the "code" key
        assert!(parsed.get("code").is_some());
        let code_detections = parsed["code"].as_array().unwrap();
        assert_eq!(code_detections.len(), 2);
        assert_eq!(code_detections[0]["sub_type"], "python");
        assert_eq!(code_detections[0]["score"], 0.85);
    }

    #[test]
    fn test_py_classification_empty_detections() {
        let c = Classification {
            category: TextCategory::Prose,
            sub_type: Some(ContentSubType::Plain),
            confidence: 0.90,
            reason: "prose detected".to_string(),
            tier: Tier::Structural,
            detections: BTreeMap::new(),
        };
        let py_c = PyClassification::from(c);
        let parsed: serde_json::Value = serde_json::from_str(&py_c.detections)
            .expect("detections should be valid JSON");
        assert!(parsed.is_object());
        assert_eq!(parsed.as_object().unwrap().len(), 0);
    }

    #[test]
    fn test_py_classification_repr_includes_detections() {
        let c = make_classification_with_detections();
        let py_c = PyClassification::from(c);
        let repr = py_c.__repr__();
        assert!(repr.contains("2 detections"), "repr should mention detection count, got: {repr}");
    }
}
