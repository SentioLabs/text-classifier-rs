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
}

impl From<Classification> for PyClassification {
    fn from(c: Classification) -> Self {
        PyClassification {
            category: c.category.to_string(),
            sub_type: c.sub_type.map(|s| s.label().to_string()),
            text_type: c.category.to_string(),
            confidence: c.confidence,
            reason: c.reason,
            tier: c.tier.to_string(),
        }
    }
}

#[pymethods]
impl PyClassification {
    fn __repr__(&self) -> String {
        format!(
            "Classification(category='{}', sub_type={}, confidence={:.2}, reason='{}', tier='{}')",
            self.category,
            self.sub_type
                .as_deref()
                .map_or("None".to_string(), |s| format!("'{s}'")),
            self.confidence,
            self.reason,
            self.tier
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
