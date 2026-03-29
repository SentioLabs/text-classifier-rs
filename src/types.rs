use serde::{Deserialize, Serialize};

/// Text content category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextType {
    Prose,
    Code,
    Tabular,
    PdfDump,
    Skip,
}

impl TextType {
    pub fn is_prose(self) -> bool {
        matches!(self, TextType::Prose)
    }
}

impl std::fmt::Display for TextType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextType::Prose => write!(f, "prose"),
            TextType::Code => write!(f, "code"),
            TextType::Tabular => write!(f, "tabular"),
            TextType::PdfDump => write!(f, "pdf_dump"),
            TextType::Skip => write!(f, "skip"),
        }
    }
}

/// Which classification tier made the decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Structural,
    Model,
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tier::Structural => write!(f, "structural"),
            Tier::Model => write!(f, "model"),
        }
    }
}

/// Classification result for a single text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub text_type: TextType,
    pub confidence: f32,
    pub reason: String,
    pub tier: Tier,
}

/// Raw structural features extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub line_length_cv: f32,
    pub char_entropy: f32,
    pub leading_whitespace_ratio: f32,
    pub tab_density: f32,
    pub sentence_punctuation_rate: f32,
    pub paragraph_break_rate: f32,
    pub alpha_ratio: f32,
    pub line_uniqueness: f32,
    pub short_line_ratio: f32,
    pub symbol_ratio: f32,
    /// Number of lines in the sampled text. Used by rules that need
    /// a minimum sample size (e.g. tabular detection).
    pub line_count: usize,
    /// Consistency of delimiter counts across lines (CSV/TSV detection).
    pub delimiter_consistency: f32,
    /// Fraction of JSON brace/bracket characters in text.
    pub json_brace_depth: f32,
    /// Fraction of lines with key-value patterns (key: value or key=value).
    pub key_value_ratio: f32,
    /// Fraction of lines containing XML/HTML tags.
    pub xml_tag_ratio: f32,
    /// Fraction of lines starting with timestamp-like patterns.
    pub log_line_ratio: f32,
    /// Fraction of lines that are comments (# // /* -- %).
    pub comment_ratio: f32,
    /// Fraction of whitespace-delimited tokens that parse as numbers.
    pub numeric_field_ratio: f32,
    /// How many lines share the most common "shape" (token count + delimiters).
    pub repetitive_structure_score: f32,
}
