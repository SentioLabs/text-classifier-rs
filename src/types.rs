use serde::{Deserialize, Serialize};

/// Text content category for translation eligibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextType {
    Translatable,
    Code,
    Tabular,
    PdfDump,
    Skip,
}

impl TextType {
    pub fn is_translatable(self) -> bool {
        matches!(self, TextType::Translatable)
    }
}

impl std::fmt::Display for TextType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextType::Translatable => write!(f, "translatable"),
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
}
