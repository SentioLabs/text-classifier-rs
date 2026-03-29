use serde::{Deserialize, Serialize};

/// Text content category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextCategory {
    Prose,
    Code,
    Structured,
    Artifact,
    Tabular,
    PdfDump,
    Skip,
}

/// Backward-compatible alias for `TextCategory`.
pub type TextType = TextCategory;

impl TextCategory {
    pub fn is_prose(self) -> bool {
        matches!(self, TextCategory::Prose)
    }
}

impl std::fmt::Display for TextCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextCategory::Prose => write!(f, "prose"),
            TextCategory::Code => write!(f, "code"),
            TextCategory::Structured => write!(f, "structured"),
            TextCategory::Artifact => write!(f, "artifact"),
            TextCategory::Tabular => write!(f, "tabular"),
            TextCategory::PdfDump => write!(f, "pdf_dump"),
            TextCategory::Skip => write!(f, "skip"),
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

/// Content sub-type providing finer-grained classification within a category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentSubType {
    // Prose
    Plain,
    Markdown,
    Rst,
    Latex,
    // Code
    Python,
    JavaScript,
    TypeScript,
    Rust,
    Go,
    Java,
    Sql,
    Shell,
    Css,
    // Code > Config
    Yaml,
    Toml,
    Ini,
    Dockerfile,
    Makefile,
    // Code > Markup
    Html,
    Xml,
    Sgml,
    // Structured > Tabular
    Csv,
    Tsv,
    PipeTable,
    FixedWidth,
    // Structured > Data
    Json,
    Jsonl,
    KeyValue,
    LogLines,
    // Artifact
    PdfDump,
    OcrGarbage,
    Boilerplate,
    // Skip
    TooShort,
    Empty,
    Ambiguous,
    // Fallback
    Unknown,
}

impl ContentSubType {
    /// Returns the parent `TextCategory` for this sub-type.
    pub fn category(&self) -> TextCategory {
        match self {
            // Prose
            ContentSubType::Plain
            | ContentSubType::Markdown
            | ContentSubType::Rst
            | ContentSubType::Latex => TextCategory::Prose,

            // Code (languages, config, markup)
            ContentSubType::Python
            | ContentSubType::JavaScript
            | ContentSubType::TypeScript
            | ContentSubType::Rust
            | ContentSubType::Go
            | ContentSubType::Java
            | ContentSubType::Sql
            | ContentSubType::Shell
            | ContentSubType::Css
            | ContentSubType::Yaml
            | ContentSubType::Toml
            | ContentSubType::Ini
            | ContentSubType::Dockerfile
            | ContentSubType::Makefile
            | ContentSubType::Html
            | ContentSubType::Xml
            | ContentSubType::Sgml => TextCategory::Code,

            // Structured (tabular + data)
            ContentSubType::Csv
            | ContentSubType::Tsv
            | ContentSubType::PipeTable
            | ContentSubType::FixedWidth
            | ContentSubType::Json
            | ContentSubType::Jsonl
            | ContentSubType::KeyValue
            | ContentSubType::LogLines => TextCategory::Structured,

            // Artifact
            ContentSubType::PdfDump | ContentSubType::OcrGarbage | ContentSubType::Boilerplate => {
                TextCategory::Artifact
            }

            // Skip + Fallback
            ContentSubType::TooShort
            | ContentSubType::Empty
            | ContentSubType::Ambiguous
            | ContentSubType::Unknown => TextCategory::Skip,
        }
    }

    /// Returns the snake_case label for this sub-type.
    pub fn label(&self) -> &'static str {
        match self {
            ContentSubType::Plain => "plain",
            ContentSubType::Markdown => "markdown",
            ContentSubType::Rst => "rst",
            ContentSubType::Latex => "latex",
            ContentSubType::Python => "python",
            ContentSubType::JavaScript => "javascript",
            ContentSubType::TypeScript => "typescript",
            ContentSubType::Rust => "rust",
            ContentSubType::Go => "go",
            ContentSubType::Java => "java",
            ContentSubType::Sql => "sql",
            ContentSubType::Shell => "shell",
            ContentSubType::Css => "css",
            ContentSubType::Yaml => "yaml",
            ContentSubType::Toml => "toml",
            ContentSubType::Ini => "ini",
            ContentSubType::Dockerfile => "dockerfile",
            ContentSubType::Makefile => "makefile",
            ContentSubType::Html => "html",
            ContentSubType::Xml => "xml",
            ContentSubType::Sgml => "sgml",
            ContentSubType::Csv => "csv",
            ContentSubType::Tsv => "tsv",
            ContentSubType::PipeTable => "pipe_table",
            ContentSubType::FixedWidth => "fixed_width",
            ContentSubType::Json => "json",
            ContentSubType::Jsonl => "jsonl",
            ContentSubType::KeyValue => "key_value",
            ContentSubType::LogLines => "log_lines",
            ContentSubType::PdfDump => "pdf_dump",
            ContentSubType::OcrGarbage => "ocr_garbage",
            ContentSubType::Boilerplate => "boilerplate",
            ContentSubType::TooShort => "too_short",
            ContentSubType::Empty => "empty",
            ContentSubType::Ambiguous => "ambiguous",
            ContentSubType::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for ContentSubType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Classification result for a single text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub category: TextCategory,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub_type: Option<ContentSubType>,
    pub confidence: f32,
    pub reason: String,
    pub tier: Tier,
}

impl Classification {
    /// Backward-compatible accessor for the category field.
    #[deprecated(note = "use `category` field instead")]
    pub fn text_type(&self) -> TextCategory {
        self.category
    }
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
    pub delimiter_consistency: f32,
    pub json_brace_depth: f32,
    pub key_value_ratio: f32,
    pub xml_tag_ratio: f32,
    pub log_line_ratio: f32,
    pub comment_ratio: f32,
    pub numeric_field_ratio: f32,
    pub repetitive_structure_score: f32,
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

impl FeatureVector {
    /// Creates a zeroed feature vector with all fields set to their default values.
    pub fn zeroed() -> Self {
        Self {
            line_length_cv: 0.0,
            char_entropy: 0.0,
            leading_whitespace_ratio: 0.0,
            tab_density: 0.0,
            sentence_punctuation_rate: 0.0,
            paragraph_break_rate: 0.0,
            alpha_ratio: 0.0,
            line_uniqueness: 0.0,
            short_line_ratio: 0.0,
            symbol_ratio: 0.0,
            delimiter_consistency: 0.0,
            json_brace_depth: 0.0,
            key_value_ratio: 0.0,
            xml_tag_ratio: 0.0,
            log_line_ratio: 0.0,
            comment_ratio: 0.0,
            numeric_field_ratio: 0.0,
            repetitive_structure_score: 0.0,
            line_count: 0,
        }
    }
}

/// Per-category confidence thresholds for Tier 1 acceptance.
pub mod thresholds {
    pub const PROSE: f32 = 0.65;
    pub const CODE: f32 = 0.70;
    pub const STRUCTURED: f32 = 0.60;
    pub const ARTIFACT: f32 = 0.75;
    pub const SUB_TYPE: f32 = 0.80;
}
