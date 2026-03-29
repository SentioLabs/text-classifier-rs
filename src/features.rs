use std::collections::{HashMap, HashSet};

use crate::types::FeatureVector;

/// Maximum chars to sample from input text.
const SAMPLE_SIZE: usize = 10_000;

/// Maximum lines to consider for line uniqueness.
const UNIQUENESS_LINES: usize = 500;

/// Extract structural features from text.
///
/// Samples the first 10k characters for performance on large inputs.
pub fn extract_features(text: &str) -> FeatureVector {
    if text.is_empty() {
        return FeatureVector::zeroed();
    }

    let sample = if text.len() > SAMPLE_SIZE {
        // Don't split mid-char for UTF-8 safety
        let end = text.floor_char_boundary(SAMPLE_SIZE);
        &text[..end]
    } else {
        text
    };

    let lines: Vec<&str> = sample.lines().collect();
    let n_lines = lines.len().max(1);
    let total_chars = sample.chars().count().max(1) as f32;

    // Count words (whitespace-delimited tokens)
    let word_count = sample.split_whitespace().count().max(1) as f32;

    FeatureVector {
        line_length_cv: compute_line_length_cv(&lines),
        char_entropy: compute_char_entropy(sample),
        leading_whitespace_ratio: compute_leading_whitespace_ratio(&lines, n_lines),
        tab_density: compute_tab_density(sample, total_chars),
        sentence_punctuation_rate: compute_sentence_punctuation_rate(sample, word_count),
        paragraph_break_rate: compute_paragraph_break_rate(sample, n_lines),
        alpha_ratio: compute_alpha_ratio(sample, total_chars),
        line_uniqueness: compute_line_uniqueness(&lines),
        short_line_ratio: compute_short_line_ratio(&lines, n_lines),
        symbol_ratio: compute_symbol_ratio(sample, total_chars),
        delimiter_consistency: 0.0,
        json_brace_depth: 0.0,
        key_value_ratio: 0.0,
        xml_tag_ratio: 0.0,
        log_line_ratio: 0.0,
        comment_ratio: 0.0,
        numeric_field_ratio: 0.0,
        repetitive_structure_score: 0.0,
        line_count: n_lines,
    }
}

/// Coefficient of variation of line lengths: std_dev / mean.
/// High CV = variable line lengths (prose). Low CV = uniform (tables).
fn compute_line_length_cv(lines: &[&str]) -> f32 {
    if lines.len() < 2 {
        return 0.0;
    }

    let lengths: Vec<f32> = lines.iter().map(|l| l.len() as f32).collect();
    let n = lengths.len() as f32;
    let mean = lengths.iter().sum::<f32>() / n;

    if mean < 1.0 {
        return 0.0;
    }

    let variance = lengths.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();

    std_dev / mean
}

/// Shannon entropy of character distribution (bits per character).
/// Not currently used in Tier 1 rules — available for analysis and Tier 2.
fn compute_char_entropy(text: &str) -> f32 {
    let mut freq: HashMap<char, u32> = HashMap::new();
    let mut total = 0u32;

    for ch in text.chars() {
        *freq.entry(ch).or_insert(0) += 1;
        total += 1;
    }

    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;
    let entropy: f64 = freq
        .values()
        .map(|&count| {
            let p = count as f64 / total_f;
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        })
        .sum();

    entropy as f32
}

/// Fraction of lines that start with >2 columns of whitespace.
/// Tabs count as 4 columns to correctly detect tab-indented code (Go, Makefiles).
/// High = code indentation pattern.
fn compute_leading_whitespace_ratio(lines: &[&str], n_lines: usize) -> f32 {
    let count = lines
        .iter()
        .filter(|line| {
            let leading: usize = line
                .chars()
                .take_while(|c| c.is_whitespace())
                .map(|c| if c == '\t' { 4 } else { 1 })
                .sum();
            leading > 2
        })
        .count();

    count as f32 / n_lines as f32
}

/// Tab characters as a fraction of total characters.
/// High = TSV or spreadsheet data.
fn compute_tab_density(text: &str, total_chars: f32) -> f32 {
    let tabs = text.chars().filter(|c| *c == '\t').count();
    tabs as f32 / total_chars
}

/// Sentence-ending punctuation (. ! ?) followed by space or end-of-line, per word.
/// Prose: ~0.04-0.08 (one sentence per 12-25 words). Code/tables: ~0.
fn compute_sentence_punctuation_rate(text: &str, word_count: f32) -> f32 {
    let chars: Vec<char> = text.chars().collect();
    let mut count = 0;

    for i in 0..chars.len() {
        if matches!(chars[i], '.' | '!' | '?') {
            // Must be followed by whitespace, newline, or end of text
            let next = chars.get(i + 1);
            if next.is_none() || next.is_some_and(|c| c.is_whitespace()) {
                // Avoid counting things like "3.14" or "e.g."
                // Simple heuristic: previous char should be a letter
                if i > 0 && chars[i - 1].is_alphabetic() {
                    count += 1;
                }
            }
        }
    }

    count as f32 / word_count
}

/// Double-newline (paragraph break) frequency per line.
/// Not currently used in Tier 1 rules — available for analysis and Tier 2.
fn compute_paragraph_break_rate(text: &str, n_lines: usize) -> f32 {
    let breaks = text.matches("\n\n").count();
    breaks as f32 / n_lines as f32
}

/// Fraction of characters that are alphanumeric or space.
/// Prose > 0.75. Code/data lower due to symbols.
fn compute_alpha_ratio(text: &str, total_chars: f32) -> f32 {
    let alpha = text
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ')
        .count();
    alpha as f32 / total_chars
}

/// Ratio of unique lines to total lines within the first 500 lines.
/// Data dumps have many repeated lines (< 0.3).
fn compute_line_uniqueness(lines: &[&str]) -> f32 {
    let sample: &[&str] = if lines.len() > UNIQUENESS_LINES {
        &lines[..UNIQUENESS_LINES]
    } else {
        lines
    };

    let unique: HashSet<&&str> = sample.iter().collect();
    let sample_count = sample.len().max(1);

    unique.len() as f32 / sample_count as f32
}

/// Fraction of lines with 1-14 characters.
/// OCR/PDF dumps have many very short lines.
fn compute_short_line_ratio(lines: &[&str], n_lines: usize) -> f32 {
    let short = lines
        .iter()
        .filter(|l| {
            let trimmed_len = l.trim().len();
            (1..=14).contains(&trimmed_len)
        })
        .count();

    short as f32 / n_lines as f32
}

/// Fraction of non-alphanumeric chars, excluding common punctuation.
/// Common punctuation excluded: space, newline, tab, . , ; : ! ? - ' "
/// Code/garbled text > 0.15. Prose < 0.05.
fn compute_symbol_ratio(text: &str, total_chars: f32) -> f32 {
    let symbols = text
        .chars()
        .filter(|c| {
            !c.is_alphanumeric()
                && !matches!(
                    c,
                    ' ' | '\n' | '\t' | '\r' | '.' | ',' | ';' | ':' | '!' | '?' | '-' | '\'' | '"'
                )
        })
        .count();

    symbols as f32 / total_chars
}
