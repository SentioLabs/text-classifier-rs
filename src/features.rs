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
        delimiter_consistency: compute_delimiter_consistency(&lines),
        json_brace_depth: compute_json_brace_depth(sample, total_chars),
        key_value_ratio: compute_key_value_ratio(&lines, n_lines),
        xml_tag_ratio: compute_xml_tag_ratio(&lines, n_lines),
        log_line_ratio: compute_log_line_ratio(&lines, n_lines),
        comment_ratio: compute_comment_ratio(&lines, n_lines),
        numeric_field_ratio: compute_numeric_field_ratio(sample),
        repetitive_structure_score: compute_repetitive_structure_score(&lines),
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

/// Consistency of delimiter counts across lines.
/// For each candidate delimiter (`,` `|` `\t` `;`), count occurrences per line,
/// find the mode count, and compute fraction of lines matching that mode.
/// Returns the best (highest) consistency across all delimiters.
/// Returns 0.0 if fewer than 3 lines.
fn compute_delimiter_consistency(lines: &[&str]) -> f32 {
    if lines.len() < 3 {
        return 0.0;
    }

    let delimiters = [',', '|', '\t', ';'];
    let mut best = 0.0f32;

    for &delim in &delimiters {
        // Count occurrences of this delimiter per line
        let counts: Vec<usize> = lines
            .iter()
            .map(|line| line.chars().filter(|&c| c == delim).count())
            .collect();

        // Find mode count (most common count)
        let mut freq: HashMap<usize, usize> = HashMap::new();
        for &count in &counts {
            *freq.entry(count).or_insert(0) += 1;
        }

        // Find how many lines have the mode count
        let mode_freq = freq.values().copied().max().unwrap_or(0);

        // Skip if mode is 0 occurrences (delimiter not present)
        let mode_count = freq
            .iter()
            .filter(|&(_, v)| *v == mode_freq)
            .map(|(&k, _)| k)
            .next()
            .unwrap_or(0);

        if mode_count == 0 {
            continue;
        }

        let consistency = mode_freq as f32 / lines.len() as f32;
        if consistency > best {
            best = consistency;
        }
    }

    best
}

/// Fraction of JSON brace/bracket characters (`{` `}` `[` `]`) in text.
fn compute_json_brace_depth(text: &str, total_chars: f32) -> f32 {
    let count = text
        .chars()
        .filter(|c| matches!(c, '{' | '}' | '[' | ']'))
        .count();
    count as f32 / total_chars
}

/// Fraction of lines with key-value patterns (word followed by `:` or `=` then a value).
/// Uses simple char-level checks, not regex.
fn compute_key_value_ratio(lines: &[&str], n_lines: usize) -> f32 {
    let count = lines
        .iter()
        .filter(|line| {
            let trimmed = line.trim();
            // Look for `: ` or `=` preceded by a word character
            if let Some(pos) = trimmed.find(": ") {
                // Check there's at least one non-whitespace char before the colon
                pos > 0 && trimmed[..pos].chars().any(|c| c.is_alphanumeric())
            } else if let Some(pos) = trimmed.find('=') {
                // Check there's a word before = and something after
                pos > 0
                    && pos + 1 < trimmed.len()
                    && trimmed[..pos].chars().any(|c| c.is_alphanumeric())
            } else {
                false
            }
        })
        .count();

    count as f32 / n_lines as f32
}

/// Fraction of lines containing XML/HTML tags (`<` followed by a letter or `</` followed by a letter).
fn compute_xml_tag_ratio(lines: &[&str], n_lines: usize) -> f32 {
    let count = lines
        .iter()
        .filter(|line| {
            let chars: Vec<char> = line.chars().collect();
            for i in 0..chars.len() {
                if chars[i] == '<'
                    && let Some(&next) = chars.get(i + 1)
                {
                    if next.is_alphabetic() {
                        return true;
                    }
                    if next == '/'
                        && let Some(&after) = chars.get(i + 2)
                        && after.is_alphabetic()
                    {
                        return true;
                    }
                }
            }
            false
        })
        .count();

    count as f32 / n_lines as f32
}

/// Fraction of lines starting with timestamp-like patterns.
/// Detects: `\d{4}-\d{2}-\d{2}`, `\d{2}:\d{2}:\d{2}`, or `[\d{4}` (bracket timestamps).
/// Uses char-level checks for performance.
fn compute_log_line_ratio(lines: &[&str], n_lines: usize) -> f32 {
    let count = lines
        .iter()
        .filter(|line| {
            let trimmed = line.trim();
            let chars: Vec<char> = trimmed.chars().collect();

            // Pattern: \d{4}-\d{2}-\d{2}
            if chars.len() >= 10
                && chars[0].is_ascii_digit()
                && chars[1].is_ascii_digit()
                && chars[2].is_ascii_digit()
                && chars[3].is_ascii_digit()
                && chars[4] == '-'
                && chars[5].is_ascii_digit()
                && chars[6].is_ascii_digit()
                && chars[7] == '-'
                && chars[8].is_ascii_digit()
                && chars[9].is_ascii_digit()
            {
                return true;
            }

            // Pattern: \d{2}:\d{2}:\d{2}
            if chars.len() >= 8
                && chars[0].is_ascii_digit()
                && chars[1].is_ascii_digit()
                && chars[2] == ':'
                && chars[3].is_ascii_digit()
                && chars[4].is_ascii_digit()
                && chars[5] == ':'
                && chars[6].is_ascii_digit()
                && chars[7].is_ascii_digit()
            {
                return true;
            }

            // Pattern: [\d{4} (bracket timestamp)
            if chars.len() >= 5
                && chars[0] == '['
                && chars[1].is_ascii_digit()
                && chars[2].is_ascii_digit()
                && chars[3].is_ascii_digit()
                && chars[4].is_ascii_digit()
            {
                return true;
            }

            false
        })
        .count();

    count as f32 / n_lines as f32
}

/// Fraction of lines where trimmed content starts with comment markers.
/// Detects: `#`, `//`, `/*`, `--`, `%`.
fn compute_comment_ratio(lines: &[&str], n_lines: usize) -> f32 {
    let count = lines
        .iter()
        .filter(|line| {
            let trimmed = line.trim();
            trimmed.starts_with('#')
                || trimmed.starts_with("//")
                || trimmed.starts_with("/*")
                || trimmed.starts_with("--")
                || trimmed.starts_with('%')
        })
        .count();

    count as f32 / n_lines as f32
}

/// Fraction of whitespace-delimited tokens that parse as numbers.
/// Strips commas before parsing (e.g. "1,000" → "1000").
fn compute_numeric_field_ratio(text: &str) -> f32 {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.is_empty() {
        return 0.0;
    }

    let numeric = tokens
        .iter()
        .filter(|token| {
            let stripped: String = token.chars().filter(|&c| c != ',').collect();
            stripped.parse::<f64>().is_ok()
        })
        .count();

    numeric as f32 / tokens.len() as f32
}

/// Repetitive structure score: fraction of lines sharing the most common "shape".
/// Shape = (number of whitespace-separated tokens, set of delimiter chars present).
/// Samples first min(20, lines.len()) lines. Returns 0.0 if fewer than 3 lines.
fn compute_repetitive_structure_score(lines: &[&str]) -> f32 {
    if lines.len() < 3 {
        return 0.0;
    }

    let sample_size = lines.len().min(20);
    let sample = &lines[..sample_size];

    let delimiters = [',', '|', '\t', ';'];

    // Compute shape for each line
    let shapes: Vec<(usize, Vec<bool>)> = sample
        .iter()
        .map(|line| {
            let token_count = line.split_whitespace().count();
            let delim_present: Vec<bool> = delimiters.iter().map(|&d| line.contains(d)).collect();
            (token_count, delim_present)
        })
        .collect();

    // Count how many lines share each shape
    let mut freq: HashMap<(usize, Vec<bool>), usize> = HashMap::new();
    for shape in &shapes {
        *freq.entry(shape.clone()).or_insert(0) += 1;
    }

    let max_freq = freq.values().copied().max().unwrap_or(0);

    max_freq as f32 / sample_size as f32
}
