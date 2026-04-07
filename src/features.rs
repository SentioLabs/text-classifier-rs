use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use crate::types::FeatureVector;

static WORDLIST: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    include_str!("wordlist.txt")
        .lines()
        .filter(|l| !l.is_empty())
        .collect()
});

/// Maximum chars to sample from input text.
const SAMPLE_SIZE: usize = 10_000;

/// Maximum lines to consider for line uniqueness.
const UNIQUENESS_LINES: usize = 500;

/// Extract structural features from text.
///
/// Samples the first 10k characters for performance on large inputs.
/// Normalizes literal `\n` sequences to real newlines before processing.
pub fn extract_features(text: &str) -> FeatureVector {
    if text.is_empty() {
        return FeatureVector::zeroed();
    }

    let truncated = if text.len() > SAMPLE_SIZE {
        // Don't split mid-char for UTF-8 safety
        let end = text.floor_char_boundary(SAMPLE_SIZE);
        &text[..end]
    } else {
        text
    };

    // Normalize literal \n to real newlines so line-based features work
    // correctly on text arriving from JSON/CSV/API payloads.
    let normalized = truncated.replace("\\n", "\n");
    let sample = normalized.as_str();

    let lines: Vec<&str> = sample.lines().collect();
    let n_lines = lines.len().max(1);
    let total_chars = sample.chars().count().max(1) as f32;

    // Count words (whitespace-delimited tokens)
    let word_count = sample.split_whitespace().count().max(1) as f32;

    // Non-empty lines (used by several features)
    let non_empty: Vec<&str> = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .copied()
        .collect();
    let n_non_empty = non_empty.len().max(1);

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
        hyphenated_line_break_ratio: compute_hyphenated_line_break_ratio(&lines),
        short_repeated_line_ratio: compute_short_repeated_line_ratio(&lines),
        page_number_density: compute_page_number_density(&lines, n_lines),
        label_value_line_ratio: compute_label_value_line_ratio(&lines, n_lines),
        table_fragment_score: compute_table_fragment_score(&lines),
        uppercase_header_ratio: compute_uppercase_header_ratio(&lines),
        dictionary_word_ratio: compute_dictionary_word_ratio(sample),
        encoding_error_ratio: compute_encoding_error_ratio(sample, total_chars),
        repeated_ngram_ratio: compute_repeated_ngram_ratio(sample),
        sentence_coherence_score: compute_sentence_coherence_score(&lines),
        // New features (v2)
        avg_words_per_line: compute_avg_words_per_line(&non_empty, n_non_empty),
        operator_density: compute_operator_density(sample, total_chars),
        inline_markup_count: compute_inline_markup_count(sample, total_chars),
        indentation_consistency: compute_indentation_consistency(&lines),
        markup_heading_ratio: compute_markup_heading_ratio(&non_empty, n_non_empty),
        code_fence_density: compute_code_fence_density(&lines, n_lines),
        prose_paragraph_ratio: compute_prose_paragraph_ratio(&lines, n_lines),
        semicolon_line_ending_ratio: compute_semicolon_line_ending_ratio(&non_empty, n_non_empty),
        list_item_ratio: compute_list_item_ratio(&non_empty, n_non_empty),
        parenthesis_density: compute_parenthesis_density(sample, total_chars),
        section_header_ratio: compute_section_header_ratio(&non_empty, n_non_empty),
        json_lines_ratio: compute_json_lines_ratio(&non_empty, n_non_empty),
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

/// Fraction of non-alphanumeric chars, excluding common punctuation and
/// Unicode decorative characters (box-drawing, block elements, symbols).
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
                // Exclude Unicode decorative characters commonly found in
                // formatted documents, OCR output, and ASCII art:
                // Box Drawing (U+2500-257F), Block Elements (U+2580-259F),
                // Geometric Shapes (U+25A0-25FF), Misc Symbols (U+2600-26FF),
                // Dingbats (U+2700-27BF), Arrows (U+2190-21FF)
                && !matches!(*c as u32, 0x2190..=0x21FF | 0x2500..=0x259F | 0x25A0..=0x27BF)
                // Also exclude common typographic symbols: bullets, dashes, quotes
                && !matches!(c, '–' | '—' | '•' | '°' | '©' | '®' | '™' | '…' | '×' | '÷')
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

/// Fraction of line transitions where current line ends with `-` and next starts
/// with a lowercase letter (hyphenated line break pattern from OCR/PDF).
fn compute_hyphenated_line_break_ratio(lines: &[&str]) -> f32 {
    let n_transitions = (lines.len().saturating_sub(1)).max(1);

    let mut count = 0;
    for i in 0..lines.len().saturating_sub(1) {
        let current = lines[i].trim_end();
        if !current.ends_with('-') {
            continue;
        }
        let next = lines[i + 1].trim_start();
        if let Some(ch) = next.chars().next()
            && ch.is_alphabetic()
            && ch.is_lowercase()
        {
            count += 1;
        }
    }

    count as f32 / n_transitions as f32
}

/// Fraction of short lines (1-40 trimmed chars) that appear more than once.
fn compute_short_repeated_line_ratio(lines: &[&str]) -> f32 {
    let short_lines: Vec<&str> = lines
        .iter()
        .map(|l| l.trim())
        .filter(|l| {
            let len = l.len();
            (1..=40).contains(&len)
        })
        .collect();

    if short_lines.is_empty() {
        return 0.0;
    }

    let mut freq: HashMap<&str, usize> = HashMap::new();
    for line in &short_lines {
        *freq.entry(line).or_insert(0) += 1;
    }

    let repeated_instances: usize = freq.values().filter(|&&c| c > 1).sum();

    repeated_instances as f32 / short_lines.len() as f32
}

/// Fraction of non-empty trimmed lines matching page number patterns.
fn compute_page_number_density(lines: &[&str], n_lines: usize) -> f32 {
    let mut count = 0;

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if is_page_number_only(trimmed) || is_page_n_of_m(trimmed) || is_page_fraction(trimmed) {
            count += 1;
        }
    }

    count as f32 / n_lines as f32
}

/// Matches `^\d{1,4}$`
fn is_page_number_only(s: &str) -> bool {
    let len = s.len();
    (1..=4).contains(&len) && s.chars().all(|c| c.is_ascii_digit())
}

/// Matches `(?i)^page\s+\d{1,4}(\s+of\s+\d{1,4})?$`
fn is_page_n_of_m(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    let bytes = lower.as_bytes();

    if !lower.starts_with("page") {
        return false;
    }

    let mut i = 4; // past "page"

    // Require at least one whitespace
    if i >= bytes.len() || !bytes[i].is_ascii_whitespace() {
        return false;
    }
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }

    // Read 1-4 digits
    let digit_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let digit_count = i - digit_start;
    if !(1..=4).contains(&digit_count) {
        return false;
    }

    // End here? That's valid.
    if i == bytes.len() {
        return true;
    }

    // Otherwise expect " of \d{1,4}"
    if !bytes[i].is_ascii_whitespace() {
        return false;
    }
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }

    // "of"
    if i + 2 > bytes.len() || &lower[i..i + 2] != "of" {
        return false;
    }
    i += 2;

    // whitespace
    if i >= bytes.len() || !bytes[i].is_ascii_whitespace() {
        return false;
    }
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }

    // 1-4 digits
    let digit_start2 = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let digit_count2 = i - digit_start2;
    if !(1..=4).contains(&digit_count2) {
        return false;
    }

    i == bytes.len()
}

/// Matches `^\d{1,4}\s*/\s*\d{1,4}$`
fn is_page_fraction(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;

    // 1-4 digits
    let digit_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let digit_count = i - digit_start;
    if !(1..=4).contains(&digit_count) {
        return false;
    }

    // optional whitespace
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }

    // slash
    if i >= bytes.len() || bytes[i] != b'/' {
        return false;
    }
    i += 1;

    // optional whitespace
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }

    // 1-4 digits
    let digit_start2 = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let digit_count2 = i - digit_start2;
    if !(1..=4).contains(&digit_count2) {
        return false;
    }

    i == bytes.len()
}

/// Fraction of lines matching label-value pattern:
/// `^[A-Za-z][A-Za-z0-9 _/().]{0,30}\s*[:\-]\s+\S`
fn compute_label_value_line_ratio(lines: &[&str], n_lines: usize) -> f32 {
    let count = lines
        .iter()
        .filter(|line| is_label_value_line(line.trim()))
        .count();

    count as f32 / n_lines as f32
}

fn is_label_value_line(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return false;
    }

    // First char must be [A-Za-z]
    if !chars[0].is_ascii_alphabetic() {
        return false;
    }

    // Next 0-30 chars must be [A-Za-z0-9 _/().]
    let mut i = 1;
    let label_max = (chars.len()).min(32); // 1 + up to 30 more = 31 max index, but we need room for separator
    while i < label_max {
        let ch = chars[i];
        if ch.is_ascii_alphanumeric() || matches!(ch, ' ' | '_' | '/' | '(' | ')' | '.') {
            i += 1;
        } else {
            break;
        }
    }

    // Optional whitespace before separator
    while i < chars.len() && chars[i].is_ascii_whitespace() {
        i += 1;
    }

    // Separator must be : or -
    if i >= chars.len() || !matches!(chars[i], ':' | '-') {
        return false;
    }
    i += 1;

    // Must have at least one whitespace after separator
    if i >= chars.len() || !chars[i].is_ascii_whitespace() {
        return false;
    }
    while i < chars.len() && chars[i].is_ascii_whitespace() {
        i += 1;
    }

    // Must have a non-whitespace char after
    if i >= chars.len() {
        return false;
    }
    !chars[i].is_whitespace()
}

/// Fraction of non-empty lines that look table-like (have delimiters or
/// multi-space column separation).
fn compute_table_fragment_score(lines: &[&str]) -> f32 {
    let non_empty: Vec<&str> = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .copied()
        .collect();
    if non_empty.is_empty() {
        return 0.0;
    }

    let mut score = 0;
    for line in &non_empty {
        let delimiter_hits = line
            .chars()
            .filter(|&c| matches!(c, ',' | '|' | '\t' | ';'))
            .count();
        if delimiter_hits >= 2 || has_spaced_columns(line) {
            score += 1;
        }
    }

    score as f32 / non_empty.len() as f32
}

/// Check if line has `\S+\s{2,}\S+` pattern (multi-space column separation).
fn has_spaced_columns(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut i = 0;

    // Find a non-whitespace char
    while i < len {
        if !chars[i].is_whitespace() {
            // Found non-whitespace, now look for 2+ spaces followed by non-whitespace
            i += 1;
            while i < len && !chars[i].is_whitespace() {
                i += 1;
            }
            // Now at whitespace or end
            let ws_start = i;
            while i < len && chars[i].is_whitespace() {
                i += 1;
            }
            let ws_count = i - ws_start;
            if ws_count >= 2 && i < len && !chars[i].is_whitespace() {
                return true;
            }
            // Continue scanning from current position
        } else {
            i += 1;
        }
    }

    false
}

/// Fraction of non-empty trimmed lines that look like uppercase section headers.
fn compute_uppercase_header_ratio(lines: &[&str]) -> f32 {
    let non_empty: Vec<&str> = lines
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if non_empty.is_empty() {
        return 0.0;
    }

    let mut count = 0;
    for line in &non_empty {
        if line.len() > 80 {
            continue;
        }
        let alpha_chars: Vec<char> = line.chars().filter(|c| c.is_alphabetic()).collect();
        if alpha_chars.len() < 3 {
            continue;
        }
        let upper_count = alpha_chars.iter().filter(|c| c.is_uppercase()).count();
        if (upper_count as f32 / alpha_chars.len() as f32) >= 0.8 {
            // Last char must NOT be . ! ?
            if let Some(last) = line.chars().next_back()
                && matches!(last, '.' | '!' | '?')
            {
                continue;
            }
            count += 1;
        }
    }

    count as f32 / non_empty.len() as f32
}

/// Fraction of whitespace-delimited tokens found in the dictionary wordlist.
fn compute_dictionary_word_ratio(text: &str) -> f32 {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut valid_count = 0;
    let mut found_count = 0;

    for token in &tokens {
        let lower = token.to_lowercase();
        let stripped = lower
            .trim_start_matches(|c: char| c.is_ascii_punctuation())
            .trim_end_matches(|c: char| c.is_ascii_punctuation());
        if stripped.is_empty() {
            continue;
        }
        valid_count += 1;
        if WORDLIST.contains(stripped) {
            found_count += 1;
        }
    }

    if valid_count == 0 {
        return 0.0;
    }

    found_count as f32 / valid_count as f32
}

/// Fraction of characters that are encoding errors (U+FFFD) or mojibake sequences.
fn compute_encoding_error_ratio(text: &str, total_chars: f32) -> f32 {
    let fffd_count = text.chars().filter(|&c| c == '\u{FFFD}').count();

    let mojibake_sequences: &[&str] = &[
        "\u{00C3}\u{00A9}",         // Ã©
        "\u{00C3}\u{00A8}",         // Ã¨
        "\u{00C3}\u{00BC}",         // Ã¼
        "\u{00C3}\u{00B6}",         // Ã¶
        "\u{00C3}\u{00A4}",         // Ã¤
        "\u{00C2}\u{00B0}",         // Â°
        "\u{00C2}\u{00A9}",         // Â©
        "\u{00E2}\u{0080}\u{0093}", // em-dash mojibake
        "\u{00E2}\u{0080}\u{0099}", // right-single-quote mojibake
        "\u{00E2}\u{0080}\u{009C}", // left-double-quote mojibake
        "\u{00E2}\u{0080}\u{009D}", // right-double-quote mojibake
    ];

    let mojibake_count: usize = mojibake_sequences
        .iter()
        .map(|seq| text.matches(seq).count())
        .sum();

    (fffd_count + mojibake_count) as f32 / total_chars
}

/// Fraction of unique 3-gram types that appear more than once.
fn compute_repeated_ngram_ratio(text: &str) -> f32 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 3 {
        return 0.0;
    }

    let mut freq: HashMap<(&str, &str, &str), usize> = HashMap::new();
    for i in 0..words.len() - 2 {
        let ngram = (words[i], words[i + 1], words[i + 2]);
        *freq.entry(ngram).or_insert(0) += 1;
    }

    let total_unique = freq.len();
    if total_unique == 0 {
        return 0.0;
    }

    let repeated = freq.values().filter(|&&c| c > 1).count();
    repeated as f32 / total_unique as f32
}

/// Fraction of non-empty trimmed lines that start with uppercase and end with `.!?`.
fn compute_sentence_coherence_score(lines: &[&str]) -> f32 {
    let non_empty: Vec<&str> = lines
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if non_empty.is_empty() {
        return 0.0;
    }

    let mut proper = 0;
    for line in &non_empty {
        let first = line.chars().next().unwrap();
        let last = line.chars().last().unwrap();
        if first.is_uppercase() && matches!(last, '.' | '!' | '?') {
            proper += 1;
        }
    }

    proper as f32 / non_empty.len() as f32
}

// ---------------------------------------------------------------------------
// New features (v2): 10 additional features
// ---------------------------------------------------------------------------

/// Average whitespace-delimited tokens per non-empty line.
/// Prose: ~10-80 words/line. Code: ~3-8. Structured: ~2-10.
fn compute_avg_words_per_line(non_empty: &[&str], n_non_empty: usize) -> f32 {
    if n_non_empty == 0 {
        return 0.0;
    }
    let total_words: usize = non_empty.iter().map(|l| l.split_whitespace().count()).sum();
    total_words as f32 / n_non_empty as f32
}

/// Multi-character programming operators per 1000 characters.
/// Counts ==, !=, >=, <=, &&, ||, =>, ->, +=, -=, etc.
fn compute_operator_density(text: &str, total_chars: f32) -> f32 {
    const OPERATORS: &[&str] = &[
        "==", "!=", ">=", "<=", "&&", "||", "=>", "->", "+=", "-=", "*=", "/=", "**", "<<", ">>",
        "::",
    ];
    let count: usize = OPERATORS.iter().map(|op| text.matches(op).count()).sum();
    count as f32 / total_chars * 1000.0
}

/// Inline markup patterns per 1000 characters.
/// Detects **bold**, `code`, *italic*, [link](url).
fn compute_inline_markup_count(text: &str, total_chars: f32) -> f32 {
    let mut count = 0usize;

    // Count **bold** patterns
    let mut rest = text;
    while let Some(start) = rest.find("**") {
        let after = &rest[start + 2..];
        if let Some(end) = after.find("**") {
            if end > 0 {
                count += 1;
            }
            rest = &after[end + 2..];
        } else {
            break;
        }
    }

    // Count `code` patterns
    rest = text;
    while let Some(start) = rest.find('`') {
        let after = &rest[start + 1..];
        if let Some(stripped) = after.strip_prefix('`') {
            // Skip `` (double backtick)
            rest = stripped;
            continue;
        }
        if let Some(end) = after.find('`') {
            if end > 0 {
                count += 1;
            }
            rest = &after[end + 1..];
        } else {
            break;
        }
    }

    // Count [link](url) patterns
    rest = text;
    while let Some(start) = rest.find("](") {
        // Check for [ before ]
        let before = &rest[..start];
        if before.rfind('[').is_some() {
            let after = &rest[start + 2..];
            if let Some(end) = after.find(')') {
                if end > 0 {
                    count += 1;
                }
                rest = &after[end + 1..];
            } else {
                break;
            }
        } else {
            rest = &rest[start + 2..];
        }
    }

    count as f32 / total_chars * 1000.0
}

/// Whether indentation follows a regular pattern (0.0-1.0).
/// Code has consistent indentation (multiples of 2, 3, or 4).
fn compute_indentation_consistency(lines: &[&str]) -> f32 {
    let mut indent_levels: Vec<usize> = Vec::new();

    for line in lines {
        let stripped = line.trim_start();
        if stripped.is_empty() || stripped.len() == line.len() {
            continue;
        }
        let indent: usize = line
            .chars()
            .take_while(|c| c.is_whitespace())
            .map(|c| if c == '\t' { 4 } else { 1 })
            .sum();
        indent_levels.push(indent);
    }

    if indent_levels.len() < 3 {
        return 0.0;
    }

    let n = indent_levels.len() as f32;
    let mut best = 0.0f32;
    for base in [2, 3, 4] {
        let consistent = indent_levels.iter().filter(|&&i| i % base == 0).count();
        let ratio = consistent as f32 / n;
        if ratio > best {
            best = ratio;
        }
    }

    best
}

/// Fraction of non-empty lines that are Markdown/RST section headings.
fn compute_markup_heading_ratio(non_empty: &[&str], n_non_empty: usize) -> f32 {
    if n_non_empty == 0 {
        return 0.0;
    }

    const RST_UNDERLINE_CHARS: &[char] = &['=', '-', '*', '~', '^', '"', '\'', '`'];

    let mut count = 0;
    for line in non_empty {
        let stripped = line.trim();
        // Markdown heading: # followed by space
        if stripped.starts_with('#') {
            let rest = stripped.trim_start_matches('#');
            if rest.starts_with(' ') && !rest.trim().is_empty() {
                count += 1;
                continue;
            }
        }
        // RST underline: all same char from the set, length >= 3
        if stripped.len() >= 3 && stripped.chars().all(|c| RST_UNDERLINE_CHARS.contains(&c)) {
            count += 1;
        }
    }

    count as f32 / n_non_empty as f32
}

/// Fraction of lines inside triple-backtick fenced code blocks.
fn compute_code_fence_density(lines: &[&str], n_lines: usize) -> f32 {
    if n_lines == 0 {
        return 0.0;
    }

    let mut inside = false;
    let mut fenced_lines = 0;

    for line in lines {
        let stripped = line.trim();
        if stripped.starts_with("```") {
            inside = !inside;
            continue;
        }
        if inside {
            fenced_lines += 1;
        }
    }

    fenced_lines as f32 / n_lines as f32
}

/// Fraction of lines in multi-sentence paragraph blocks.
/// A paragraph block is 3+ consecutive non-empty lines of >40 chars
/// without structural delimiters.
fn compute_prose_paragraph_ratio(lines: &[&str], n_lines: usize) -> f32 {
    if n_lines == 0 {
        return 0.0;
    }

    let structural_chars: &[char] = &['|', '{', '}', '\t'];
    let mut para_lines = 0;
    let mut streak = 0;

    for line in lines {
        let stripped = line.trim();
        let is_para = stripped.len() > 40
            && !stripped.chars().any(|c| structural_chars.contains(&c))
            && !stripped.starts_with('#');

        if is_para {
            streak += 1;
        } else {
            if streak >= 3 {
                para_lines += streak;
            }
            streak = 0;
        }
    }
    if streak >= 3 {
        para_lines += streak;
    }

    para_lines as f32 / n_lines as f32
}

/// Fraction of non-empty lines ending with semicolons.
fn compute_semicolon_line_ending_ratio(non_empty: &[&str], n_non_empty: usize) -> f32 {
    if n_non_empty == 0 {
        return 0.0;
    }
    let count = non_empty
        .iter()
        .filter(|l| l.trim_end().ends_with(';'))
        .count();
    count as f32 / n_non_empty as f32
}

/// Fraction of non-empty lines that are list items.
/// Detects: - item, * item, • item, 1. item, a) item.
fn compute_list_item_ratio(non_empty: &[&str], n_non_empty: usize) -> f32 {
    if n_non_empty == 0 {
        return 0.0;
    }

    let count = non_empty.iter().filter(|line| is_list_item(line)).count();
    count as f32 / n_non_empty as f32
}

fn is_list_item(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.is_empty() {
        return false;
    }

    // Bullet: - item, * item, • item
    if let Some(first) = trimmed.chars().next()
        && matches!(first, '-' | '*' | '•')
        && let Some(second) = trimmed.chars().nth(1)
        && second == ' '
        && let Some(third) = trimmed.chars().nth(2)
        && !third.is_whitespace()
    {
        return true;
    }

    // Numbered: 1. item, 1) item
    let chars: Vec<char> = trimmed.chars().collect();
    if !chars.is_empty() && chars[0].is_ascii_digit() {
        let mut i = 0;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i < chars.len()
            && matches!(chars[i], '.' | ')')
            && i + 1 < chars.len()
            && chars[i + 1] == ' '
            && i + 2 < chars.len()
            && !chars[i + 2].is_whitespace()
        {
            return true;
        }
    }

    false
}

/// Parentheses per 1000 characters. Code has high density (function calls).
fn compute_parenthesis_density(text: &str, total_chars: f32) -> f32 {
    let count = text.chars().filter(|&c| c == '(' || c == ')').count();
    count as f32 / total_chars * 1000.0
}

/// Fraction of non-empty lines that are INI-style section headers: `[section.name]`.
/// Line must start with `[`, end with `]`, and contain only word chars, spaces, dots, hyphens.
fn compute_section_header_ratio(non_empty: &[&str], n_non_empty: usize) -> f32 {
    if n_non_empty == 0 {
        return 0.0;
    }

    let count = non_empty
        .iter()
        .filter(|line| is_section_header(line.trim()))
        .count();
    count as f32 / n_non_empty as f32
}

fn is_section_header(trimmed: &str) -> bool {
    let bytes = trimmed.as_bytes();
    if bytes.len() < 3 {
        return false;
    }
    if bytes[0] != b'[' || bytes[bytes.len() - 1] != b']' {
        return false;
    }
    // Check contents (between [ and ]) are only word chars, spaces, dots, hyphens
    for &b in &bytes[1..bytes.len() - 1] {
        if b.is_ascii_alphanumeric() || b == b'_' || b == b' ' || b == b'.' || b == b'-' {
            continue;
        }
        return false;
    }
    true
}

/// Fraction of non-empty lines that look like JSON lines (start with `{`, end with `}`).
fn compute_json_lines_ratio(non_empty: &[&str], n_non_empty: usize) -> f32 {
    if n_non_empty == 0 {
        return 0.0;
    }

    let count = non_empty
        .iter()
        .filter(|line| {
            let trimmed = line.trim();
            trimmed.starts_with('{') && trimmed.ends_with('}')
        })
        .count();
    count as f32 / n_non_empty as f32
}
