use std::io::{self, BufRead, Read, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use text_classifier::tier1::MIN_CONFIDENCE;
use text_classifier::{Classifier, TextCategory, classify};

#[derive(Parser)]
#[command(name = "classify", about = "Classify text by structural type")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to fasttext model file (optional, enables Tier 2)
    #[arg(long, global = true)]
    model: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Classify texts from a JSONL file
    File {
        /// Input JSONL file (supports .gz)
        input: PathBuf,
        /// Output JSONL file
        #[arg(short, long)]
        output: PathBuf,
        /// Field to classify (dot notation, e.g. "fields.bodytext")
        #[arg(long, default_value = "text")]
        text_field: String,
    },
    /// Filter JSONL into prose + skipped files
    Filter {
        /// Input JSONL file (supports .gz)
        input: PathBuf,
        /// Output file for prose docs
        #[arg(long)]
        prose: PathBuf,
        /// Output file for skipped docs
        #[arg(long)]
        skipped: PathBuf,
        /// Comma-separated fields to classify
        #[arg(long, default_value = "bodytext,summarytext,title")]
        text_fields: String,
        /// Minimum confidence to accept classification
        #[arg(long, default_value = "0.6")]
        min_confidence: f32,
    },
    /// Dump structural features as CSV
    Features {
        /// Input JSONL file
        input: PathBuf,
        /// Output CSV file
        #[arg(short, long)]
        output: PathBuf,
        /// Field to extract features from
        #[arg(long, default_value = "text")]
        text_field: String,
    },
    /// Generate labels from Tier 1 for model training
    LabelCorpus {
        /// Input JSONL file (supports .gz)
        input: PathBuf,
        /// Output labeled JSONL file
        #[arg(short, long)]
        output: PathBuf,
        /// Field to classify
        #[arg(long, default_value = "text")]
        text_field: String,
    },
    /// Train a fasttext model from labeled data
    Train {
        /// Input labeled JSONL file
        #[arg(long)]
        input: PathBuf,
        /// Output model file
        #[arg(long)]
        output: PathBuf,
    },
    /// Validate classifier against labeled data
    Validate {
        /// Input labeled JSONL file
        #[arg(long)]
        input: PathBuf,
        /// Field containing text to classify
        #[arg(long, default_value = "text")]
        text_field: String,
        /// Output results as JSON
        #[arg(long)]
        json: bool,
    },
}

struct Evaluator {
    predictions: Vec<(String, String)>,
}

impl Evaluator {
    fn new() -> Self {
        Evaluator {
            predictions: Vec::new(),
        }
    }

    fn add(&mut self, predicted: &str, actual: &str) {
        self.predictions
            .push((predicted.to_string(), actual.to_string()));
    }

    fn total(&self) -> usize {
        self.predictions.len()
    }

    fn accuracy(&self) -> f64 {
        if self.predictions.is_empty() {
            return 0.0;
        }
        let correct = self
            .predictions
            .iter()
            .filter(|(p, a)| p == a)
            .count();
        correct as f64 / self.predictions.len() as f64
    }

    fn categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self
            .predictions
            .iter()
            .map(|(_, a)| a.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        cats.sort();
        cats
    }

    fn precision_recall_f1(&self, category: &str) -> (f64, f64, f64, usize) {
        let true_positives = self
            .predictions
            .iter()
            .filter(|(p, a)| p == category && a == category)
            .count();
        let predicted_positive = self
            .predictions
            .iter()
            .filter(|(p, _)| p == category)
            .count();
        let actual_positive = self
            .predictions
            .iter()
            .filter(|(_, a)| a == category)
            .count();

        let precision = if predicted_positive > 0 {
            true_positives as f64 / predicted_positive as f64
        } else {
            0.0
        };
        let recall = if actual_positive > 0 {
            true_positives as f64 / actual_positive as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        (precision, recall, f1, actual_positive)
    }

    fn confusion_matrix(&self) -> (Vec<String>, Vec<Vec<usize>>) {
        let labels = self.categories();
        let n = labels.len();
        let mut matrix = vec![vec![0usize; n]; n];

        for (predicted, actual) in &self.predictions {
            let actual_idx = labels.iter().position(|l| l == actual).unwrap();
            let predicted_idx = labels.iter().position(|l| l == predicted).unwrap_or(n);
            if predicted_idx < n {
                matrix[actual_idx][predicted_idx] += 1;
            }
        }

        (labels, matrix)
    }

    fn print_report(&self) {
        eprintln!("── Validation Summary ──");
        eprintln!("  Total samples:     {}", self.total());
        eprintln!("  Overall accuracy:  {:.3}", self.accuracy());
        eprintln!();
        eprintln!("── Per-Category ──────────────────────────────────");
        eprintln!(
            "{:<13}{:<11}{:<8}{:<7}N",
            "Category", "Precision", "Recall", "F1"
        );
        for cat in &self.categories() {
            let (prec, recall, f1, count) = self.precision_recall_f1(cat);
            eprintln!(
                "{:<13}{:<11.2}{:<8.2}{:<7.2}{}",
                cat, prec, recall, f1, count
            );
        }
        eprintln!();

        let (labels, matrix) = self.confusion_matrix();
        eprintln!("── Confusion Matrix ──────────────────────────────");
        // Header: abbreviated labels (first 3 chars, capitalized)
        let abbrevs: Vec<String> = labels
            .iter()
            .map(|l| {
                let mut chars = l.chars();
                let first = chars.next().unwrap_or(' ').to_uppercase().to_string();
                let rest: String = chars.take(2).collect();
                format!("{first}{rest}")
            })
            .collect();
        eprint!("{:<13}", "Predicted →");
        for abbr in &abbrevs {
            eprint!("{:<6}", abbr);
        }
        eprintln!();
        for (i, label) in labels.iter().enumerate() {
            // Capitalize first letter of label
            let display: String = {
                let mut chars = label.chars();
                match chars.next() {
                    Some(c) => c.to_uppercase().to_string() + chars.as_str(),
                    None => String::new(),
                }
            };
            eprint!("{:<13}", display);
            for cell in &matrix[i] {
                eprint!("{cell:<6}");
            }
            eprintln!();
        }
    }

    fn to_json(&self) -> serde_json::Value {
        let mut per_category = Vec::new();
        for cat in &self.categories() {
            let (prec, recall, f1, count) = self.precision_recall_f1(cat);
            per_category.push(serde_json::json!({
                "category": cat,
                "precision": prec,
                "recall": recall,
                "f1": f1,
                "count": count,
            }));
        }

        let (labels, matrix) = self.confusion_matrix();

        serde_json::json!({
            "total": self.total(),
            "accuracy": self.accuracy(),
            "per_category": per_category,
            "confusion_matrix": {
                "labels": labels,
                "matrix": matrix,
            },
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let classifier = match &cli.model {
        Some(path) => Classifier::with_model(path.to_str().ok_or("Invalid model path")?)?,
        None => Classifier::new(),
    };

    match cli.command {
        None => {
            // Default: read from stdin, classify each line
            classify_stdin(&classifier)?;
        }
        Some(Commands::File {
            input,
            output,
            text_field,
        }) => {
            classify_file(&classifier, &input, &output, &text_field)?;
        }
        Some(Commands::Filter {
            input,
            prose,
            skipped,
            text_fields,
            min_confidence,
        }) => {
            let fields: Vec<&str> = text_fields.split(',').map(|s| s.trim()).collect();
            filter_file(
                &classifier,
                &input,
                &prose,
                &skipped,
                &fields,
                min_confidence,
            )?;
        }
        Some(Commands::Features {
            input,
            output,
            text_field,
        }) => {
            extract_features_file(&classifier, &input, &output, &text_field)?;
        }
        Some(Commands::LabelCorpus {
            input,
            output,
            text_field,
        }) => {
            label_corpus(&input, &output, &text_field)?;
        }
        Some(Commands::Train { input, output }) => {
            eprintln!("Training not yet implemented — use fasttext CLI directly:");
            eprintln!("  fasttext supervised -input {input:?} -output {output:?}");
            std::process::exit(1);
        }
        Some(Commands::Validate {
            input,
            text_field,
            json,
        }) => {
            validate(&classifier, &input, &text_field, json)?;
        }
    }

    Ok(())
}

fn classify_stdin(classifier: &Classifier) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::new();
    io::stdin().lock().read_to_string(&mut text)?;

    if text.trim().is_empty() {
        return Ok(());
    }

    let result = classifier.classify(&text);
    let stdout = io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer(&mut out, &result)?;
    writeln!(out)?;
    Ok(())
}

fn open_reader(path: &PathBuf) -> Result<Box<dyn BufRead>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    if path.extension().is_some_and(|ext| ext == "gz") {
        let decoder = flate2::read::GzDecoder::new(file);
        Ok(Box::new(io::BufReader::new(decoder)))
    } else {
        Ok(Box::new(io::BufReader::new(file)))
    }
}

fn resolve_field<'a>(doc: &'a serde_json::Value, field: &str) -> Option<&'a str> {
    let mut current = doc;
    for part in field.split('.') {
        current = current.get(part)?;
    }
    current.as_str()
}

fn classify_file(
    classifier: &Classifier,
    input: &PathBuf,
    output: &PathBuf,
    text_field: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = open_reader(input)?;
    let mut out = io::BufWriter::new(std::fs::File::create(output)?);
    let mut count = 0;
    let mut missing_field = 0u64;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let mut doc: serde_json::Value = serde_json::from_str(&line)?;
        if let Some(text) = resolve_field(&doc, text_field) {
            let result = classifier.classify(text);
            doc["_classification"] = serde_json::json!({
                "category": result.category,
                "sub_type": result.sub_type,
                "confidence": result.confidence,
                "reason": result.reason,
                "tier": result.tier,
            });
        } else {
            missing_field += 1;
        }

        serde_json::to_writer(&mut out, &doc)?;
        writeln!(out)?;
        count += 1;
    }

    eprintln!("Classified {count} documents → {output:?}");
    if missing_field > 0 {
        eprintln!("  Warning: {missing_field} documents missing field \"{text_field}\"");
    }
    Ok(())
}

fn filter_file(
    classifier: &Classifier,
    input: &PathBuf,
    prose_path: &PathBuf,
    skipped_path: &PathBuf,
    text_fields: &[&str],
    min_confidence: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = open_reader(input)?;
    let mut prose_out = io::BufWriter::new(std::fs::File::create(prose_path)?);
    let mut skip_out = io::BufWriter::new(std::fs::File::create(skipped_path)?);

    let mut total = 0u64;
    let mut prose_count = 0u64;
    let mut skip_count = 0u64;
    let mut category_counts: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let doc: serde_json::Value = serde_json::from_str(&line)?;
        total += 1;

        // Classify each field independently
        let mut any_prose = false;
        let mut field_classifications = serde_json::Map::new();

        for &field_name in text_fields {
            if let Some(text) = resolve_field(&doc, field_name) {
                // Short fields (<50 bytes) auto-classified as prose
                if text.len() < 50 {
                    any_prose = true;
                    *category_counts
                        .entry(TextCategory::Prose.to_string())
                        .or_insert(0) += 1;
                    field_classifications.insert(
                        field_name.to_string(),
                        serde_json::json!({"category": "prose", "sub_type": null, "confidence": 1.0, "reason": "short field"}),
                    );
                    continue;
                }

                let result = classifier.classify(text);
                let is_prose =
                    result.category == TextCategory::Prose || (result.confidence < min_confidence); // uncertain → let through

                if is_prose {
                    any_prose = true;
                }

                *category_counts
                    .entry(result.category.to_string())
                    .or_insert(0) += 1;

                field_classifications.insert(
                    field_name.to_string(),
                    serde_json::json!({
                        "category": result.category,
                        "sub_type": result.sub_type,
                        "confidence": result.confidence,
                        "reason": result.reason,
                        "tier": result.tier,
                    }),
                );
            }
        }

        // Attach classification metadata
        let mut output_doc = doc.clone();
        output_doc["_field_classifications"] = serde_json::Value::Object(field_classifications);

        if any_prose {
            serde_json::to_writer(&mut prose_out, &output_doc)?;
            writeln!(prose_out)?;
            prose_count += 1;
        } else {
            serde_json::to_writer(&mut skip_out, &output_doc)?;
            writeln!(skip_out)?;
            skip_count += 1;
        }
    }

    eprintln!("── Filter Summary ──");
    eprintln!("  Total docs:       {total}");
    eprintln!("  Prose:            {prose_count}");
    eprintln!("  Skipped:          {skip_count}");
    eprintln!("  Categories:");
    let mut sorted: Vec<_> = category_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (cat, count) in sorted {
        eprintln!("    {cat:<15} {count}");
    }
    eprintln!("  → {prose_path:?}");
    eprintln!("  → {skipped_path:?}");
    Ok(())
}

fn extract_features_file(
    classifier: &Classifier,
    input: &PathBuf,
    output: &PathBuf,
    text_field: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = open_reader(input)?;
    let mut out = io::BufWriter::new(std::fs::File::create(output)?);

    // CSV header
    writeln!(
        out,
        "line_length_cv,char_entropy,leading_whitespace_ratio,tab_density,sentence_punctuation_rate,paragraph_break_rate,alpha_ratio,line_uniqueness,short_line_ratio,symbol_ratio,delimiter_consistency,json_brace_depth,key_value_ratio,xml_tag_ratio,log_line_ratio,comment_ratio,numeric_field_ratio,repetitive_structure_score,line_count"
    )?;

    let mut count = 0;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let doc: serde_json::Value = serde_json::from_str(&line)?;
        if let Some(text) = resolve_field(&doc, text_field) {
            let f = classifier.extract_features(text);
            writeln!(
                out,
                "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
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
                f.line_count
            )?;
            count += 1;
        }
    }

    eprintln!("Extracted features for {count} documents → {output:?}");
    Ok(())
}

fn label_corpus(
    input: &PathBuf,
    output: &PathBuf,
    text_field: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = open_reader(input)?;
    let mut out = io::BufWriter::new(std::fs::File::create(output)?);

    let mut count = 0u64;
    let mut ambiguous = 0u64;
    let mut missing_field = 0u64;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let doc: serde_json::Value = serde_json::from_str(&line)?;
        if let Some(text) = resolve_field(&doc, text_field) {
            // Always use Tier 1 only — prevents circular labeling when a
            // model is loaded (the model's own predictions must never become
            // its training labels).
            let result = classify(text);
            if result.confidence < MIN_CONFIDENCE {
                ambiguous += 1;
            }

            let labeled = serde_json::json!({
                "text": text,
                "label": result.category.to_string(),
                "sub_type": result.sub_type.map(|s| s.to_string()),
                "confidence": result.confidence,
                "tier": result.tier.to_string(),
                "reason": result.reason,
            });

            serde_json::to_writer(&mut out, &labeled)?;
            writeln!(out)?;
            count += 1;
        } else {
            missing_field += 1;
        }
    }

    eprintln!("Labeled {count} documents → {output:?}");
    eprintln!("  Ambiguous (confidence < 0.7): {ambiguous} — review these manually");
    if missing_field > 0 {
        eprintln!("  Warning: {missing_field} documents missing field \"{text_field}\"");
    }
    Ok(())
}

fn validate(
    classifier: &Classifier,
    input: &PathBuf,
    text_field: &str,
    json_output: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = open_reader(input)?;
    let mut evaluator = Evaluator::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let doc: serde_json::Value = serde_json::from_str(&line)?;
        let label = doc["label"]
            .as_str()
            .ok_or("Missing or non-string 'label' field")?;

        if let Some(text) = resolve_field(&doc, text_field) {
            let result = classifier.classify(text);
            evaluator.add(&result.category.to_string(), label);
        }
    }

    if json_output {
        let json = evaluator.to_json();
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        evaluator.print_report();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluator_new_is_empty() {
        let eval = Evaluator::new();
        assert_eq!(eval.total(), 0);
    }

    #[test]
    fn evaluator_add_increments_total() {
        let mut eval = Evaluator::new();
        eval.add("prose", "prose");
        eval.add("code", "prose");
        assert_eq!(eval.total(), 2);
    }

    #[test]
    fn evaluator_accuracy_all_correct() {
        let mut eval = Evaluator::new();
        eval.add("prose", "prose");
        eval.add("code", "code");
        eval.add("tabular", "tabular");
        assert!((eval.accuracy() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluator_accuracy_none_correct() {
        let mut eval = Evaluator::new();
        eval.add("code", "prose");
        eval.add("prose", "code");
        assert!((eval.accuracy() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluator_accuracy_partial() {
        let mut eval = Evaluator::new();
        eval.add("prose", "prose");
        eval.add("code", "prose");
        eval.add("code", "code");
        eval.add("prose", "code");
        assert!((eval.accuracy() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluator_accuracy_empty_returns_zero() {
        let eval = Evaluator::new();
        assert!((eval.accuracy() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluator_categories_sorted() {
        let mut eval = Evaluator::new();
        eval.add("prose", "code");
        eval.add("code", "prose");
        eval.add("tabular", "tabular");
        let cats = eval.categories();
        assert_eq!(cats, vec!["code", "prose", "tabular"]);
    }

    #[test]
    fn evaluator_precision_recall_f1() {
        let mut eval = Evaluator::new();
        // 3 actual prose, 2 predicted correctly, 1 mislabeled as code
        eval.add("prose", "prose");
        eval.add("prose", "prose");
        eval.add("code", "prose");
        // 2 actual code, 1 predicted correctly, 1 mislabeled as prose
        eval.add("code", "code");
        eval.add("prose", "code");

        let (prec, recall, f1, count) = eval.precision_recall_f1("prose");
        // Predicted prose: 3 (2 correct + 1 was actually code) → precision = 2/3
        assert!((prec - 2.0 / 3.0).abs() < 1e-9);
        // Actual prose: 3, correctly predicted 2 → recall = 2/3
        assert!((recall - 2.0 / 3.0).abs() < 1e-9);
        // F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3
        assert!((f1 - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(count, 3); // actual count for prose
    }

    #[test]
    fn evaluator_precision_recall_f1_no_predictions() {
        let mut eval = Evaluator::new();
        eval.add("code", "prose");
        // No predictions for "prose" category — precision 0, recall 0
        // Wait, "code" was predicted, "prose" was actual.
        // precision_recall_f1("prose"): predicted prose = 0, actual prose = 1
        // precision = 0/0 = 0 (no predictions of prose)
        // recall = 0/1 = 0 (none of actual prose predicted correctly)
        let (prec, recall, f1, count) = eval.precision_recall_f1("prose");
        assert!((prec - 0.0).abs() < f64::EPSILON);
        assert!((recall - 0.0).abs() < f64::EPSILON);
        assert!((f1 - 0.0).abs() < f64::EPSILON);
        assert_eq!(count, 1);
    }

    #[test]
    fn evaluator_confusion_matrix() {
        let mut eval = Evaluator::new();
        eval.add("prose", "prose");
        eval.add("prose", "prose");
        eval.add("code", "prose");
        eval.add("code", "code");
        eval.add("prose", "code");

        let (labels, matrix) = eval.confusion_matrix();
        assert_eq!(labels, vec!["code", "prose"]);
        // matrix[actual_idx][predicted_idx]
        // actual=code (idx 0): predicted code=1, predicted prose=1
        assert_eq!(matrix[0], vec![1, 1]);
        // actual=prose (idx 1): predicted code=1, predicted prose=2
        assert_eq!(matrix[1], vec![1, 2]);
    }

    #[test]
    fn evaluator_to_json_structure() {
        let mut eval = Evaluator::new();
        eval.add("prose", "prose");
        eval.add("code", "code");

        let json = eval.to_json();
        assert_eq!(json["total"], 2);
        assert!((json["accuracy"].as_f64().unwrap() - 1.0).abs() < f64::EPSILON);
        assert!(json["per_category"].is_array());
        assert!(json["confusion_matrix"].is_object());
    }
}
