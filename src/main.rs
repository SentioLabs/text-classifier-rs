use std::io::{self, BufRead, Read, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use text_classifier::{Classifier, TextType};

#[derive(Parser)]
#[command(name = "classify", about = "Classify text for translation eligibility")]
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
    /// Filter JSONL into translatable + skipped files
    Filter {
        /// Input JSONL file (supports .gz)
        input: PathBuf,
        /// Output file for translatable docs
        #[arg(long)]
        translatable: PathBuf,
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
    /// Validate model against labeled data
    Validate {
        /// Input labeled JSONL file
        #[arg(long)]
        input: PathBuf,
        /// Model file to validate
        #[arg(long)]
        model: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let classifier = match &cli.model {
        Some(path) => Classifier::with_model(
            path.to_str().ok_or("Invalid model path")?,
        )?,
        None => Classifier::new(),
    };

    match cli.command {
        None => {
            // Default: read from stdin, classify each line
            classify_stdin(&classifier)?;
        }
        Some(Commands::File { input, output, text_field }) => {
            classify_file(&classifier, &input, &output, &text_field)?;
        }
        Some(Commands::Filter { input, translatable, skipped, text_fields, min_confidence }) => {
            let fields: Vec<&str> = text_fields.split(',').map(|s| s.trim()).collect();
            filter_file(&classifier, &input, &translatable, &skipped, &fields, min_confidence)?;
        }
        Some(Commands::Features { input, output, text_field }) => {
            extract_features_file(&classifier, &input, &output, &text_field)?;
        }
        Some(Commands::LabelCorpus { input, output, text_field }) => {
            label_corpus(&classifier, &input, &output, &text_field)?;
        }
        Some(Commands::Train { input, output }) => {
            eprintln!("Training not yet implemented — use fasttext CLI directly:");
            eprintln!("  fasttext supervised -input {input:?} -output {output:?}");
            std::process::exit(1);
        }
        Some(Commands::Validate { input, model }) => {
            eprintln!("Validation not yet implemented");
            eprintln!("  input: {input:?}");
            eprintln!("  model: {model:?}");
            std::process::exit(1);
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

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let mut doc: serde_json::Value = serde_json::from_str(&line)?;
        if let Some(text) = resolve_field(&doc, text_field) {
            let result = classifier.classify(text);
            doc["_classification"] = serde_json::json!({
                "text_type": result.text_type,
                "confidence": result.confidence,
                "reason": result.reason,
                "tier": result.tier,
            });
        }

        serde_json::to_writer(&mut out, &doc)?;
        writeln!(out)?;
        count += 1;
    }

    eprintln!("Classified {count} documents → {output:?}");
    Ok(())
}

fn filter_file(
    classifier: &Classifier,
    input: &PathBuf,
    translatable_path: &PathBuf,
    skipped_path: &PathBuf,
    text_fields: &[&str],
    min_confidence: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = open_reader(input)?;
    let mut trans_out = io::BufWriter::new(std::fs::File::create(translatable_path)?);
    let mut skip_out = io::BufWriter::new(std::fs::File::create(skipped_path)?);

    let mut total = 0u64;
    let mut trans_count = 0u64;
    let mut skip_count = 0u64;
    let mut category_counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let doc: serde_json::Value = serde_json::from_str(&line)?;
        total += 1;

        // Classify each field independently
        let mut any_translatable = false;
        let mut field_classifications = serde_json::Map::new();

        for &field_name in text_fields {
            if let Some(text) = resolve_field(&doc, field_name) {
                // Short fields (<50 chars) auto-translatable
                if text.len() < 50 {
                    any_translatable = true;
                    field_classifications.insert(
                        field_name.to_string(),
                        serde_json::json!({"text_type": "translatable", "confidence": 1.0, "reason": "short field"}),
                    );
                    continue;
                }

                let result = classifier.classify(text);
                let is_trans = result.text_type == TextType::Translatable
                    || (result.confidence < min_confidence); // uncertain → let through

                if is_trans {
                    any_translatable = true;
                }

                *category_counts.entry(result.text_type.to_string()).or_insert(0) += 1;

                field_classifications.insert(
                    field_name.to_string(),
                    serde_json::json!({
                        "text_type": result.text_type,
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

        if any_translatable {
            serde_json::to_writer(&mut trans_out, &output_doc)?;
            writeln!(trans_out)?;
            trans_count += 1;
        } else {
            serde_json::to_writer(&mut skip_out, &output_doc)?;
            writeln!(skip_out)?;
            skip_count += 1;
        }
    }

    eprintln!("── Filter Summary ──");
    eprintln!("  Total docs:       {total}");
    eprintln!("  Translatable:     {trans_count}");
    eprintln!("  Skipped:          {skip_count}");
    eprintln!("  Categories:");
    let mut sorted: Vec<_> = category_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (cat, count) in sorted {
        eprintln!("    {cat:<15} {count}");
    }
    eprintln!("  → {translatable_path:?}");
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
    writeln!(out, "line_length_cv,char_entropy,leading_whitespace_ratio,tab_density,sentence_punctuation_rate,paragraph_break_rate,alpha_ratio,line_uniqueness,short_line_ratio,symbol_ratio")?;

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
                "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                f.line_length_cv, f.char_entropy, f.leading_whitespace_ratio,
                f.tab_density, f.sentence_punctuation_rate, f.paragraph_break_rate,
                f.alpha_ratio, f.line_uniqueness, f.short_line_ratio, f.symbol_ratio
            )?;
            count += 1;
        }
    }

    eprintln!("Extracted features for {count} documents → {output:?}");
    Ok(())
}

fn label_corpus(
    classifier: &Classifier,
    input: &PathBuf,
    output: &PathBuf,
    text_field: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = open_reader(input)?;
    let mut out = io::BufWriter::new(std::fs::File::create(output)?);

    let mut count = 0;
    let mut ambiguous = 0;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let doc: serde_json::Value = serde_json::from_str(&line)?;
        if let Some(text) = resolve_field(&doc, text_field) {
            let result = classifier.classify(text);
            if result.confidence < 0.7 {
                ambiguous += 1;
            }

            let labeled = serde_json::json!({
                "text": text,
                "label": result.text_type.to_string(),
                "confidence": result.confidence,
                "tier": result.tier.to_string(),
                "reason": result.reason,
            });

            serde_json::to_writer(&mut out, &labeled)?;
            writeln!(out)?;
            count += 1;
        }
    }

    eprintln!("Labeled {count} documents → {output:?}");
    eprintln!("  Ambiguous (confidence < 0.7): {ambiguous} — review these manually");
    Ok(())
}
