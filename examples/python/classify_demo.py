"""Demonstrate text-classifier usage from Python."""

from text_classifier import Classifier


def main():
    clf = Classifier()

    # --- Single classification ---
    samples = {
        "prose": (
            "The quick brown fox jumps over the lazy dog. "
            "This sentence contains every letter of the English alphabet. "
            "It has been used for decades as a typing exercise."
        ),
        "code": """\
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

for i in range(10):
    print(fibonacci(i))
""",
        "tabular": (
            "Name\tAge\tCity\n"
            "Alice\t30\tNew York\n"
            "Bob\t25\tSan Francisco\n"
            "Carol\t35\tChicago\n"
            "Dave\t28\tSeattle\n"
            "Eve\t32\tBoston\n"
        ),
        "pdf_dump": "\n".join(
            [
                "A",
                "b",
                "s",
                "t",
                "r",
                "a",
                "c",
                "t",
                "",
                "T",
                "h",
                "i",
                "s",
                "",
                "p",
                "a",
                "p",
                "e",
                "r",
            ]
        ),
        "short": "Hi",
    }

    print("=== Single Classification ===\n")
    for label, text in samples.items():
        result = clf.classify(text)
        preview = text[:60].replace("\n", "\\n")
        print(f"  [{label}] {preview}...")
        print(f"    -> {result.text_type} (confidence={result.confidence:.2f}, tier={result.tier})")
        print(f"       reason: {result.reason}")
        print()

    # --- Batch classification ---
    print("=== Batch Classification ===\n")
    texts = list(samples.values())
    results = clf.classify_batch(texts)
    for label, result in zip(samples.keys(), results):
        print(f"  {label:<10} -> {result.text_type:<10} (confidence={result.confidence:.2f})")

    # --- Feature extraction ---
    print("\n=== Feature Extraction (prose sample) ===\n")
    features = clf.extract_features(samples["prose"])
    for name, value in features.items():
        print(f"  {name:<30} {value:.4f}")


if __name__ == "__main__":
    main()
