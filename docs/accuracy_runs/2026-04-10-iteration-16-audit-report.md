# Iter16 Semantic Label Audit Report

**Date:** 2026-04-10
**Models:** gemini3flash, sonnet, gpt54mini

## Pass Criteria

1. Inter-annotator agreement >=0.995 on stratified 5k for ALL labels.
2. Recall (majority >=2 of 3) >=0.9 on injected positives.
3. Zero obvious rule violations in spot-check (manual).

## Criterion 1: Inter-Annotator Agreement (stratified 5k)

| Label | Agreement | Pass |
|---|---:|:---:|
| c_cpp | 0.9940 | FAIL |
| csharp | 0.9998 | PASS |
| css | 0.9973 | PASS |
| csv | 0.9994 | PASS |
| diff_patch | 0.9999 | PASS |
| dockerfile | 0.9999 | PASS |
| fixed_width | 0.9922 | FAIL |
| go | 0.9994 | PASS |
| graphql | 0.9998 | PASS |
| html | 0.9910 | FAIL |
| ini | 0.9918 | FAIL |
| java | 0.9987 | PASS |
| javascript | 0.9923 | FAIL |
| json | 0.9947 | FAIL |
| jsonl | 0.9981 | PASS |
| key_value | 0.9775 | FAIL |
| kotlin | 0.9998 | PASS |
| latex | 0.9922 | FAIL |
| log_content | 0.9986 | PASS |
| lua | 0.9997 | PASS |
| makefile | 0.9965 | PASS |
| markdown | 0.9507 | FAIL |
| objc | 0.9999 | PASS |
| php | 0.9995 | PASS |
| pipe_table | 0.9971 | PASS |
| plain | 0.9391 | FAIL |
| powershell | 0.9997 | PASS |
| python | 0.9981 | PASS |
| r | 0.9998 | PASS |
| rst | 0.9975 | PASS |
| ruby | 0.9999 | PASS |
| rust | 0.9999 | PASS |
| sgml | 0.9976 | PASS |
| shell | 0.9850 | FAIL |
| sql | 0.9990 | PASS |
| stack_trace | 0.9997 | PASS |
| swift | 0.9999 | PASS |
| toml | 0.9987 | PASS |
| tsv | 0.9988 | PASS |
| typescript | 0.9978 | PASS |
| xml | 0.9975 | PASS |
| yaml | 0.9949 | FAIL |

**Criterion 1 verdict: FAIL**

## Criterion 2: Recall on Injected Positives

| Label | Recall (majority) | Pass |
|---|---:|:---:|
| diff_patch | 1.0000 | PASS |
| log_content | 0.8400 | FAIL |
| stack_trace | 0.6250 | FAIL |

**Criterion 2 verdict: FAIL**

## Criterion 3: Spot-Check Rule Violations

_Manual review required -- scan disagreement rows and injected positives for obvious rule violations (e.g., log_content firing on lowercase error in prose)._

## Overall Decision

**Gate: FAIL.** Iterate on label definitions; re-run audit.
