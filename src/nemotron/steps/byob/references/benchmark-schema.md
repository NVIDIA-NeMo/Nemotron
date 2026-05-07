# Benchmark Schema

## Final MCQ Parquet

The final MCQ benchmark must contain these columns:

| Column | Meaning |
| --- | --- |
| `question_id` | Stable unique ID for the generated question. |
| `question` | Question text. |
| `options` | Ordered option list. |
| `answer_index` | Zero-based index into `options`. |
| `answer` | Letter label for the correct answer. |
| `cot_content` | Chain-of-thought placeholder or content, `-` when unavailable. |
| `src` | Source marker, `-` when unavailable. |
| `category` | Target subject/category. |

## Source Documents

For MCQ generation, `input_dir/<target_subject>/` may contain `.txt` documents. Alternatively,
`input_dir/<target_subject>.parquet` may contain prepared source text. Target subject names must match
`target_source_mapping` keys.

## Translation

Translated MCQ outputs keep the same final schema. Translation cache files may contain additional
`question_translated`, `options_translated`, backtranslation, Curator metadata, and quality metric columns.
