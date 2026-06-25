# llm-translation-eng-dialect
## AIM
Build a small Machine Translation (MT) system for a low-resource language. The team picks one language they have access to (an Italian regional language like Sardinian or Neapolitan, a North-African dialect like Tunisian Arabic, a small Berber language like Tamazight, or any minority language they know personally). The team curates a tiny parallel corpus, fine-tunes a massively multilingual MT model (NLLB-200), and compares against the same model used zero-shot. Reports BLEU/chrF plus an honest writeup of what is hard about low-resource MT.


## Dataset pipeline

Run the full pipeline with:

```bash
python scripts/main.py
```

This will:

1. generate the FLORES dataset in `data/`
2. normalize `data/vec_sentences.jsonl` into the standard record file in `data/normalized/`
3. convert `data/raw/wikipedia_candidates.jsonl` into standard dataset records in `data/wikipedia_candidates.jsonl`
4. rebuild `data/full_dataset.jsonl` from those standardized records
5. create normalized datasets in `data/normalized/`
6. create processed `train/dev/test` splits in `data/processed/`
7. export normalized `train/dev/test` files in `data/normalized/`

## Normalization

The normalization profile lives in [config/venetian_normalization.json](<your_path>/llm-translation-eng-dialect/config/venetian_normalization.json).

It currently performs conservative normalization:

- orthographic cleanup such as Unicode normalization, apostrophe cleanup, `ł -> l`, `xè -> xe`
- light lexical canonicalization such as `voialtri/valtri/vialtri -> vualtri`
- metadata cleanup such as `coversation -> conversation`
- prompt standardization so every record uses `translate English to Venetian: ...`

To normalize a single dataset manually:

```bash
python scripts/normalize_dataset.py \
  --input-file data/vec_sentences.jsonl \
  --output-file data/normalized/vec_sentences.jsonl \
  --report-file data/normalized/vec_sentences.report.json
```

## Data Provenance

The dataset sources are split by role:

- `train`: manual translations collected in [data/full_dataset.jsonl](/llm-translation-eng-dialect/data/full_dataset.jsonl)
- `dev`: sampled from FLORES, generated through [scripts/dataset_flores.py](/llm-translation-eng-dialect/scripts/dataset_flores.py)
- `test`: sampled from FLORES, generated through [scripts/dataset_flores.py](/llm-translation-eng-dialect/scripts/dataset_flores.py)

In practice:

- the manual translation side is the curated training source
- `data/normalized/vec_sentences.jsonl` is the standardized manual-record source used to rebuild `full_dataset`
- `data/wikipedia_candidates.jsonl` is the standardized Wikipedia-candidate source used to extend `full_dataset`
- the FLORES English to Venetian dataset is the evaluation pool
- `dev` and `test` are created from that FLORES pool during the processing pipeline

Relevant files:

- [data/full_dataset.jsonl](llm-translation-eng-dialect/data/full_dataset.jsonl): merged manual-translation training source
- [data/eng_Latn_vec_Latn_dataset.jsonl](llm-translation-eng-dialect/data/eng_Latn_vec_Latn_dataset.jsonl): FLORES-derived dataset
- [data/processed/train.jsonl](llm-translation-eng-dialect/data/processed/train.jsonl): final train split
- [data/processed/dev.jsonl](llm-translation-eng-dialect/data/processed/dev.jsonl): final dev split from FLORES
- [data/processed/test.jsonl](llm-translation-eng-dialect/data/processed/test.jsonl): final test split from FLORES
- [data/normalized/train.jsonl](llm-translation-eng-dialect/data/normalized/train.jsonl): normalized train split
- [data/normalized/dev.jsonl](llm-translation-eng-dialect/data/normalized/dev.jsonl): normalized dev split
- [data/normalized/test.jsonl](llm-translation-eng-dialect/data/normalized/test.jsonl): normalized test split

# STRUCTURE

- `data/full_dataset.jsonl`: manual-translation dataset used as the training source.
- `data/eng_Latn_vec_Latn_dataset.jsonl`: FLORES-derived dataset used as the evaluation source.
- `data/normalized/`: normalized versions of the raw datasets.
- `data/normalized/train.jsonl`, `dev.jsonl`, `test.jsonl`: normalized split files.
- `data/processed/`: final train/dev/test splits and summary files.
- `scripts/dataset_flores.py`: generates the FLORES-based English to Venetian dataset.
- `scripts/rebuild_full_dataset.py`: rebuilds the merged manual dataset.
- `scripts/normalize_dataset.py`: applies normalization rules.
- `scripts/import_dataset.py`: creates processed splits.
- `scripts/main.py`: runs the whole pipeline.
- `config/venetian_normalization.json`: normalization rules for orthography, lexical variants, and metadata.
- `src/dataset.py`: dataset loading, writing, splitting, and renumbering helpers.
- `src/normalization.py`: implementation of the Venetian normalizer.

- `scripts/demo_app.py`: Streamlit demo for aggregate evaluation, sentence-level comparison, and interactive English-to-Venetian translation.

## Interactive demo

The interactive demo is implemented in `scripts/demo_app.py` with Streamlit. It combines precomputed evaluation results with runtime translation of user-provided English sentences.

The evaluation compares three systems:

- NLLB zero-shot
- LLM zero-shot
- Fine-tuned NLLB

The current dataset contains:

- 653 training examples
- 82 development examples
- 82 test examples

The demo expects these files:

- `data/normalized/train.jsonl`
- `data/normalized/dev.jsonl`
- `data/normalized/test.jsonl`
- `data/results/nllb_zeroshot_predictions.jsonl`
- `data/results/llm_zeroshot_predictions.jsonl`
- `data/results/ablation_A0_predictions.jsonl`
- `data/results/evaluation_results.json`

The current aggregate test results are:

| System | BLEU | chrF | Human rating (1–5) |
|---|---:|---:|---:|
| NLLB zero-shot | 15.89 | 49.97 | 3.43 |
| LLM zero-shot | 16.43 | 49.74 | 3.61 |
| Fine-tuned NLLB | 16.07 | 51.50 | 3.57 |

Each of the 82 test outputs for the three principal systems has a saved manual `human_score` on a 1–5 scale. The `human_rating` values above are the corresponding means.

Relative to NLLB zero-shot, the fine-tuned NLLB model improves by 0.18 BLEU, 1.53 chrF, and 0.14 human-rating points. The LLM zero-shot baseline obtains the highest BLEU and mean human rating, while the fine-tuned NLLB model obtains the highest chrF.

Install the project dependencies, including Streamlit, with:

```bash
python -m pip install -r requirements.txt
```

Run the demo from the repository root with:

```bash
python -m streamlit run scripts/demo_app.py
```

Streamlit will open the application in the browser. By default, it is available at:

```text
http://localhost:8501
```

The demo displays:

1. the train, development, and test split sizes
2. aggregate BLEU, chrF, and mean human ratings for all three evaluated systems
3. sentence-level comparisons against the Venetian reference, including each saved human score
4. a fixed presentation example, the largest fine-tuning improvements and regressions, and all test examples
5. an interactive text field for translating new English sentences into Venetian
6. the main experimental takeaway

The aggregate metrics, mean human ratings, and sentence-level comparisons use the saved prediction files. Free-text translation performs inference at runtime. The human ratings are descriptive summaries of the saved 1–5 scores.

The public NLLB zero-shot model is downloaded automatically from Hugging Face when first used. To use the fine-tuned NLLB system, provide either:

- a local checkpoint path, such as `models/nllb_finetuned_updated`
- a compatible Hugging Face model ID

The local fine-tuned checkpoint directory must contain the model configuration, tokenizer files, and model weights, for example:

```text
models/nllb_finetuned_updated/
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer_config.json
└── tokenizer.json
```

Model checkpoints are not stored in the standard Git repository because of their size. The `models/` directory is excluded through `.gitignore`. A shared Drive or Hugging Face location can be used to distribute the fine-tuned checkpoint.
