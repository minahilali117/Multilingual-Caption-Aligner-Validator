# Multilingual Caption Aligner & Validator

Automatically translate English image captions into a user-specified target language (e.g., Urdu, Spanish, Chinese), and validate that the translations preserve semantic meaning using embedding-based similarity scoring.

## Features

- Translate captions from English to any target language using Hugging Face Transformers (`opus-mt` models)
- Validate semantic similarity with multilingual sentence embeddings via LaBSE
- Discard inaccurate translations based on cosine similarity threshold (default: 0.75)
- Output results in JSON or CSV format
- Designed for datasets with image-caption pairs

## Requirements

Install the required Python packages using:

```bash
pip install torch transformers pandas tqdm
