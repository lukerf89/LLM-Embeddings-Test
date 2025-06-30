# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a simple Python project for testing and comparing embeddings from different Hugging Face Transformer models. The main script demonstrates how to generate embeddings using various pre-trained models and calculate semantic similarity between sentences.

## Dependencies

- `transformers` - Hugging Face Transformers library
- `torch` - PyTorch for tensor operations

Install with: `python3 -m pip install transformers torch`

## Running the Code

Execute the main script:
```bash
python3 embeddings.py
```

This will:
1. Load three different models (BERT, sentence-transformers, RoBERTa)
2. Generate embeddings for sample sentences
3. Calculate and display cosine similarity between embeddings
4. Show embedding dimensions for each model

## Architecture

The codebase consists of a single file with:
- `get_embeddings()` function that handles model loading, tokenization, and embedding generation using mean pooling
- Test sentences comparing semantically similar phrases
- Model comparison loop that tests BERT-base, sentence-transformers/all-MiniLM-L6-v2, and RoBERTa-base