# Subword Tokenization

## Overview
This repository contains my own implementation of 3 subword tokenization methods (Byte Pair Encoding, Unigram Language Model, WordPiece) and results after training with data in Spanish.

## Structure
The implementations can be found in the folder `tokenization_methods`. Inside of it you can find the following files:
- `tokenization_method.py`: contains the abstract class the concrete classes must inherit from `TokenizationMethod`.
- `byte_pair_encoding.py`: contains class `BytePairEncoding` with the concrete implementation of the BPE method.
- `unigram_language_model.py`: contains class `UnigramLanguageModel` with the concrete implementation of the Unigram Language Model method.
- `wordpiece.py`: contains class `WordPiece` with the concrete implementation of the WordPiece method.

## Usage
To reproduce the experiment follow the steps in [train_and_tokenize.ipynb](train_and_tokenize.ipynb).
