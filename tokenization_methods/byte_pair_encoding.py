from typing import List
from .tokenization_method import TokenizationMethod

class BytePairEncoding(TokenizationMethod):
    __space_token = "_"
    __space_char = " "

    def create_vocabulary(self, corpus: List[str], vocab_size: int) -> List[tuple]:
        # The vocab (the keys are the tokens, the values are the sequences)
        vocab = []

        # Replace the whitespace with some other character
        for i,sentence in enumerate(corpus):
            corpus[i] = sentence.replace(BytePairEncoding.__space_char, BytePairEncoding.__space_token)
        
        # Preprocess:
        # 1. Convert the corpus into a list of lists
        # 2. Create a new token for each individual character in the corpus
        # 3. Convert all the characters into tokens
        tokenized_corpus: List[list] = []
        for sentence in corpus:
            tokenized_corpus.append([])
            for c in sentence:
                tk_list = [tk for (tk,seq) in vocab if seq == c]
                if any(tk_list):
                    assert len(tk_list) == 1
                    tk = tk_list[0]
                else:
                    tk = self.__create_token(vocab, c, c)
                tokenized_corpus[-1].append(tk)
        
        del corpus

        # Iterate until the vocabulary reaches the maximum length
        while len(vocab) < vocab_size:
            pair = self.__get_most_frequent_pair(tokenized_corpus)
            if pair is None:
                break
            token = self.__create_token(vocab, pair)
            self.__merge_tokens(tokenized_corpus, pair, token)

        return vocab
    
    def __create_token(self, current_vocab, sequence_of_tokens, new_token=None):
        """Given a sequence creates a new token and adds it to the vocabulary"""
        if new_token is None:
            new_token = len(current_vocab)
        current_vocab.append((new_token, sequence_of_tokens))
        return new_token
    
    def __merge_tokens(self, list_of_tokenized_sentences: List[list], sequence_of_tokens, replacement: int):
        """Merges a sequence into a token"""
        replacement = [replacement]
        sequence_of_tokens = list(sequence_of_tokens)
        sequence_length = len(sequence_of_tokens)

        for sentence in list_of_tokenized_sentences:
            i = 0
            while i <= (len(sentence) - sequence_length):
                if sentence[i : i+sequence_length] == sequence_of_tokens:
                    sentence[i : i+sequence_length] = replacement
                    i += 1
                else:
                    i += 1

    def __get_most_frequent_pair(self, corpus: List[list]):
        pairs = {}
        for sentence in corpus:
            for t1,t2 in zip(sentence, sentence[1:]):
                pairs[(t1,t2)] = pairs.get((t1,t2), 0) + 1

        if pairs:
            return max(pairs, key=pairs.get)
        else:   # There are no pairs
            return None

    def tokenize_text(self, vocabulary, text: str):
        tokenized_text = []

        # Replace spaces with its own token
        text = text.replace(BytePairEncoding.__space_char, BytePairEncoding.__space_token)

        # Replace all individual characters with tokens
        for c in text:
            tokenized_text.append(next(tk for (tk,seq) in vocabulary if seq==c))
            
        # Make it have the shape of a corpus
        tokenized_text = [tokenized_text]

        # Iterate over the text and replace a pair of tokens (or chars) with their token
        # Replace tokens in the same order as they were created
        for token,sequence in vocabulary:
            self.__merge_tokens(tokenized_text, sequence, token)
        
        return tokenized_text[0]
