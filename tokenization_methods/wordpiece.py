from .tokenization_method import TokenizationMethod
from typing import List

space_char = " "

class WordPiece(TokenizationMethod):
    __prefix_char_inside_word = "##"

    def create_vocabulary(self, corpus: List[str], vocab_size: int) -> List[tuple]:
        # The vocab (the keys are the tokens, the values are the sequences)
        vocab = []
        
        # Preprocess:
        # 1. Convert the corpus into a list of lists
        # 2. Create a new token for each individual character in the corpus NOTE: An individual character at the beginning
        # of a word does not have the same token as that character in the middle (or end) of the word.
        # 3. Convert all the characters into tokens
        tokenized_corpus: List[list] = []
        for sentence in corpus:
            tokenized_corpus.append([])
            was_last_character_space = True

            for c in sentence:
                # Check if the character is the beginning of a word (leave as it is), or is inside the word (add the prefix)
                is_space = c == space_char
                c = (c) if (was_last_character_space or is_space) else (self.__prefix_char_inside_word+c)
                tk_list = [tk for (tk,seq) in vocab if seq == c]
                if any(tk_list):
                    assert len(tk_list) == 1
                    # Use a token already created
                    tk = tk_list[0]
                else:
                    # Create a new token
                    tk = self.__create_token(vocab, c, c)
                # Convert the character to a token
                tokenized_corpus[-1].append(tk)

                was_last_character_space = is_space
        
        # Iterate until the vocabulary reaches the maximum length
        while len(vocab) < vocab_size:
            pair = self.__get_highests_scoring_pair(tokenized_corpus)
            if pair is None:
                break
            token = self.__create_token(vocab, pair)
            self.__merge_tokens(tokenized_corpus, pair, token)

        return vocab
    
    def __create_token(self, current_vocab, sequence_of_tokens, new_token=None):
        """Given a sequence creates a new token and adds it to the vocabulary"""
        if new_token is None:
            assert len(sequence_of_tokens) == 2
            tk1,tk2 = sequence_of_tokens
            new_token = tk1 + tk2.lstrip(self.__prefix_char_inside_word)
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
        
    def __get_highests_scoring_pair(self, corpus):
        """Returns the pair of tokens that has the highest score"""
        pairs, tokens = {}, {}
        for sentence in corpus:
            if len(sentence) <= 1:
                continue
            for t1,t2 in zip(sentence, sentence[1:]):
                pairs[(t1,t2)] = pairs.get((t1,t2), 0) + 1
                tokens[t1] = tokens.get(t1, 0) + 1
            tokens[t2] = tokens.get(t2, 1) + 1

        if pairs:
            return max(pairs, key = lambda pair: pairs[pair] / (tokens[pair[0]] * tokens[pair[1]]))
        else:
            return None
    
    def tokenize_text(self, vocabulary: List[tuple], text: str):
        tokenized_text = []

        # Replace all individual characters with tokens
        is_last_space = True
        for c in text:
            is_space = c == space_char
            tokenized_text.append(
                next(
                    tk
                    for (tk,seq) in vocabulary
                    if seq==(c if (is_space or is_last_space) else self.__prefix_char_inside_word+c)
                )
            )
            is_last_space = is_space

        # Make it have the shape of a corpus
        tokenized_text = [tokenized_text]
        
        # Merge tokens according to the vocab
        for token,sequence in vocabulary:
            self.__merge_tokens(tokenized_text, sequence, token)
        
        return tokenized_text[0]
