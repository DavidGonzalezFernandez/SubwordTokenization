from typing import Dict, List, Union, Tuple
from .tokenization_method import TokenizationMethod

class BytePairEncoding(TokenizationMethod):
    def create_vocabulary(self, corpus: List[str], vocab_size: int) -> Dict[int, Tuple[Union[str,int],Union[str, int]]]:
        # The vocab (the keys are the tokens, the values are the sequences)
        vocab = {}
        
        # Create a token for whitespace
        space = " "
        space_token = self.__create_token(vocab, space)

        # Convert the corpus into a list of lists
        tokenized_corpus: List[List[Union[str, int]]] = []
        for sentence in corpus:
            tokenized_corpus.append([])
            for c in sentence:
                tokenized_corpus[-1].append(space_token if c==space else c)
        del corpus

        while len(vocab) < vocab_size:
            pair = self.__get_most_frequent_pair(tokenized_corpus)
            if pair is None:
                break
            token = self.__create_token(vocab, pair)
            self.__merge_tokens(tokenized_corpus, pair, token)

        return vocab
    
    def __create_token(self, current_vocab, sequence_of_tokens: Union[str, Tuple[Union[str,int],Union[str, int]]]):
        """Given a sequence creates a new token and adds it to the vocabulary"""
        new_token = len(current_vocab)
        current_vocab[new_token] = sequence_of_tokens
        return new_token
    
    def __merge_tokens(
        self,
        list_of_tokenized_sentences: List[List[Union[str, int]]],
        sequence_of_tokens: Tuple[Union[str,int],Union[str, int]],
        replacement: int
    ):
        """Merges a sequence into a token"""
        replacement = [replacement]
        sequence_of_tokens = list(sequence_of_tokens)
        sequence_length = len(sequence_of_tokens)
        assert sequence_length == 2

        for sentence in list_of_tokenized_sentences:
            i = 0
            while i <= (len(sentence) - sequence_length):
                if sentence[i : i+sequence_length] == sequence_of_tokens:
                    sentence[i : i+sequence_length] = replacement
                    i += 1
                else:
                    i += 1

    def __get_most_frequent_pair(self, corpus: List[List[Union[str, int]]]):
        pairs = {}
        for sentence in corpus:
            for t1,t2 in zip(sentence, sentence[1:]):
                pairs[(t1,t2)] = pairs.get((t1,t2), 0) + 1

        if pairs:
            return max(pairs, key=pairs.get)
        else:   # There are no pairs
            return None

    def tokenize_text(self, vocabulary, text: str):
        space = " "
        space_token = 0
        assert vocabulary[space_token] == space

        tokenized_text = []
        # Replace all the spaces with space token
        for c in text:
            tokenized_text.append(space_token if c==space else c)
            
        # Make it have the shape of corpus
        tokenized_text = [tokenized_text]

        # Iterate over the text and replace a pair of tokens (or chars) with their token
        # Replace tokens in the same order as they were created
        for token in range(1, len(vocabulary.keys())):
            sequence = vocabulary[token]
            self.__merge_tokens(tokenized_text, sequence, token)
        
        return tokenized_text[0]
