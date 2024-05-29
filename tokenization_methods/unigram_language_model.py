from .tokenization_method import TokenizationMethod
from typing import List
import math

class UnigramLanguageModel(TokenizationMethod):
    def create_vocabulary(
        self,
        corpus: List[str],
        vocab_size: int,
        base_vocabulary: List[tuple],
        p: float
    ):
        """
        Creates the vocabulary of size 'vocab_size' given an initial vocabulary and a training corpus.

        Arguments
        ---------
        corpus: List[str]
            A list of string that makes up the training corpus from which to learn the tokens.
        vocab_size: int
            The maximum size for the vocabulary.
        initial_vocabulary: List[tuple]
            The initial big vocabulary. Iterativelly tokens will be removed until its size reaches vocab_size
        p: float
            The percentage of symbols to be deleted in each iteration.

        Returns
        -------
        vocab: List[tuple]
            The learned vocabulary.
        """
        # Transform corpus into a list of words
        words = [word for sentence in corpus for word in sentence.split(self.__space_char)]
        del corpus
        
        # The initial vocabulary is big, in each iteration tokens will be removed
        vocab = base_vocabulary.copy()
        del base_vocabulary

        while len(vocab) > vocab_size:
            # Calculate current loss
            current_loss = self.__get_loss(words, vocab)
            # Calculate the hypothetical losses after removing 1 token in the vocab
            losses_after_removing_one_token = self.__get_loss_after_removing_one_token(words, vocab)
            # Calculate the increase in the losses
            increse_in_loss = {k:(v-current_loss) for k,v in losses_after_removing_one_token.items()}

            # Calculate how many tokens to remove
            n_tokens_to_remove = min(len(vocab)-vocab_size, len(vocab)*p)
            # Select n tokens
            tokens_to_remove = sorted(increse_in_loss, key=increse_in_loss.get)[:n_tokens_to_remove]
            # Remove those tokens
            vocab = {k:v for k,v in vocab if k not in tokens_to_remove}

        return vocab
    
    # TODO implement
    def __get_tokenization_and_score_for_word(self, word, vocab):
        pass

    def __get_loss(self, words, vocab):
        return sum((self.__get_tokenization_and_score_for_word(word, vocab)[1] for word in words))

    def __get_loss_after_removing_one_token(self, words, vocab):
        losses = {}
        keys_to_remove = None   # TODO select all tokens that don't pair with 1 individual character

        for token_to_remove in keys_to_remove:
            new_vocab = {k:v for k,v in vocab.items() if k!=token_to_remove}
            losses[token_to_remove] = self.__get_loss(words, new_vocab)     

        return losses
    
    # TODO implement
    def tokenize_text(self, vocabulary, text: str):
        raise NotImplementedError()