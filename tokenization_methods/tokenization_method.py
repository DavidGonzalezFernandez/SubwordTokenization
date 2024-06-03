from abc import ABC, abstractmethod
from typing import List

class TokenizationMethod(ABC):
    @abstractmethod
    def create_vocabulary(self, corpus: List[str], vocab_size: int, **kwargs):
        """
        Creates
        
        Arguments
        ---------
        corpus: List[str]
            A list of string that makes up the training corpus from which to learn the tokens.
        vocab_size: int
            The maximum size for the vocabulary.
        kwargs:
            Other parameters that are optional in some concrete implementations
        
        Returns
        -------
        vocab
            The learned vocabulary.
        """
        pass

    @abstractmethod
    def tokenize_text(self, vocabulary, text: str):
        """
        Transform the text into tokens.

        Arguments
        ---------
        vocabulary
            The vocabulary learned in the training phase. 
        text: str
            The text to tokenize.

        Returns
        -------
        tokens
            The text as tokens
        """
        pass
