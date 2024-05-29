from abc import ABC, abstractmethod
from typing import List

class TokenizationMethod(ABC):
    @abstractmethod
    def create_vocabulary(self, corpus: List[str], vocab_size: int):  # TODO add return type
        """
        Creates
        
        Arguments
        ---------
        corpus: List[str]
            A list of string that makes up the training corpus from which to learn the tokens.
        vocab_size: int
            The maximum size for the vocabulary.
        
        Returns
        -------
        vocab: TODO add 
            The learned vocabulary.
        """
        pass


    @abstractmethod
    def tokenize_text(self, vocabulary, text: str):  # TODO add return type
        """
        Transform the text into tokens

        Arguments
        ---------
        vocabulary: TODO add type
            The vocabulary learned in the training phase. 
        text: str
            The text to tokenize.

        Returns
        -------
        tokens: TODO add type
            The text as tokens
        """
        pass

    @classmethod
    def get_all_characters():
        return "a b c d e f g h i j k l m n ñ o p q r s t u v w x y z".split(" ")