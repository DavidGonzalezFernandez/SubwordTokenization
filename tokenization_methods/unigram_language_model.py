from .tokenization_method import TokenizationMethod
from typing import List, Dict, Optional
import math
from collections import defaultdict
from operator import mul
from functools import reduce

space_char = " "

class UnigramLanguageModel(TokenizationMethod):
    def __init__(self):
        self.segmentations = {}     # Dictionary to memoize available segmentations

    def create_vocabulary(
        self,
        corpus: List[str],
        vocab_size: int,
        percentage_to_remove,
        starting_vocabulary: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Creates the vocabulary of size 'vocab_size' given an initial vocabulary and a training corpus.

        Arguments
        ---------
        corpus: List[str]
            A list of string that makes up the training corpus from which to learn the tokens.
        vocab_size: int
            The desired size for the vocabulary.
        percentage_to_remove: float
            The percentage of symbols to be deleted from the current vocabulary in each iteration.
        ratio_initial_vocab_to_final_vocab
            How much larger the initial vocabulary will be compared to the desired vocabulary size.
            First, the vocabulary is made up from all the substrings in all the words in the corpus.
            Then, only the (vocab_size * ratio_initial_vocab_to_final_vocab) substrings with the highest
            frequence are selected to form the initial vocabulary to start the training phase.

        Returns
        -------
        vocab: List[tuple]
            The learned vocabulary.
        """
        # Count the number of appearances of each word in the corpus
        words_in_corpus = defaultdict(int)
        for sentence in corpus:
            for word in sentence.split(space_char):
                word = word.strip()
                if word:
                    words_in_corpus[word] += 1
        del corpus

        # Create the vocabulary
        if starting_vocabulary is None:
            # Get use all the substrings in the corpus as vocabulary
            substring_frequencies = self.__get_all_substrings_and_frequences(words_in_corpus)
            # To reduce execution time, select only a subset of all the substrings
            substrings_to_select = {k:v for k,v in substring_frequencies.items() if v>1}
            substring_frequencies = {k:substring_frequencies[k] for k in substrings_to_select}
            # Calculate the probabilities of each substring
            substring_probabilities_in_corpus = self.__get_probabilities(substring_frequencies)
        else:
            substring_frequencies = self.__get_substring_frequencies_in_corpus(words_in_corpus, starting_vocabulary.keys())
            substring_probabilities_in_corpus = self.__get_probabilities(substring_frequencies)
        del starting_vocabulary

        # Remove tokens until the vocab has the desired size
        while len(substring_probabilities_in_corpus) > vocab_size:
            # Empty the memoization dict
            self.segmentations.clear()

            # Calculate how many tokens to remove in this iteration
            n_tokens_to_remove = min(len(substring_probabilities_in_corpus)-vocab_size, len(substring_probabilities_in_corpus)*percentage_to_remove)

            # Get the loss with the current vocabulary
            current_loss = self.__get_corpus_score(words_in_corpus, substring_probabilities_in_corpus)

            # Calculate all the new losses after removing only one token
            losses_after_removing_tokens = self.__get__losses_after_removing_one_token(words_in_corpus, substring_frequencies)
            if not losses_after_removing_tokens:
                break

            # Calculate the increase in loss
            losses_incresase_after_removing_tokens = {k:(v-current_loss) for (k,v) in losses_after_removing_tokens.items()}

            # Get the N lowest tokens with the lowest increases
            tokens_to_remove = sorted(losses_incresase_after_removing_tokens, key=losses_incresase_after_removing_tokens.get)[:int(n_tokens_to_remove)]

            # Create the new vocabulary removing the selected tokens
            substring_frequencies = {k:v for (k,v) in substring_frequencies.items() if k not in tokens_to_remove}
            substring_probabilities_in_corpus = self.__get_probabilities(substring_frequencies)

            assert len(substring_frequencies) == len(substring_probabilities_in_corpus)
            assert set(substring_frequencies.keys()) == set(substring_probabilities_in_corpus.keys())

        return substring_probabilities_in_corpus

    def __get_substring_frequencies_in_corpus(self, corpus_words, substrings):
        return {
            substring: sum(appearances for (w,appearances) in corpus_words.items() if substring in w)
            for substring in substrings
        }
    
    def __get__losses_after_removing_one_token(self, corpus_words: dict, frequences):
        """Calculates the loss after removing each token from the frequency distribution. The tokens that belong
        to an individual character are not removed from the corpus."""
        return {
            token: self.__get_corpus_score(
                corpus_words, self.__get_probabilities({k:v for k,v in frequences.items() if k!=token})
            ) for token in frequences.keys() if len(token)>1    # Don't remove tokens for 1 character
        }
    
    def __get_probabilities(self, frequences):
        """Given a list of frequences, calculates the probabilities of each substring."""
        sum_frequences = sum(frequences.values())
        return {s:(f/sum_frequences) for (s,f) in frequences.items()}

    def __get_corpus_score(self, corpus_words: dict, vocab: dict):
        """Returns the score of a corpus. The score if the sum of the logs (with positive sign) for all the words."""
        return sum(
            appearances * (-math.log(self.__get_tokenization_and_score(vocab, word)[1]))
            for word, appearances in corpus_words.items()
        )

    def __get_tokenization_and_score(self, vocab, word):
        """Returns the tokenization and score of a word. For a given word, of all the possible tokenizations,
        the chosen tokenization is the one with the highest score. The score is just the result of multiplying
        the individual probabilities of each of the tokens (substrings) in the sequence of tokens.
        """
        segmentations = self.__get_posible_segmentations(vocab.keys(), word)

        max_score = 1e-308 # Don't use probability equal to 0 because then the code will calculate the log
        best_tokenization = []

        for segmentation in segmentations:
            score = reduce(mul, (vocab[tk] for tk in segmentation), 1)
            if score > max_score:
                max_score = score
                best_tokenization = segmentation

        return best_tokenization, max_score
    
    def __get_posible_segmentations(self, substrings, word):
        """For a given word, returns a list with all the possible segmentations it can be tokenized as."""
        segmentations = self.segmentations.get(word, None)  # Try to get the segmentation from the memoization dictionary

        if segmentations is not None:
            # Check that all the segmentations only contain substrings that belong to the substrings set
            return [segmentation for segmentation in segmentations if all(tk in substrings for tk in segmentation)]
            
        else:
            substrings_in_the_word = {s for s in substrings if s in word}

            segmentations = []
            def backtrack(current, remaining):
                if not remaining:
                    segmentations.append(current)
                    return
                for s in substrings_in_the_word:
                    if remaining.startswith(s):
                        backtrack(current + (s,), remaining[len(s):])

            # Calculate the segmentations
            backtrack(tuple(), word)

            # Add the segmentations to the memoization dictionary to avoid recalculating them
            self.segmentations[word] = segmentations
            return segmentations
    
    def __get_all_substrings_and_frequences(self, word_list: dict) -> dict:
        """
        Given an the dictionary of words (and number of appearances), calculates all the substrings contained
        in the words and their number of appearances in the whole list of words.
        """
        substrings = defaultdict(int)

        for word in word_list.keys():
            if not word:
                continue

            word_length = len(word)
            for i in range(word_length):  # Start of the substring
                for j in range(i+1, word_length+1):  # End of the substring
                    substring = word[i:j]
                    if substring != word:
                        substrings[substring] += word_list[word]
            
        return substrings
    
    def tokenize_text(self, vocabulary: Dict[str, float], text: str):
        tokenized_text = []

        for word in text.split(space_char):
            tokenized_text.append(self.__get_tokenization_and_score(vocabulary, word)[0])

        return tokenized_text
