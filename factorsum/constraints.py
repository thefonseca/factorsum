import re

import nltk

from .redundancy import has_similar_content

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)


class LengthConstraint:
    def __init__(self, min_length=None, max_length=None):
        self.min_length = min_length
        self.max_length = max_length
        self.length_cache = {}

    def get_length(self, text):
        if type(text) == str:
            text = [text]
        length = 0
        for sent in text:
            sent_length = self.length_cache.get(sent)
            if sent_length is None:
                sent_length = len(nltk.word_tokenize(sent))
                self.length_cache[sent] = sent_length
            length += sent_length
        return length

    def check(self, *texts):
        lengths = [self.get_length(text) for text in texts]
        length = sum(lengths)
        is_valid = True
        if self.min_length:
            is_valid = is_valid and length >= self.min_length
        if self.max_length:
            is_valid = is_valid and length <= self.max_length
        return is_valid


class BudgetConstraint(LengthConstraint):
    def __init__(self, max_length):
        super().__init__(max_length=max_length)

    def check(self, current_summary, candidate_view):
        if type(candidate_view) == str:
            candidate_view = [candidate_view]
        return super().check(current_summary, candidate_view)


class SummaryViewConstraint(LengthConstraint):
    def __init__(self, min_length=5):
        super().__init__(min_length=min_length)
        self.non_alpha_pattern = re.compile("[^\w_\s]+")

    def check(self, candidate_view):
        view_sents = []
        if type(candidate_view) == str:
            candidate_view = [candidate_view]
        for sent in candidate_view:
            view_sents.append(self.non_alpha_pattern.sub("", sent))
        return super().check(view_sents)


class RedundancyConstraint:
    def __init__(self, threshold=0.4, max_tokens=30):
        self.threshold = threshold
        self.max_tokens = max_tokens

    def check(self, current_summary, candidate_view):
        return not has_similar_content(
            candidate_view,
            current_summary,
            threshold=self.threshold,
            max_tokens=self.max_tokens,
        )
