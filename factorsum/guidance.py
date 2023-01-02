import nltk

from .metrics import rouge_score

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)


class ROUGEContentGuidance:
    def __init__(self, target_content, weight=1.0, rouge_ngrams=None):
        if type(target_content) == list:
            target_content = "\n".join(target_content)
        self.target_content = target_content
        if rouge_ngrams is None:
            rouge_ngrams = ["rouge1"]
        self.rouge_ngrams = rouge_ngrams
        self.weight = weight

    def score(self, candidate_summary):
        score = {}
        if self.target_content is not None:

            rouge = rouge_score(
                candidate_summary, self.target_content, rouge_ngrams=self.rouge_ngrams
            )

            for key, value in rouge["rouge"].items():
                score[key] = value

            total_score = sum([score[ngram].recall for ngram in self.rouge_ngrams])
            return self.weight * total_score


class BudgetGuidance:
    def __init__(self, target_budget, weight=1.0):
        self.target_budget = target_budget
        self.weight = weight

    def score(self, candidate_summary):
        summary_tokens = 0

        for sent in candidate_summary:
            summary_tokens += len(nltk.word_tokenize(sent))

        error = abs(self.target_budget - summary_tokens) / self.target_budget  # ** 2
        return self.weight * -error
