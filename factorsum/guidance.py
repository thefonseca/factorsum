import nltk

from .metrics import rouge_score

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)


class Guidance:
    def __init__(self, parallelizable=True):
        """A base class for guidance types.

        Args:
            parallelizable (bool, optional): indicates that it is safe to
            perform guidance computation using multiprocessing. Should be
            set to false if guidance involves CUDA computation. Defaults to True.
        """
        self.parallelizable = parallelizable


class ROUGEContentGuidance(Guidance):
    def __init__(
        self, target_content, weight=1.0, rouge_ngrams=None, score_key=None, **kwargs
    ):
        super().__init__(**kwargs)

        if isinstance(target_content, list):
            target_content = "\n".join(target_content)
        self.target_content = target_content
        self.weight = weight
        if rouge_ngrams is None:
            rouge_ngrams = ["rouge1"]
        self.rouge_ngrams = rouge_ngrams
        if score_key is None:
            score_key = "recall"
        self.score_key = score_key

    def score(self, candidate_summary):
        score = {}
        if self.target_content is not None:

            if isinstance(candidate_summary, (list, tuple)):
                candidate_summary = "\n".join(candidate_summary)

            rouge = rouge_score(
                candidate_summary, self.target_content, rouge_ngrams=self.rouge_ngrams
            )

            for key, value in rouge["rouge"].items():
                score[key] = value

            total_score = sum(
                [getattr(score[ngram], self.score_key) for ngram in self.rouge_ngrams]
            )
            return self.weight * total_score


class BudgetGuidance(Guidance):
    def __init__(self, target_budget, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.target_budget = target_budget
        self.weight = weight

    def score(self, candidate_summary):
        summary_tokens = 0

        for sent in candidate_summary:
            summary_tokens += len(nltk.word_tokenize(sent))

        error = abs(self.target_budget - summary_tokens) / self.target_budget  # ** 2
        return self.weight * -error
