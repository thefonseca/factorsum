import re

from tqdm import tqdm

"""
This code is adapted from BERTSum implementation:
https://github.com/nlpyang/BertSum/blob/master/src/prepro/data_builder.py
https://arxiv.org/pdf/1903.10318.pdf
"""


def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences."""
    assert len(sentences) > 0
    assert n > 0
    words = sum(sentences, [])
    return _get_ngrams(n, words)


def _cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def _greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    sents = [_rouge_clean(s).split() for s in doc_sent_list]

    selected = []

    for abstract_sent in abstract_sent_list:
        abstract = _rouge_clean(abstract_sent).split()
        evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
        reference_1grams = _get_word_ngrams(1, [abstract])
        evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
        reference_2grams = _get_word_ngrams(2, [abstract])

        for s in range(summary_size):
            cur_max_rouge = max_rouge
            cur_id = None
            selected_cur_id = None

            for i in range(len(sents)):
                c = [i]
                candidates_1 = [evaluated_1grams[idx] for idx in c]
                candidates_1 = set.union(*map(set, candidates_1))
                candidates_2 = [evaluated_2grams[idx] for idx in c]
                candidates_2 = set.union(*map(set, candidates_2))
                rouge_1 = _cal_rouge(candidates_1, reference_1grams)["f"]
                rouge_2 = _cal_rouge(candidates_2, reference_2grams)["f"]
                rouge_score = rouge_1 + rouge_2

                # print('candidates_1', candidates_1)
                # print('reference_1', reference_1grams)
                # print('candidates_2', candidates_2)
                # print('reference_2', reference_2grams)
                # print(rouge_1, rouge_2, rouge_score)
                if rouge_score > cur_max_rouge:
                    if i in selected:
                        selected_cur_id = i
                    else:
                        cur_max_rouge = rouge_score
                        cur_id = i

            if cur_id is not None:
                selected.append(cur_id)
            elif selected_cur_id is not None:
                selected.append(selected_cur_id)
            else:
                # print('Abstract sentence:', abstract_sent)
                # print('No sentence with ROUGE > 0 found in article!')
                selected.append(None)

    return selected


def get_oracles(
    articles, abstracts, oracle_type="rouge", min_words=10, progress_bar=True
):
    oracles = []

    items = zip(articles, abstracts)
    if progress_bar:
        items = tqdm(items)

    for article, abstract in items:
        abstract = [s if len(s.split()) > min_words else "" for s in abstract]

        if oracle_type == "rouge":
            selection = _greedy_selection(article, abstract, summary_size=1)
        else:
            raise ValueError("Unsupported oracle type:", oracle_type)

        oracles.append(selection)

        assert len(selection) == len(abstract), (len(selection), len(abstract))

    return oracles
