from rouge_score import rouge_scorer
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)


def log_rouge_scores(scores):
    for k, v in sorted(scores.items()):
        if hasattr(v, "low"):
            print("%s-R: %f,%f,%f" % (k, v.low.recall, v.mid.recall, v.high.recall))
            print(
                "%s-P: %f,%f,%f"
                % (k, v.low.precision, v.mid.precision, v.high.precision)
            )
            print(
                "%s-F: %f,%f,%f\n"
                % (k, v.low.fmeasure, v.mid.fmeasure, v.high.fmeasure)
            )
        else:
            print("%s-R: %f,%f,%f" % (k, v.recall, v.recall, v.recall))
            print("%s-P: %f,%f,%f" % (k, v.precision, v.precision, v.precision))
            print("%s-F: %f,%f,%f\n" % (k, v.fmeasure, v.fmeasure, v.fmeasure))


def rouge_score(summary, target_summary, rouge_ngrams=None):

    score = {}

    if rouge_ngrams is None or len(rouge_ngrams) == 0:
        rouge_ngrams = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge = rouge_scorer.RougeScorer(rouge_ngrams, use_stemmer=True)

    if type(target_summary) == list:
        target_summary_str = "\n".join(target_summary)
    else:
        target_summary_str = target_summary

    if type(summary) == list:
        summary_str = "\n".join(summary)
    else:
        summary_str = summary

    score["rouge"] = rouge.score(target_summary_str, summary_str)

    return score


def show_metrics(metrics):
    if "rouge" in metrics:
        print("\n> ROUGE scores:")
        log_rouge_scores(metrics["rouge"])


def summarization_metrics(
    summary, target_summary=None, rouge_ngrams=None, verbose=False
):
    metrics = {}

    if target_summary is not None:
        rouge = rouge_score(summary, target_summary, rouge_ngrams=rouge_ngrams)
        for key, value in rouge.items():
            metrics[key] = value

    if verbose:
        show_metrics(metrics)

    return metrics
