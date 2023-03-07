import logging

from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


def log_rouge_scores(scores):
    info = ["ROUGE scores:"]

    for k, v in sorted(scores.items()):
        if hasattr(v, "low"):
            score_info = [
                "%s-R: %f,%f,%f" % (k, v.low.recall, v.mid.recall, v.high.recall),
                "%s-P: %f,%f,%f"
                % (k, v.low.precision, v.mid.precision, v.high.precision),
                "%s-F: %f,%f,%f" % (k, v.low.fmeasure, v.mid.fmeasure, v.high.fmeasure),
            ]
        else:
            score_info = [
                "%s-R: %f,%f,%f" % (k, v.recall, v.recall, v.recall),
                "%s-P: %f,%f,%f" % (k, v.precision, v.precision, v.precision),
                "%s-F: %f,%f,%f" % (k, v.fmeasure, v.fmeasure, v.fmeasure),
            ]
        info.append("\n".join(score_info))
        info.append(" ")

    logger.info("\n".join(info))


def rouge_score(summary, target_summary, rouge_ngrams=None):
    score = {}
    if rouge_ngrams is None or len(rouge_ngrams) == 0:
        rouge_ngrams = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge = rouge_scorer.RougeScorer(rouge_ngrams, use_stemmer=True)

    if isinstance(target_summary, (list, tuple)):
        target_summary_str = "\n".join(target_summary)
    else:
        target_summary_str = target_summary

    if isinstance(summary, (list, tuple)):
        summary_str = "\n".join(summary)
    else:
        summary_str = summary

    score["rouge"] = rouge.score(target_summary_str, summary_str)
    return score


def show_metrics(metrics):
    if "rouge" in metrics:
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
