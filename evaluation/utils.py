import os
import textwrap
import logging
import time

import nltk
import numpy as np
from rich.logging import RichHandler

from factorsum.data import load_dataset, load_summary_views, load_summaries
from factorsum.model import get_source_guidance

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)


def get_sources(data):
    if "sources" in data:
        return data["sources"]
    return data["articles"]


def get_targets(data):
    if "targets" in data:
        return data["targets"]
    return data["abstracts"]


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


def log_summary(doc_id, pred, target, score, bad_score, good_score, score_key="rouge1"):
    if bad_score and score[score_key].fmeasure < bad_score:
        logger.info("BAD SUMMARY ============")
        logger.info(f"DOC ID: {doc_id}")
        for sent in pred.split("\n"):
            logger.info(textwrap.fill(f"- {sent}", 80))
        logger.info("Abstract:")
        logger.info(textwrap.fill(target, 80))
        log_rouge_scores(score)

    if good_score and score[score_key].fmeasure > good_score:
        logger.info("GOOD SUMMARY ============")
        logger.info(f"DOC ID: {doc_id}")
        for sent in pred.split("\n"):
            logger.info(textwrap.fill(f"- {sent}", 80))
        logger.info("Abstract:")
        logger.info(textwrap.fill(target, 80))
        log_rouge_scores(score)


def get_output_path(
    save_dir,
    dataset,
    split,
    content,
    budget,
    training_domain=None,
    timestr=None,
    custom_suffix=None,
):
    save_to = None
    suffix = f"{content}_content-{budget}_budget"
    if custom_suffix:
        suffix = f"{suffix}-{custom_suffix}"

    if save_dir:
        save_dir = f"{save_dir}-{dataset}-{split}"
        if training_domain:
            save_dir = f"{save_dir}-{training_domain}"
        if timestr:
            save_dir = f"{save_dir}_{timestr}"
        save_to = f"{dataset}-{split}-{suffix}.csv"
        save_to = os.path.join(save_dir, save_to)
    return save_to


def get_log_path(
    log_dir,
    dataset,
    split,
    training_domain=None,
    timestr=None,
    prefix=None,
    suffix=None,
):
    if log_dir:
        if prefix:
            log_path = f"{prefix}-{dataset}-{split}"
        else:
            log_path = f"{dataset}-{split}"

        if training_domain:
            log_path = f"{log_path}-{training_domain}"
        if timestr:
            log_path = f"{log_path}_{timestr}"

        if suffix:
            log_path = f"{log_path}-{suffix}.txt"
        else:
            log_path = f"{log_path}.txt"

        log_path = os.path.join(log_dir, log_path)
    return log_path


def config_logging(
    dataset_name, split, save_dir, training_domain=None, prefix="factorsum"
):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_path = get_log_path(
        save_dir,
        dataset_name,
        split,
        training_domain=training_domain,
        timestr=timestr,
        prefix=prefix,
    )
    handlers = [RichHandler()]
    if log_path:
        os.makedirs(save_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w"))
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        handlers=handlers,
    )
    logging.getLogger("absl").setLevel(logging.WARNING)
    return timestr


def _load_eval_data(
    dataset_path,
    dataset_name,
    split,
    data_dir,
    cache_dir=None,
    training_domain=None,
    sample_type=None,
    sample_factor=5,
    views_per_doc=20,
    intrinsic_model_id=None,
):
    data = {}

    if training_domain is None:
        training_domain = dataset_name

    eval_dataset = load_dataset(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        split=split,
        data_dir=data_dir,
        cache_dir=cache_dir,
    )

    document_views = load_dataset(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        split=split,
        data_dir=data_dir,
        sample_type=sample_type,
        sample_factor=sample_factor,
        views_per_doc=views_per_doc,
    )

    summary_views = load_summary_views(
        dataset_name, split, data_dir, training_domain, intrinsic_model_id
    )

    if len(document_views["doc_ids"]) != len(summary_views):
        raise ValueError(
            "Number of document views does not match the number of "
            "summary views! Please check the consistency of the datasets."
        )

    data["eval_dataset"] = eval_dataset
    data["document_views"] = document_views
    data["summary_views"] = summary_views

    for model_name in ["pegasus", "bigbird-pegasus-large", "bart-base", "bart-large"]:
        n_samples = len(get_sources(eval_dataset))

        summaries = load_summaries(
            dataset_name, split, model_name, training_domain, data_dir, n_samples
        )

        if summaries:
            summary_lengths = [len(nltk.word_tokenize(s)) for s in summaries]
            data[f"{model_name}_summaries"] = summaries
            data[f"{model_name}_summary_lengths"] = summary_lengths

    return data


def _get_source_guidance(sources, token_budget):
    source_guidance = []

    for src in sources:
        guidance = get_source_guidance(src, token_budget)
        source_guidance.append(guidance)

    logger.info(f"Source guidance token budget: {token_budget}")
    logger.info(
        f"Avg sentences in source guidance: {np.mean([len(s) for s in source_guidance])}"
    )
    return source_guidance


def load_eval_data(
    dataset_path,
    dataset_name,
    split,
    params,
    data_dir,
    training_domain=None,
    cache_dir=None,
):
    eval_data = _load_eval_data(
        dataset_path,
        dataset_name,
        split,
        data_dir,
        training_domain=training_domain,
        intrinsic_model_id=params["intrinsic_importance_model_id"],
        sample_type=params["sample_type"],
        sample_factor=params["sample_factor"],
        views_per_doc=params["views_per_doc"],
        cache_dir=cache_dir,
    )

    eval_dataset = eval_data["eval_dataset"]
    source_token_budget = params.get("source_token_budget")
    if source_token_budget is None:
        source_token_budget = params["token_budget"]
    sources = get_sources(eval_dataset)
    source_guidance = _get_source_guidance(sources, source_token_budget)

    eval_data["source_summaries"] = source_guidance
    eval_data["oracle_summaries"] = get_targets(eval_dataset)
    eval_data["oracle_summary_lengths"] = None
    eval_data["bigbird_summaries"] = eval_data.get("bigbird-pegasus-large_summaries")
    eval_data["bigbird_summary_lengths"] = eval_data.get(
        "bigbird-pegasus-large_summary_lengths"
    )
    return eval_data
