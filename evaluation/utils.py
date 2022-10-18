import os
import textwrap
import logging

import nltk
import numpy as np

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


def log_summary(doc_id, pred, target, score, bad_score, good_score, score_key="rouge1"):
    if bad_score and score[score_key].fmeasure < bad_score:
        print("\n>> BAD SUMMARY ============")
        print("> DOC ID:", doc_id)
        for sent in pred.split("\n"):
            print(textwrap.fill(f"- {sent}", 80))
        print()
        print("> Abstract:")
        print(textwrap.fill(target, 80))
        print("\n> Scores:")
        log_rouge_scores(score)

    if good_score and score[score_key].fmeasure > good_score:
        print("\n>> GOOD SUMMARY ============")
        print("> DOC ID:", doc_id)
        for sent in pred.split("\n"):
            print(textwrap.fill(f"- {sent}", 80))
        print()
        print("> Abstract:")
        print(textwrap.fill(target, 80))
        print("\n> Scores:")
        log_rouge_scores(score)


def get_output_path(
    save_dir, dataset, split, content, budget, training_domain=None, timestr=None
):
    save_to = None
    suffix = f"{content}_content-{budget}_budget"

    if save_dir:
        save_dir = f"{save_dir}-{dataset}-{split}"
        if training_domain:
            save_dir = f"{save_dir}-{training_domain}"
        if timestr:
            save_dir = f"{save_dir}_{timestr}"
        save_to = f"{dataset}-{split}-{suffix}.csv"
        save_to = os.path.join(save_dir, save_to)
    return save_to


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
    guidance_tokens_list = []

    for src in sources:
        guidance = get_source_guidance(src, token_budget)
        guidance_tokens = 0
        source_guidance.append(guidance)
        guidance_tokens_list.append(guidance_tokens)

    print("Source guidance token budget:", token_budget)
    print("Avg tokens in source guidance:", np.mean(guidance_tokens_list))
    print(
        "Avg sentences in source guidance:", np.mean([len(s) for s in source_guidance])
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
        intrinsic_model_id=params["intrinsic_model_id"],
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
