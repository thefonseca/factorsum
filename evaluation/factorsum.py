import os
import itertools
import time
import logging

import fire

from .utils import get_targets, get_output_path, load_eval_data
from .evaluation import evaluate_sampled
from factorsum.config import model_params


def _join_texts(content, eval_data):
    content_list = content.split("+")
    text_guidance = []

    for item in content_list:
        summaries = eval_data.get(f"{item}_summaries")
        if summaries is None:
            raise ValueError("Could not load summaries:", item)
        if len(text_guidance) == 0:
            text_guidance = summaries
            continue
        if len(text_guidance) != len(summaries):
            raise ValueError(
                f"{item} content guidance length is inconsistent: \
            {len(text_guidance)} != {len(summaries)}"
            )

        # merge summary content
        for idx, summary in enumerate(summaries):
            if type(summary) == list:
                text_guidance[idx].extend(summary)
            elif type(text_guidance[idx]) == str:
                text_guidance[idx] = f"{text_guidance[idx]} {summary}"
            else:
                text_guidance[idx].append(summary)

    return text_guidance


def default_eval_args(params, max_samples, seed):
    default_kwargs = dict(
        strict_budget=False,
        oracle_budget=False,
        max_target_tokens=None,
        content_weight=params["content_weight"],
        budget_weight=params["budget_weight"],
        min_words_per_view=params["min_words_per_view"],
        n_samples=max_samples,
        text_guidance=None,
        seed=seed,
    )
    return default_kwargs


def get_guidance_kwargs(initial_kwargs, eval_data, params, budget_type, content_type):
    kwargs = initial_kwargs.copy()

    if budget_type == "fixed":
        print(f'token_budget: {params["token_budget"]}')
        kwargs["token_budget"] = params["token_budget"]
    else:
        kwargs["token_budget"] = eval_data.get(f"{budget_type}_summary_lengths")

    if (
        kwargs["token_budget"] is None and budget_type != "oracle"
    ):  # and budget in ['pegasus', 'bigbird']:
        print(f"Skipping: {budget_type}_summary_lengths not found in evaluation data")
        return None

    if content_type != "no":
        kwargs["text_guidance"] = _join_texts(content_type, eval_data)

    empty_text_guidance = (
        kwargs["text_guidance"] is None or len(kwargs["text_guidance"]) == 0
    )
    if empty_text_guidance and content_type in ["pegasus", "bigbird"]:
        print(f"Skipping: {content_type}_summaries not found in evaluation data")
        return None

    kwargs["oracle_budget"] = budget_type == "oracle"
    budget_adjust_key = f"{content_type}_content_{budget_type}_budget_adjust"
    budget_adjust = params.get(budget_adjust_key)

    if budget_adjust is None:
        print(f"Warning: budget adjustment is not set ({budget_adjust_key})")
        print(f"Using default value: {budget_adjust_key} = 0")
        budget_adjust = 0
    print(f"adjust_budget: {budget_adjust}")
    kwargs["adjust_budget"] = budget_adjust

    kwargs["content_weight"] = params["content_weight"]
    kwargs["budget_weight"] = params["budget_weight"]
    print("Content weight:", kwargs["content_weight"])
    print("Budget weight:", kwargs["budget_weight"])

    return kwargs


def get_summary_views(eval_data, summary_type):
    print("Summary views type:", summary_type)
    if summary_type is None or summary_type == "summary_views":
        summary_views = eval_data["summary_views"]
    else:
        """
        In this case we use regular summary predictions as "views"
        to test how they fare with FactorSum extrinsic ranker.
        For more information, refer to Section 4.3 (Ablation Study) of
        the paper: https://arxiv.org/pdf/2205.12486.pdf
        """
        summary_views = _join_texts(summary_type, eval_data)
        # summary_views = pd.Series(summary_views)
        # doc_ids = pd.Series(range(len(summary_views)))

    print("Summary views count:", len(summary_views))
    return summary_views


def eval_with_guidance(
    eval_data,
    content_type,
    budget_type,
    params,
    default_kwargs,
    save_to=None,
    method="factorsum",
    summary_type=None,
):
    print(">> Summarization method:", method)
    print(f"> {content_type} content guidance, {budget_type} budget")

    kwargs = get_guidance_kwargs(
        default_kwargs, eval_data, params, budget_type, content_type
    )

    if kwargs is None:
        return

    doc_ids = eval_data["document_views"]["doc_ids"]
    targets = get_targets(eval_data["eval_dataset"])
    summary_views = get_summary_views(eval_data, summary_type)

    return evaluate_sampled(
        summary_views, targets, doc_ids, method=method, save_preds_to=save_to, **kwargs
    )


def get_content_types(summarization_method, content_types):
    if summarization_method == "textrank":
        allowed_content_types = ["no"]
    else:
        allowed_content_types = [
            "no",
            "oracle",
            "source",
            "pegasus",
            "bigbird",
            "bart-base",
            "bart-large",
            "source+bigbird",
        ]

    if content_types is None or len(content_types) == 0:
        content_types = allowed_content_types

    return content_types, allowed_content_types


def get_budget_types(summarization_method, budget_types):
    if summarization_method == "textrank":
        allowed_budget_types = ["fixed", "oracle"]
    else:
        allowed_budget_types = [
            "fixed",
            "oracle",
            "pegasus",
            "bigbird",
            "bart-base",
            "bart-large",
        ]

    if budget_types is None or len(budget_types) == 0:
        budget_types = allowed_budget_types

    return budget_types, allowed_budget_types


def evaluate(
    data_dir="data",
    max_samples=10000,
    dataset_name="arxiv",
    split="test",
    content_types=None,
    budget_types=None,
    training_domain=None,
    budget_weight=None,
    source_token_budget=None,
    token_budget=None,
    summary_type="summary_views",
    intrinsic_model_id=None,
    samples_per_doc=None,
    sample_factor=None,
    min_words_per_view=None,
    method="factorsum",
    cache_dir=None,
    save_dir="output/factorsum",
    seed=17,
):

    timestr = time.strftime("%Y%m%d-%H%M%S")
    params = model_params(
        dataset_name,
        source_token_budget=source_token_budget,
        budget_weight=budget_weight,
        token_budget=token_budget,
        samples_per_doc=samples_per_doc,
        sample_factor=sample_factor,
        intrinsic_model_id=intrinsic_model_id,
        min_words_per_view=min_words_per_view,
    )

    eval_data = load_eval_data(
        params["dataset_path"],
        dataset_name,
        split,
        params,
        data_dir,
        training_domain=training_domain,
        cache_dir=cache_dir,
    )
    default_kwargs = default_eval_args(params, max_samples, seed)
    content_types, allowed_content_types = get_content_types(method, content_types)
    budget_types, allowed_budget_types = get_budget_types(method, budget_types)

    for content_type, budget_type in itertools.product(content_types, budget_types):
        assert content_type in allowed_content_types
        assert budget_type in allowed_budget_types

        if content_type != "no" and eval_data.get(f"{content_type}_summaries") is None:
            continue

        save_to = get_output_path(
            save_dir,
            dataset_name,
            split,
            content_type,
            budget_type,
            training_domain=training_domain,
            timestr=timestr,
        )
        eval_with_guidance(
            eval_data,
            content_type,
            budget_type,
            params,
            default_kwargs,
            save_to,
            method,
            summary_type=summary_type,
        )


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logging.getLogger("absl").setLevel(logging.WARNING)
    fire.Fire(evaluate)
