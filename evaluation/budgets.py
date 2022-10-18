import os
import itertools
import logging

from tqdm.auto import tqdm
import fire
import numpy as np

from .utils import load_eval_data
from .factorsum import eval_with_guidance, default_eval_args
from factorsum.config import model_params


def evaluate_budgets(eval_fn, start=150, end=300, step=10, title=None):
    scores = {
        "kl": [],
        "rouge1_recall": [],
        "rouge1_precision": [],
        "rouge1_fmeasure": [],
    }
    budgets = []

    for token_budget in tqdm(range(start, end + step, step)):
        if title:
            print("\n>>", title)
        print("> token_budget:", token_budget)
        print()
        score = eval_fn(token_budget)
        scores["rouge1_precision"].append(score["rouge"]["rouge1"].mid.precision)
        scores["rouge1_recall"].append(score["rouge"]["rouge1"].mid.recall)
        scores["rouge1_fmeasure"].append(score["rouge"]["rouge1"].mid.fmeasure)
        budgets.append(token_budget)

    for k, v in scores.items():
        # print all values for graphs
        if len(v) > 0:
            print("\n>", k, "values:")
            print(v)

    print("\nbest budget:", budgets[np.argmax(scores["rouge1_fmeasure"])])
    print("best ROUGE-1 F1 score:", max(scores["rouge1_fmeasure"]))
    return scores, budgets


def _eval_budget(
    budget,
    eval_data,
    content_type,
    budget_type,
    params,
    default_kwargs,
    method,
    summary_type,
):

    params["token_budget"] = budget
    return eval_with_guidance(
        eval_data,
        content_type,
        budget_type,
        params,
        default_kwargs,
        save_to=None,
        method=method,
        summary_type=summary_type,
    )


def evaluate(
    data_dir="data",
    max_samples=10000,
    dataset_name="arxiv",
    split="test",
    training_domain=None,
    budget_weight=None,
    source_token_budget=None,
    summary_type=None,
    intrinsic_model_id=None,
    samples_per_doc=None,
    sample_factor=None,
    method="factorsum",
    cache_dir=None,
    seed=17,
):

    if dataset_name == "pubmed":
        start_budget = 150
        end_budget = 300
    elif dataset_name == "govreport":
        start_budget = 550
        end_budget = 700
    elif dataset_name == "arxiv":
        start_budget = 100
        end_budget = 250

    content_types = ["no", "source", "bigbird", "bart-large"]
    budget_types = ["fixed"]

    params = model_params(
        dataset_name,
        source_token_budget=source_token_budget,
        budget_weight=budget_weight,
        samples_per_doc=samples_per_doc,
        sample_factor=sample_factor,
        intrinsic_model_id=intrinsic_model_id,
    )
    eval_data = load_eval_data(
        params["dataset_path"],
        dataset_name,
        split,
        params,
        data_dir,
        training_domain,
        cache_dir,
    )
    default_kwargs = default_eval_args(params, max_samples, seed)

    for content_type, budget_type in itertools.product(content_types, budget_types):
        evaluate_budgets(
            lambda budget: _eval_budget(
                budget,
                eval_data,
                content_type,
                budget_type,
                params,
                default_kwargs,
                method,
                summary_type,
            ),
            start=start_budget,
            end=end_budget,
        )


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logging.getLogger("absl").setLevel(logging.WARNING)
    fire.Fire(evaluate)
