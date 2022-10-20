import textwrap
import logging
import os

import fire
import nltk

from .extrinsic import find_best_summary
from .config import model_params
from .data import load_dataset, load_summaries
from .sampling import get_document_views
from .score import extrinsic_scores, show_extrinsic_scores
from .utils import load_intrinsic_model

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)


def get_source_guidance(source, token_budget, verbose=False):
    source_guidance = []
    guidance_tokens = 0
    for sent in source:
        source_guidance.append(sent)
        guidance_tokens += len(sent.split())
        if guidance_tokens > token_budget:
            break

    if verbose:
        print("Source guidance token budget:", token_budget)
        print("Tokens in source guidance:", guidance_tokens)
        print("Sentences in source guidance:", len(source_guidance))
    return source_guidance


class FactorSum:
    def __init__(self, model_name_or_path):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = None

    def get_summary_views(self, source_views, batch_size=20):
        if self.model is None:
            self.model = load_intrinsic_model(self.model_name_or_path)

        if type(source_views) == list:
            source_views = ["\n".join(view) for view in source_views]

        views = []

        for out in self.model(source_views, batch_size=batch_size, truncation=True):
            views.append(out["summary_text"])

        return views

    def summarize(
        self,
        source,
        budget_guidance,
        source_token_budget=0,
        content_guidance=None,
        verbose=False,
        seed=17,
    ):

        if content_guidance is None and source_token_budget:
            content_guidance = get_source_guidance(source, source_token_budget)

        doc_views = get_document_views(
            source, sample_factor=5, views_per_doc=20, seed=seed
        )
        summary_views = self.get_summary_views(doc_views["source_views"])
        summary, _ = find_best_summary(
            summary_views,
            content_guidance=content_guidance,
            budget_guidance=budget_guidance,
            verbose=verbose,
        )
        return summary


def _summarize(
    model,
    source,
    params,
    target=None,
    content_guidance_type=None,
    content_guidance=None,
    source_token_budget=None,
):

    if content_guidance_type is None:
        content_guidance_type = "no"

    budget_adjust_key = f"{content_guidance_type}_content_fixed_budget_adjust"
    budget_adjust = params.get(budget_adjust_key, 0)
    budget_guidance = params["token_budget"] + budget_adjust

    if content_guidance_type == "source":
        if source_token_budget is None:
            source_token_budget = budget_guidance
    else:
        source_token_budget = None

    # if content_guidance_type != "no":
    #     print(f"Using content guidance from {content_guidance_type}")

    # if content_guidance:
    #     for sent in content_guidance.split("<n>"):
    #         print(textwrap.fill(f"  - {sent}", 80))

    setup_desc = f"> Generating summary with {content_guidance_type} "
    setup_desc += f"content guidance and budget guidance of {budget_guidance} ..."
    print()
    print(textwrap.fill(setup_desc, 80))

    summary = model.summarize(
        source,
        budget_guidance=budget_guidance,
        content_guidance=content_guidance,
        source_token_budget=source_token_budget,
        verbose=True,
    )
    score = extrinsic_scores(
        summary, target_summary=target, token_budget=params["token_budget"]
    )
    print("> Summary words:", sum([len(nltk.word_tokenize(sent)) for sent in summary]))
    show_extrinsic_scores(score)


def run(
    doc_id=617,
    data_dir="data",
    dataset_name="arxiv",
    split="test",
    training_domain=None,
    budget_weight=None,
    source_token_budget=None,
    budget_guidance=None,
    intrinsic_model_id=None,
    content_guidance_type=None,
    views_per_doc=None,
    sample_factor=None,
    cache_dir=None,
):

    params = model_params(
        dataset_name,
        budget_weight=budget_weight,
        token_budget=budget_guidance,
        views_per_doc=views_per_doc,
        sample_factor=sample_factor,
        intrinsic_model_id=intrinsic_model_id,
    )

    eval_data = load_dataset(
        dataset_path=params["dataset_path"],
        dataset_name=dataset_name,
        split=split,
        data_dir=data_dir,
        cache_dir=cache_dir,
    )
    logger.info("Loaded eval dataset with keys:", list(eval_data.keys()))

    target = eval_data["targets"][doc_id]
    source = eval_data["sources"][doc_id]
    n_words = sum([len(nltk.word_tokenize(sent)) for sent in target])

    print(f"> Reference summary ID: {doc_id} ({n_words} words)\n")
    for sent in target:
        print(textwrap.fill(f"  - {sent}", 80))

    if training_domain is None:
        training_domain = dataset_name

    model = FactorSum(training_domain)

    content_guidance = None
    if content_guidance_type and content_guidance_type != "source":
        content_guidance = load_summaries(
            dataset_name,
            split,
            content_guidance_type,
            training_domain,
            data_dir,
            expected_sample_count=len(eval_data["sources"]),
        )

        if content_guidance is None:
            return

        content_guidance = content_guidance[doc_id]

    _summarize(
        model,
        source,
        params,
        target=target,
        content_guidance=content_guidance,
        content_guidance_type=content_guidance_type,
        source_token_budget=source_token_budget,
    )


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logging.getLogger("absl").setLevel(logging.WARNING)
    fire.Fire(run)
