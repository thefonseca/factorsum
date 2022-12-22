import textwrap
import logging
import os

import fire
import nltk

from .extrinsic import find_best_summary
from .config import model_params
from .data import load_dataset, load_summaries
from .sampling import get_document_views
from .metrics import summarization_metrics
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
        target_budget,
        source_target_budget=0,
        target_content=None,
        custom_guidance=None,
        sample_factor=5,
        views_per_doc=20,
        verbose=False,
        seed=17,
    ):
        if type(source) == str:
            source = nltk.sent_tokenize(source)
        else:
            source = [s for x in source for s in nltk.sent_tokenize(x)]
        source = [x.replace("\n", "") for x in source if x != "\n"]

        if target_content is None and source_target_budget:
            target_content = get_source_guidance(source, source_target_budget)

        doc_views = get_document_views(
            source, sample_factor=sample_factor, views_per_doc=views_per_doc, seed=seed
        )
        summary_views = self.get_summary_views(doc_views["source_views"])
        summary, _, guidance_scores = find_best_summary(
            summary_views,
            target_budget,
            target_content=target_content,
            custom_guidance=custom_guidance,
            verbose=verbose,
        )

        return summary, guidance_scores


def _summarize(
    model,
    source,
    params,
    target=None,
    content_guidance_type=None,
    target_content=None,
    source_target_budget=None,
    custom_guidance=None,
    sample_factor=5,
    views_per_doc=20,
):

    if content_guidance_type is None:
        content_guidance_type = "no"

    budget_adjust_key = f"{content_guidance_type}_content_fixed_budget_adjust"
    budget_adjust = params.get(budget_adjust_key, 0)
    target_budget = params["token_budget"] + budget_adjust

    if content_guidance_type == "source":
        if source_target_budget is None:
            source_target_budget = target_budget
    else:
        source_target_budget = None

    setup_desc = f"> Generating summary with {content_guidance_type} "
    setup_desc += f"content guidance and budget guidance of {target_budget} ..."
    print()
    print(textwrap.fill(setup_desc, 80))

    summary, guidance_scores = model.summarize(
        source,
        target_budget=target_budget,
        target_content=target_content,
        source_target_budget=source_target_budget,
        custom_guidance=custom_guidance,
        sample_factor=sample_factor,
        views_per_doc=views_per_doc,
        verbose=True,
    )
    print("> Summary words:", sum([len(nltk.word_tokenize(sent)) for sent in summary]))
    print("> Guidance scores:")
    print(guidance_scores)
    _ = summarization_metrics(summary, target_summary=target, verbose=True)

    return summary


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
    views_per_doc=20,
    sample_factor=5,
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
    logger.info(f"Loaded eval dataset with keys: {list(eval_data.keys())}")

    target = eval_data["targets"][doc_id]
    source = eval_data["sources"][doc_id]
    n_words = sum([len(nltk.word_tokenize(sent)) for sent in target])

    print(f"> Reference summary ID: {doc_id} ({n_words} words)\n")
    for sent in target:
        print(textwrap.fill(f"  - {sent}", 80))

    if training_domain is None:
        training_domain = dataset_name

    model = FactorSum(training_domain)

    target_content = None
    if content_guidance_type and content_guidance_type != "source":
        target_content = load_summaries(
            dataset_name,
            split,
            content_guidance_type,
            training_domain,
            data_dir,
            expected_sample_count=len(eval_data["sources"]),
        )

        if target_content is None:
            return

        target_content = target_content[doc_id]

    _ = _summarize(
        model,
        source,
        params,
        target=target,
        target_content=target_content,
        content_guidance_type=content_guidance_type,
        source_target_budget=source_token_budget,
        sample_factor=sample_factor,
        views_per_doc=views_per_doc,
    )


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logging.getLogger("absl").setLevel(logging.WARNING)
    fire.Fire(run)
