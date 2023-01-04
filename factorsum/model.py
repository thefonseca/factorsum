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
from .utils import load_model

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


def sent_tokenize(text):
    if type(text) == str:
        sents = nltk.sent_tokenize(text)
    else:
        sents = [s for x in text for s in nltk.sent_tokenize(x)]
    sents = [x.replace("\n", "") for x in sents if x != "\n"]
    return sents


class FactorSum:
    def __init__(self, model_name_or_path):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = None

    def get_summary_views(self, source_views, batch_size=20):
        if self.model is None:
            self.model, _ = load_model(self.model_name_or_path, "intrinsic_importance")

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
        min_words_per_view=5,
        verbose=False,
        seed=17,
    ):
        source_sents = sent_tokenize(source)
        doc_views = get_document_views(
            source_sents,
            sample_factor=sample_factor,
            views_per_doc=views_per_doc,
            seed=seed,
        )
        summary_views = self.get_summary_views(doc_views["source_views"])
        if target_content is None and source_target_budget:
            target_content = get_source_guidance(source_sents, source_target_budget)

        summary, guidance_scores = find_best_summary(
            summary_views,
            target_budget,
            target_content=target_content,
            custom_guidance=custom_guidance,
            min_words_per_view=min_words_per_view,
            verbose=verbose,
        )

        return summary, guidance_scores


def load_target_content(
    doc_id,
    sources,
    content_guidance_type,
    dataset_name,
    split,
    training_domain,
    data_dir,
):
    target_content = None
    if content_guidance_type and content_guidance_type != "source":
        target_content = load_summaries(
            dataset_name,
            split,
            content_guidance_type,
            training_domain,
            data_dir,
            expected_sample_count=len(sources),
        )

        if target_content is None:
            return

        target_content = target_content[doc_id]

    return target_content


def summarize(
    model,
    source,
    params,
    target=None,
    content_guidance_type=None,
    target_content=None,
    custom_guidance=None,
):

    if target:
        n_words = sum([len(nltk.word_tokenize(sent)) for sent in target])

        print(f"> Reference summary: ({n_words} words)\n")
        for sent in target:
            print(textwrap.fill(f"  - {sent}", 80))

    if content_guidance_type is None:
        content_guidance_type = "no"

    target_budget = None
    if params.get("token_budget"):
        budget_adjust_key = f"{content_guidance_type}_content_fixed_budget_adjust"
        budget_adjust = params.get(budget_adjust_key, 0)
        target_budget = params["token_budget"] + budget_adjust

    source_token_budget = params["source_token_budget"]
    if content_guidance_type == "source":
        if source_token_budget is None and target_budget:
            source_token_budget = target_budget
    else:
        source_token_budget = None

    setup_desc = f">> Generating summary"
    if target_budget:
        setup_desc += f"\n> budget guidance: {target_budget}"
    if content_guidance_type:
        setup_desc += f"\n> content guidance: {content_guidance_type}"
        setup_desc += f"\n> source token budget: {source_token_budget}"

    print()
    print(setup_desc)

    summary, guidance_scores = model.summarize(
        source,
        target_budget=target_budget,
        target_content=target_content,
        source_target_budget=source_token_budget,
        custom_guidance=custom_guidance,
        sample_factor=params["sample_factor"],
        views_per_doc=params["views_per_doc"],
        min_words_per_view=params["min_words_per_view"],
        verbose=True,
    )
    print("> Summary words:", sum([len(nltk.word_tokenize(sent)) for sent in summary]))
    print("> Guidance scores:")
    for key, score in guidance_scores.items():
        print(f"{key}: {score:.3f}")
    _ = summarization_metrics(summary, target_summary=target, verbose=True)

    return summary


def run(
    doc_id=617,
    data_dir="data",
    dataset_name="arxiv",
    split="test",
    training_domain=None,
    content_weight=None,
    source_token_budget=None,
    token_budget=None,
    intrinsic_model_id=None,
    content_guidance_type=None,
    views_per_doc=20,
    sample_factor=5,
    min_words_per_view=5,
    cache_dir=None,
):

    params = model_params(
        dataset_name,
        content_weight=content_weight,
        token_budget=token_budget,
        source_token_budget=source_token_budget,
        views_per_doc=views_per_doc,
        sample_factor=sample_factor,
        intrinsic_model_id=intrinsic_model_id,
        min_words_per_view=min_words_per_view,
    )

    eval_data = load_dataset(
        dataset_path=params["dataset_path"],
        dataset_name=dataset_name,
        split=split,
        data_dir=data_dir,
        cache_dir=cache_dir,
    )
    logger.info(f"Loaded eval dataset with keys: {list(eval_data.keys())}")

    if training_domain is None:
        training_domain = dataset_name

    target_content = load_target_content(
        doc_id,
        eval_data["sources"],
        content_guidance_type,
        dataset_name,
        split,
        training_domain,
        data_dir,
    )

    model = FactorSum(training_domain)
    source = eval_data["sources"][doc_id]

    _ = summarize(
        model,
        source,
        params,
        target=eval_data["targets"][doc_id],
        target_content=target_content,
        content_guidance_type=content_guidance_type,
    )


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logging.getLogger("absl").setLevel(logging.WARNING)
    fire.Fire(run)
