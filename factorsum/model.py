import logging
import os
from functools import lru_cache

import fire
import nltk
from rich.logging import RichHandler

from .extrinsic import find_best_summary
from .sampling import get_document_views
from .metrics import summarization_metrics
from .utils import load_model, show_summary, sent_tokenize
from .memoizer import memoize

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)


def _print_guidance_scores(scores):
    info = ["Guidances scores:"]
    for key in scores.keys():
        if type(scores[key]) == dict:
            _scores = [f"{scores[key][x]:.3f}" for x in ["low", "mean", "high"]]
            _scores = ", ".join(_scores)
        else:
            _scores = f"{scores[key]:.3f}"
        info.append(f"{key}: {_scores}")
    logger.info("\n".join(info))


def _log_reference_summary(target, sent_tokenize_fn=None):
    n_words = sum([len(nltk.word_tokenize(sent)) for sent in target])
    logger.info(f"Reference summary: ({n_words} words)")

    if type(target) == list:
        target = "\n".join(target)

    if sent_tokenize_fn:
        target = sent_tokenize_fn(target)
    else:
        target = sent_tokenize(target)

    show_summary(target)


def get_source_guidance(source, token_budget, verbose=False):
    source_guidance = []
    guidance_tokens = 0

    for sent in source:
        source_guidance.append(sent)
        guidance_tokens += len(sent.split())
        if guidance_tokens > token_budget:
            break

    if verbose:
        logger.info(f"Source guidance token budget: {token_budget}")
        logger.info(f"Tokens in source guidance: {guidance_tokens}")
        logger.info(f"Sentences in source guidance: {len(source_guidance)}")
    return source_guidance


class FactorSum:
    def __init__(self, model_name_or_path):
        super().__init__()
        self.model_name_or_path = model_name_or_path

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_intrinsic_model(model_name_or_path):
        model, _ = load_model(model_name_or_path, "intrinsic_importance")
        return model

    @memoize()
    def _get_summary_views(self, source_views, model_name_or_path, batch_size=20):
        model = FactorSum._load_intrinsic_model(model_name_or_path)

        if isinstance(source_views, (list, tuple)):
            source_views = ["\n".join(view) for view in source_views]

        logger.debug(f"Generating summary views for {len(source_views)} source views")
        views = []

        for out in model(source_views, batch_size=batch_size, truncation=True):
            views.append(out["summary_text"])

        return views

    @staticmethod
    @memoize()
    def _get_document_views(source_sents, sample_factor, views_per_doc, seed):
        doc_views = get_document_views(
            source_sents,
            sample_factor=sample_factor,
            views_per_doc=views_per_doc,
            seed=seed,
        )
        return doc_views

    @staticmethod
    @memoize()
    def _find_best_summary(
        summary_views,
        target_budget,
        target_content,
        custom_guidance,
        min_words_per_view,
        sent_tokenize_fn,
        method,
        verbose,
    ):
        return find_best_summary(
            summary_views,
            target_budget,
            target_content=target_content,
            custom_guidance=custom_guidance,
            min_words_per_view=min_words_per_view,
            sent_tokenize_fn=sent_tokenize_fn,
            method=method,
            verbose=verbose,
        )

    @memoize()
    def generate_summary_views(
        self,
        source,
        batch_size=20,
        sample_factor=5,
        views_per_doc=20,
        min_words_per_view=5,
        sent_tokenize_fn=None,
        return_source_sents=False,
        seed=17,
    ):
        if sent_tokenize_fn is None:
            sent_tokenize_fn = sent_tokenize

        source_sents = sent_tokenize_fn(source, min_words=min_words_per_view)

        doc_views = FactorSum._get_document_views(
            source_sents,
            sample_factor=sample_factor,
            views_per_doc=views_per_doc,
            seed=seed,
        )

        views = self._get_summary_views(
            doc_views["source_views"], self.model_name_or_path, batch_size=batch_size
        )

        if return_source_sents:
            return views, source_sents

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
        sent_tokenize_fn=None,
        method="factorsum",
        verbose=False,
        seed=17,
    ):
        summary_views, source_sents = self.generate_summary_views(
            source,
            sample_factor=sample_factor,
            views_per_doc=views_per_doc,
            min_words_per_view=min_words_per_view,
            sent_tokenize_fn=sent_tokenize_fn,
            return_source_sents=True,
            seed=seed,
        )

        if target_content is None and source_target_budget:
            target_content = get_source_guidance(source_sents, source_target_budget)

        summary, guidance_scores = FactorSum._find_best_summary(
            summary_views,
            target_budget,
            target_content=target_content,
            custom_guidance=custom_guidance,
            min_words_per_view=min_words_per_view,
            sent_tokenize_fn=sent_tokenize_fn,
            method=method,
            verbose=verbose,
        )

        return summary, guidance_scores


def summarize(
    source,
    training_domain="arxiv",
    target=None,
    target_budget=None,
    source_token_budget=None,
    content_guidance_type=None,
    target_content=None,
    custom_guidance=None,
    sent_tokenize_fn=None,
    sample_factor=5,
    views_per_doc=20,
    min_words_per_view=5,
    verbose=True,
):
    if content_guidance_type == "source":
        if source_token_budget is None and target_budget:
            source_token_budget = target_budget
    else:
        source_token_budget = None

    if target and verbose:
        _log_reference_summary(target, sent_tokenize_fn=sent_tokenize_fn)
        logger.info("Generating summary")
        if target_budget:
            logger.info(f"Budget guidance: {target_budget}")
        if content_guidance_type:
            logger.info(f"Content guidance: {content_guidance_type}")
            logger.info(f"Source token budget: {source_token_budget}")

    model = FactorSum(training_domain)
    summary, guidance_scores = model.summarize(
        source,
        target_budget=target_budget,
        target_content=target_content,
        source_target_budget=source_token_budget,
        custom_guidance=custom_guidance,
        sample_factor=sample_factor,
        views_per_doc=views_per_doc,
        min_words_per_view=min_words_per_view,
        sent_tokenize_fn=sent_tokenize_fn,
        verbose=verbose,
    )

    if guidance_scores and verbose:
        logger.info(
            f"Summary words: {sum([len(nltk.word_tokenize(sent)) for sent in summary])}"
        )
        _print_guidance_scores(guidance_scores)
    _ = summarization_metrics(summary, target_summary=target, verbose=verbose)

    return summary, guidance_scores


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"), handlers=[RichHandler()]
    )
    logging.getLogger("absl").setLevel(logging.WARNING)
    fire.Fire(summarize)
