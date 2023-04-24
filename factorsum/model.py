import logging
import os
from functools import lru_cache

import fire
import nltk
from rich.logging import RichHandler

from .constraints import BudgetConstraint, RedundancyConstraint, SummaryViewConstraint
from .extrinsic import greedy_summary, textrank_summary
from .guidance import ROUGEContentGuidance
from .memoizer import memoize
from .metrics import summarization_metrics
from .oracle import get_oracles
from .sampling import get_document_views
from .utils import (
    load_model,
    log_summary,
    log_reference_summary,
    log_guidance_scores,
    sent_tokenize,
    word_tokenize,
    sent_tokenize_views,
)

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
        guidance_tokens += len(word_tokenize(sent))
        if guidance_tokens > token_budget:
            break

    if verbose:
        logger.info(f"Source guidance token budget: {token_budget}")
        logger.info(f"Tokens in source guidance: {guidance_tokens}")
        logger.info(f"Sentences in source guidance: {len(source_guidance)}")
    return source_guidance


class FactorSum:
    def __init__(self, model_name_or_path, model_id=None, model_url=None):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model_id = model_id
        self.model_url = model_url

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_intrinsic_model(model_domain_or_path, model_id, model_url=None):
        model, _ = load_model(
            model_domain_or_path=model_domain_or_path,
            model_type="intrinsic_importance",
            model_id=model_id,
            model_url=model_url,
        )
        return model

    @staticmethod
    @lru_cache(maxsize=None)
    @memoize()
    def _get_summary_views(source_views, model_name_or_path, model_id, model_url=None):
        model = FactorSum._load_intrinsic_model(
            model_name_or_path, model_id=model_id, model_url=model_url
        )

        logger.debug(f"Generating summary views for {len(source_views)} source views")
        views = []

        if isinstance(source_views, tuple):
            source_views = list(source_views)

        for out in model(source_views, batch_size=len(source_views), truncation=True):
            views.append(out["summary_text"])
        return views

    @staticmethod
    @lru_cache(maxsize=100000)
    def _get_document_views(source_sents, sample_factor, views_per_doc, seed):
        doc_views = get_document_views(
            source_sents,
            sample_factor=sample_factor,
            views_per_doc=views_per_doc,
            seed=seed,
        )
        return doc_views

    @staticmethod
    def _get_valid_views(views, min_words=5):
        n_ignored = sum([p is None for p in views])
        if n_ignored > 0:
            logger.warning(f"Ignoring {n_ignored} views")
            views = [p for p in views if p is not None]

        _views = []
        constraint = SummaryViewConstraint(min_length=min_words)

        for view in views:
            if not constraint.check(view):
                continue
            if view not in _views:
                _views.append(view)

        return _views

    @staticmethod
    def _get_oracle_idxs(source_sents, target_sents):
        oracle_idxs = get_oracles([source_sents], [target_sents], progress_bar=False)[0]
        oracle_idxs = [x if x is not None else len(oracle_idxs) for x in oracle_idxs]
        return oracle_idxs

    @staticmethod
    def _reorder_sentences(summary, target_content):
        _target_content = target_content.replace("<n>", "")
        _target_content = sent_tokenize(_target_content)
        oracle_idxs = FactorSum._get_oracle_idxs(_target_content, summary)
        summary = [x for _, x in sorted(zip(oracle_idxs, summary))]
        return summary

    @staticmethod
    def find_best_summary(
        summary_views,
        target_budget,
        constraints,
        target_content,
        guidance,
        min_words_per_view,
        sent_tokenize_fn,
        strict_budget,
        method,
    ):
        summary_views = FactorSum.preprocess_summary_views(
            summary_views, min_words_per_view, sent_tokenize_fn
        )

        if method == "factorsum":
            summary = greedy_summary(summary_views, guidance, constraints)
        elif method == "textrank":
            summary = textrank_summary(summary_views, target_budget)
        else:
            raise ValueError(f"Unsupported summarization method: {method}")

        if strict_budget:
            summary = FactorSum._apply_word_limit(
                summary, target_budget, return_list=True
            )

        if target_content:
            summary = FactorSum._reorder_sentences(summary, target_content)

        guidance_scores = {}
        if guidance:
            guidance_scores = {g.__class__.__name__: g.score(summary) for g in guidance}

        return summary, guidance_scores

    @staticmethod
    def get_guidance(
        target_content=None,
        target_budget=None,
        content_weight=1.0,
        content_score_key=None,
        custom_guidance=None,
    ):
        guidance = []

        if content_score_key is None:
            if target_budget and target_budget > 0:
                content_score_key = "recall"
            else:
                content_score_key = "fmeasure"

        if target_content:
            guidance.append(
                ROUGEContentGuidance(
                    target_content, weight=content_weight, score_key=content_score_key
                )
            )

        if custom_guidance:
            if type(custom_guidance) == list:
                guidance.extend(custom_guidance)
            else:
                guidance.append(custom_guidance)

        return guidance

    @staticmethod
    def get_constraints(target_budget=None, custom_constraints=None):
        constraints = [RedundancyConstraint()]

        if target_budget and target_budget > 0:
            constraints.append(BudgetConstraint(target_budget))

        if custom_constraints:
            if type(custom_constraints) == list:
                constraints.extend(custom_constraints)
            else:
                constraints.append(custom_constraints)
        return constraints

    def generate_summary_views(
        self,
        source,
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
            tuple(source_sents),
            sample_factor=sample_factor,
            views_per_doc=views_per_doc,
            seed=seed,
        )

        if isinstance(doc_views["source_views"], (list, tuple)):
            source_views = tuple(
                ["\n".join(view) for view in doc_views["source_views"]]
            )

        views = FactorSum._get_summary_views(
            source_views,
            self.model_name_or_path,
            self.model_id,
            model_url=self.model_url,
        )

        if return_source_sents:
            return views, source_sents

        return views

    @staticmethod
    def preprocess_summary_views(views, min_words_per_view, sent_tokenize_fn=None):
        views = FactorSum._get_valid_views(views, min_words=min_words_per_view)
        # sent tokenize all preds to get more fine-grained information
        if sent_tokenize_fn is None:
            sent_tokenize_fn = sent_tokenize
        views = sent_tokenize_views(
            views, min_words=min_words_per_view, sent_tokenize_fn=sent_tokenize_fn
        )
        return views

    def summarize(
        self,
        source,
        target_budget,
        summary_views=None,
        source_target_budget=0,
        target_content=None,
        custom_guidance=None,
        custom_constraints=None,
        content_weight=1.0,
        sample_factor=5,
        views_per_doc=20,
        min_words_per_view=5,
        sent_tokenize_fn=None,
        strict_budget=False,
        method="factorsum",
        verbose=False,
        seed=17,
    ):
        if sent_tokenize_fn is None:
            sent_tokenize_fn = sent_tokenize

        if summary_views:
            source_sents = sent_tokenize_fn(source, min_words=min_words_per_view)
        else:
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

        if target_content is not None:
            if type(target_content) == list:
                target_content = "\n".join(target_content)

        guidance = FactorSum.get_guidance(
            target_content=target_content,
            target_budget=target_budget,
            content_weight=content_weight,
            custom_guidance=custom_guidance,
        )

        constraints = FactorSum.get_constraints(
            target_budget=target_budget, custom_constraints=custom_constraints
        )

        summary, guidance_scores = FactorSum.find_best_summary(
            summary_views,
            target_budget,
            constraints,
            guidance=guidance,
            target_content=target_content,
            min_words_per_view=min_words_per_view,
            sent_tokenize_fn=sent_tokenize_fn,
            strict_budget=strict_budget,
            method=method,
        )

        if verbose:
            log_summary(summary)

        return summary, guidance_scores


def summarize(
    source,
    training_domain="arxiv",
    model_id=None,
    model_url=None,
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
        log_reference_summary(target, sent_tokenize_fn=sent_tokenize_fn)
        logger.info("Generating summary")
        if target_budget:
            logger.info(f"Budget guidance: {target_budget}")
        if content_guidance_type:
            logger.info(f"Content guidance: {content_guidance_type}")
            logger.info(f"Source token budget: {source_token_budget}")

    model = FactorSum(training_domain, model_id=model_id, model_url=model_url)

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
        log_guidance_scores(guidance_scores)
    _ = summarization_metrics(summary, target_summary=target, verbose=verbose)

    return summary, guidance_scores


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"), handlers=[RichHandler()]
    )
    logging.getLogger("absl").setLevel(logging.WARNING)
    fire.Fire(summarize)
