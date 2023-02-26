import itertools
import logging
import os

import fire
from p_tqdm import p_map

from .utils import (
    get_output_path,
    config_logging,
    get_progress_bar,
    add_progress_task
)
from .evaluation import evaluate as evaluate_summaries
from .utils import compute_metric
from factorsum.config import model_params
from factorsum.data import load_dataset, load_summaries
from factorsum.model import summarize, FactorSum
from factorsum.utils import word_tokenize

logger = logging.getLogger(__name__)


def default_eval_args(params, max_samples, seed):
    default_kwargs = dict(
        strict_budget=False,
        oracle_budget=False,
        max_target_tokens=None,
        # content_weight=params["content_weight"],
        min_words_per_view=params["min_words_per_view"],
        n_samples=max_samples,
        text_guidance=None,
        seed=seed,
    )
    return default_kwargs


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
            "no",
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


def evaluate_all(
    data_dir="data",
    max_samples=10000,
    dataset_name="arxiv",
    split="test",
    content_types=None,
    budget_types=None,
    training_domain=None,
    source_token_budget=None,
    token_budget=None,
    intrinsic_model_id=None,
    views_per_doc=None,
    sample_factor=None,
    min_words_per_view=None,
    method="factorsum",
    cache_dir=None,
    output_dir=None,
    seed=17,
):
    timestr = config_logging(
        dataset_name, split, output_dir, training_domain=training_domain
    )

    content_types, allowed_content_types = get_content_types(method, content_types)
    budget_types, allowed_budget_types = get_budget_types(method, budget_types)

    for content_type, budget_type in itertools.product(content_types, budget_types):
        assert content_type in allowed_content_types, content_type
        assert budget_type in allowed_budget_types, budget_type

        evaluate(
            max_samples=max_samples,
            data_dir=data_dir,
            dataset_name=dataset_name,
            split=split,
            training_domain=training_domain,
            source_token_budget=source_token_budget,
            token_budget=token_budget,
            intrinsic_model_id=intrinsic_model_id,
            content_type=content_type,
            budget_type=budget_type,
            method=method,
            views_per_doc=views_per_doc,
            sample_factor=sample_factor,
            min_words_per_view=min_words_per_view,
            cache_dir=cache_dir,
            output_dir=output_dir,
            verbose=False,
            timestr=timestr,
            seed=seed,
        )


def _get_source_budget(source_budget, target_budget, content_type):
    if content_type == "source":
        if source_budget is None and target_budget:
            source_budget = target_budget
    else:
        source_budget = None
    return source_budget


def _get_target_budget(
    budget_type,
    budget_value,
    doc_id,
    sources,
    target,
    dataset_name,
    split,
    training_domain,
    data_dir,
):
    target_budget = None

    if budget_type == "oracle":
        target_budget = len(word_tokenize(target))

    elif budget_type == "fixed":
        target_budget = budget_value

    elif budget_type and budget_type not in ["no", "source"]:
        summaries = load_summaries(
            dataset_name,
            split,
            budget_type,
            training_domain,
            data_dir,
            expected_sample_count=len(sources),
        )

        if summaries is not None:
            target_budget = len(word_tokenize(summaries[doc_id]))

    return target_budget


def _get_target_content(
    doc_id,
    sources,
    target,
    content_type,
    dataset_name,
    split,
    training_domain,
    data_dir,
):
    target_content = None

    if content_type and content_type == "oracle":
        target_content = target

    elif content_type and content_type not in ["no", "source"]:
        target_content = load_summaries(
            dataset_name,
            split,
            content_type,
            training_domain,
            data_dir,
            expected_sample_count=len(sources),
        )

        if target_content is None:
            return

        target_content = target_content[doc_id]

    return target_content


def _get_custom_guidance(guidance, doc_id):
    custom_guidance = guidance
    if guidance and type(guidance) == list:
        custom_guidance = guidance[doc_id]
    return custom_guidance


def summarize_job(
    source,
    training_domain,
    target_budget,
    target_content,
    source_budget,
    content_guidance_type,
    custom_guidance,
    sample_factor,
    views_per_doc,
    min_words_per_view,
    sent_tokenize_fn,
    model_id=None,
    model_url=None,
    target=None,
    verbose=False,
):
    summary, guidance_scores = summarize(
        source,
        training_domain,
        model_id=model_id,
        model_url=model_url,
        target=target,
        target_budget=target_budget,
        source_token_budget=source_budget,
        target_content=target_content,
        content_guidance_type=content_guidance_type,
        custom_guidance=custom_guidance,
        sample_factor=sample_factor,
        views_per_doc=views_per_doc,
        min_words_per_view=min_words_per_view,
        sent_tokenize_fn=sent_tokenize_fn,
        verbose=verbose,
    )

    summary = "\n".join(summary)
    return summary, guidance_scores


def evaluate(
    doc_id=None,
    max_samples=10000,
    dataset=None,
    data_dir="data",
    dataset_name="arxiv",
    split="test",
    training_domain=None,
    source_token_budget=None,
    token_budget=None,
    intrinsic_model_id=None,
    content_type=None,
    budget_type=None,
    custom_guidance=None,
    custom_metrics=None,
    method="factorsum",
    views_per_doc=20,
    sample_factor=5,
    min_words_per_view=5,
    sent_tokenize_fn=None,
    model_kwargs=None,
    cache_dir=None,
    output_dir=None,
    verbose=None,
    timestr=None,
    progress=None,
    seed=17,
):
    if timestr is None:
        timestr = config_logging(
            dataset_name, split, output_dir, training_domain=training_domain
        )

    if model_kwargs is None:
        model_kwargs = {}

    params = model_params(
        dataset_name,
        token_budget=token_budget,
        source_token_budget=source_token_budget,
        views_per_doc=views_per_doc,
        sample_factor=sample_factor,
        intrinsic_model_id=intrinsic_model_id,
        min_words_per_view=min_words_per_view,
        **model_kwargs
    )

    if dataset is None:
        dataset = load_dataset(
            dataset_path=params["dataset_path"],
            dataset_name=dataset_name,
            split=split,
            data_dir=data_dir,
            cache_dir=cache_dir,
        )

    if training_domain is None:
        training_domain = dataset_name

    token_budget = params["token_budget"]
    source_token_budget = params.get("source_token_budget")

    if doc_id is None:
        doc_ids = range(len(dataset["sources"]))
    elif isinstance(doc_id, int):
        doc_ids = [doc_id]
    else:
        doc_ids = doc_id

    doc_ids = doc_ids[:max_samples]
    if verbose is None and len(doc_ids) == 1:
        verbose = True

    model = FactorSum(training_domain, 
                      model_id=params.get('intrinsic_importance_model_id'),
                      model_url=params.get('intrinsic_importance_model_url'))
    target_contents = []
    target_budgets = []
    source_budgets = []
    sources = []
    targets = []
    custom_guidances = []
    summary_views = []

    logger.info(f"Summarization method: {method}")
    if budget_type:
        logger.info(f"Budget guidance: {budget_type}")
    if content_type:
        logger.info(f"Content guidance: {content_type}")

    if progress is None:
        progress = get_progress_bar()

    views_task = add_progress_task(
        progress, "Generating summary views...", total=len(doc_ids), existing_ok=False
    )

    with progress:
        for idx, doc_id in enumerate(doc_ids):
            source = dataset["sources"][doc_id]
            target = dataset["targets"][doc_id]

            target_content = _get_target_content(
                doc_id,
                dataset["sources"],
                target,
                content_type,
                dataset_name,
                split,
                training_domain,
                data_dir,
            )
            if (
                content_type
                and content_type not in ["no", "source"]
                and target_content is None
            ):
                logger.warning(
                    f"Skipping evaluation: target content is empty ({content_type})"
                )
                break

            target_budget = _get_target_budget(
                budget_type,
                token_budget,
                doc_id,
                dataset["sources"],
                target,
                dataset_name,
                split,
                training_domain,
                data_dir,
            )
            if (
                target_budget is None
                and budget_type
                and budget_type not in ["fixed", "oracle", "no"]
            ):
                logger.warning(
                    f"Skipping evaluation: target budget is empty ({budget_type})"
                )
                break

            source_budget = _get_source_budget(
                source_token_budget, target_budget, content_type
            )

            views = model.generate_summary_views(
                source,
                sample_factor=params["sample_factor"],
                views_per_doc=params["views_per_doc"],
                min_words_per_view=params["min_words_per_view"],
                sent_tokenize_fn=sent_tokenize_fn,
                seed=seed,
            )

            guidance = _get_custom_guidance(custom_guidance, idx)
            custom_guidances.append(guidance)
            target_contents.append(target_content)
            target_budgets.append(target_budget)
            source_budgets.append(source_budget)
            sources.append(source)
            targets.append(target)
            summary_views.append(views)
            progress.update(views_task, advance=1)

    if len(sources) == 0:
        return

    if len(custom_guidances) > 1 and len(custom_guidances) != len(doc_ids):
        raise ValueError(
            f"Number of guidance objects is inconsistent with number of "
            f"samples: {len(custom_guidance)} != {len(doc_ids)}"
        )

    # if progress is None:
    #     progress = get_progress_bar()

    guidance_task = add_progress_task(
        progress, "Pre-computing guidance for views...", total=len(summary_views), existing_ok=False
    )
    with progress:
        for idx, view_sentences in enumerate(summary_views):
            guidances = FactorSum.get_guidance(
                target_content=target_contents[idx],
                target_budget=target_budgets[idx],
                custom_guidance=custom_guidances[idx],
            )
            view_sentences = FactorSum.preprocess_summary_views(
                view_sentences, params["min_words_per_view"]
            )
            for guidance in guidances:
                if not guidance.parallelizable:
                    [guidance.score([s]) for s in view_sentences]

            progress.update(guidance_task, advance=1)

    # In this summarization pass, FactorSum will use memoized
    # results from previous steps, so only the extrinsic optimization
    # computation will be performed. This multi-step process is necessary
    # as the seq2seq generation cannot be parallelized via p_map.
    logger.info("Generating summaries from summary views...")
    results = p_map(
        lambda source, target, target_budget, source_budget, target_content, guidance: summarize_job(
            source,
            training_domain,
            model_id=model.model_id,
            model_url=model.model_url,
            target_budget=target_budget,
            target_content=target_content,
            source_budget=source_budget,
            content_guidance_type=content_type,
            custom_guidance=guidance,
            sample_factor=params["sample_factor"],
            views_per_doc=params["views_per_doc"],
            min_words_per_view=params["min_words_per_view"],
            sent_tokenize_fn=sent_tokenize_fn,
            target=target,
            verbose=verbose,
        ),
        sources,
        targets,
        target_budgets,
        source_budgets,
        target_contents,
        custom_guidances,
    )

    summaries = [r[0] for r in results]
    guidance_scores = [r[1] for r in results]

    save_to = get_output_path(
        output_dir,
        dataset_name,
        split,
        content_type,
        budget_type,
        training_domain=training_domain,
        timestr=timestr,
    )

    scores = {'guidance_scores': guidance_scores}
    if custom_metrics:
        for metric_key, metric_fn in custom_metrics.items():
            concept_scores = compute_metric(targets, 
                                            summaries, 
                                            metric_fn, 
                                            progress=progress,
                                            min_words=params['min_words_per_view'])
            scores[metric_key] = concept_scores

    evaluate_summaries(
        summaries,
        targets,
        scores=scores,
        n_samples=max_samples,
        save_preds_to=save_to,
        seed=seed,
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fire.Fire()
