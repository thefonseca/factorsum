import logging

import fire

from factorsum.data import load_dataset, load_summaries
from factorsum.config import model_params
from .evaluation import evaluate
from .utils import get_sources, get_targets, config_logging, get_output_path

logger = logging.getLogger(__name__)


def evaluate_baseline(
    data_dir="data",
    max_samples=10000,
    dataset_name="arxiv",
    split="test",
    training_domain=None,
    output_dir=None,
    seed=17,
):
    timestr = config_logging(
        dataset_name, split, output_dir, training_domain=training_domain
    )

    params = model_params(dataset_name)
    eval_data = load_dataset(
        dataset_path=params["dataset_path"],
        dataset_name=dataset_name,
        split=split,
        data_dir=data_dir,
    )
    n_samples = len(get_sources(eval_data))
    targets = get_targets(eval_data)

    if training_domain is None:
        training_domain = dataset_name

    logger.info("Reference summary (sanity check)")

    evaluate(
        targets,
        targets,
        max_target_tokens=None,
        n_samples=max_samples,
        seed=seed,
    )

    for baseline in ["pegasus", "bigbird-pegasus-large", "bart-base", "bart-large"]:
        logger.info(f"Baseline: {baseline}")
        preds = load_summaries(
            dataset_name,
            split,
            baseline,
            training_domain,
            data_dir,
            expected_sample_count=n_samples,
        )
        if preds:
            # preds = [p.replace(".", " . ") for p in preds]
            save_to = get_output_path(
                output_dir,
                dataset_name,
                split,
                training_domain=training_domain,
                timestr=timestr,
                custom_suffix=baseline
            )
            evaluate(
                preds,
                targets,
                max_target_tokens=None,
                n_samples=max_samples,
                save_preds_to=save_to,
                seed=seed,
            )


if __name__ == "__main__":
    fire.Fire(evaluate_baseline)
