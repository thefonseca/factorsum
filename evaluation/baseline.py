import fire

from factorsum.data import load_dataset, load_summaries
from factorsum.config import model_params
from .evaluation import evaluate
from .utils import get_sources, get_targets


def evaluate_baseline(
    data_dir="data",
    max_samples=10000,
    dataset_name="arxiv",
    split="test",
    training_domain=None,
    seed=17,
):

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

    print(">> Reference summary (sanity check)")

    evaluate(
        targets,
        targets,
        max_target_tokens=None,
        n_samples=max_samples,
        seed=seed,
    )

    for baseline in ["pegasus", "bigbird-pegasus-large", "bart-base", "bart-large"]:
        print(f">> {baseline}")
        preds = load_summaries(
            dataset_name,
            split,
            baseline,
            training_domain,
            data_dir,
            expected_sample_count=n_samples,
        )
        if preds:
            preds = [p.replace(".", " . ") for p in preds]
            evaluate(
                preds,
                targets,
                max_target_tokens=None,
                n_samples=max_samples,
                seed=seed,
            )


if __name__ == "__main__":
    fire.Fire(evaluate_baseline)
