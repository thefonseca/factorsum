import os
import pickle
import logging
from pathlib import Path
from functools import lru_cache

import datasets
from tqdm import tqdm
import pandas as pd
import fire

from .utils import download_resource, sent_tokenize
from .oracle import get_oracles
from .sampling import sample_dataset
from .config import model_params

logger = logging.getLogger(__name__)


def _load_pickle(filename, data_dir):
    with open(os.path.join(data_dir, filename), "rb") as fh:
        return pickle.load(fh)


def _save_pickle(data, filename, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, filename), "wb") as fh:
        return pickle.dump(data, fh)


def _preprocess(
    dataset,
    source_key="article",
    target_key="abstract",
    min_words=10,
    oracle_type="rouge",
    include_oracles=False,
):
    """
    Preprocess a dataset with source/target texts.
    """

    sources = []
    targets = []
    # section_names = []

    def get_sentences(text):
        paragraphs = text.split("\n")
        sents = []

        for par in paragraphs:
            par_sents = sent_tokenize(par)
            sents.extend(par_sents)
        return sents

    for idx, item in tqdm(enumerate(dataset)):
        if len(item[source_key]) == 0:
            logger.warning(f"{idx}: empty article found!")
            logger.warning("Target text:")
            logger.warning(item[target_key])
            continue

        sources.append(get_sentences(item[source_key]))
        targets.append(get_sentences(item[target_key]))
        # s_names = item['section_names'].split('\n')
        # s_names = [s.strip() for s in s_names]
        # section_names.append(s_names)

    logger.info(f"Preprocessed {len(sources)} samples")
    data = {
        "sources": sources,
        "targets": targets,
        # 'section_names': section_names
    }

    if include_oracles:
        logger.info("Collecting oracles...")
        oracles = get_oracles(
            sources, targets, min_words=min_words, oracle_type=oracle_type
        )
        data["oracles"] = oracles

    return data


def _load_hf_dataset(dataset_path, dataset_name, cache_dir=None):
    if dataset_name in ["arxiv", "pubmed"]:
        dataset = datasets.load_dataset(
            "scientific_papers", dataset_name, cache_dir=cache_dir
        )
    elif dataset_name == "cnn":
        dataset = datasets.load_dataset(dataset_path, "3.0.0", cache_dir=cache_dir)
    else:
        dataset = datasets.load_dataset(dataset_path, cache_dir=cache_dir)

    return dataset


def _get_data_keys(dataset_name):
    if dataset_name == "govreport":
        source_key = "report"
        target_key = "summary"
    else:
        source_key = "article"
        target_key = "abstract"

    return source_key, target_key


def _create_views_dataset(
    dataset,
    save_to,
    views_per_doc=5,
    sample_factor=20,
    sample_fn=None,
    require_oracle=True,
):

    logger.info(f'Sampling document views for {len(dataset["sources"])} documents...')
    views_dataset, _ = sample_dataset(
        dataset,
        sample_factor=sample_factor,
        views_per_doc=views_per_doc,
        sample_fn=sample_fn,
        verbose=False,
        require_oracle=require_oracle,
    )

    save_to = save_to.format(sample_factor, views_per_doc)
    logger.info(f"Saving views dataset to {save_to}")

    with open(save_to, "wb") as f:
        pickle.dump(views_dataset, f)

    return dataset


def _data_split_filename(
    dataset_path,
    dataset_name,
    split,
    sample_type=None,
    sample_factor=None,
    views_per_doc=None,
):

    dataset_path = dataset_path.replace("/", "_")

    if sample_type and sample_factor and views_per_doc:
        filename = f"{dataset_name}-{sample_type}_k_{sample_factor}_samples_{views_per_doc}_{split}.pkl"
    elif sample_type and sample_factor:
        filename = f"{dataset_name}-{sample_type}_k_{sample_factor}_{split}.pkl"
    else:
        if dataset_name in ["arxiv", "pubmed"]:
            filename = f"{dataset_path}_{dataset_name}_{split}.pkl"
        else:
            filename = f"{dataset_path}_{split}.pkl"

    return filename


def _create_dataset(
    dataset_path,
    dataset_name,
    split=None,
    sample_type=None,
    sample_factor=None,
    views_per_doc=None,
    include_oracles=True,
    data_dir="data",
    cache_dir=None,
):

    splits = _get_splits(split)
    dataset = {}

    for _split in splits:
        filename = _data_split_filename(
            dataset_path,
            dataset_name,
            _split,
            sample_type=sample_type,
            sample_factor=sample_factor,
            views_per_doc=views_per_doc,
        )
        file_path = os.path.join(data_dir, filename)

        logger.info(f"Creating new dataset: {file_path}")

        if sample_type:
            # Load non-sampled dataset
            split_data = _load_dataset_split(
                dataset_path,
                dataset_name,
                _split,
                cache_dir=cache_dir,
                data_dir=data_dir,
                include_oracles=include_oracles,
            )

            is_train_split = _split == "train"
            split_data = _create_views_dataset(
                split_data,
                save_to=file_path,
                views_per_doc=views_per_doc,
                sample_factor=sample_factor,
                require_oracle=is_train_split,
            )
        else:
            data = _load_hf_dataset(dataset_path, dataset_name, cache_dir)
            source_key, target_key = _get_data_keys(dataset_name)
            split_data = _preprocess(
                data[_split],
                include_oracles=include_oracles,
                source_key=source_key,
                target_key=target_key,
            )
            _save_pickle(split_data, filename, data_dir)

        dataset[_split] = split_data

    if split:
        dataset = dataset[split]
    return dataset


def _load_dataset_split(
    dataset_path,
    dataset_name,
    split,
    sample_type=None,
    sample_factor=None,
    views_per_doc=None,
    cache_dir=None,
    data_dir="data",
    include_oracles=True,
):

    filename = _data_split_filename(
        dataset_path,
        dataset_name,
        split,
        sample_type=sample_type,
        sample_factor=sample_factor,
        views_per_doc=views_per_doc,
    )

    try:
        # try to load pre-processed file
        logger.info(f"Loading dataset from {filename}...")
        data = _load_pickle(filename, data_dir)
        # return data
    except:
        data = _create_dataset(
            dataset_path,
            dataset_name,
            split=split,
            sample_type=sample_type,
            sample_factor=sample_factor,
            views_per_doc=views_per_doc,
            include_oracles=include_oracles,
            data_dir=data_dir,
            cache_dir=cache_dir,
        )

    keys = list(data.keys())
    logger.info(f"Loaded dataset with {len(data[keys[0]])} samples and keys: {keys}")
    return data


def _views_dataset_split_to_dict(
    dataset,
    remove_empty_targets=True,
    max_samples=None,
    start_doc_id=None,
    end_doc_id=None,
):
    ds_dict = {"article": [], "abstract": [], "full_abstract": [], "doc_id": []}

    for item in zip(
        dataset["articles"],
        dataset["abstracts"],
        dataset["full_abstracts"],
        dataset["doc_ids"],
    ):

        article, abstract, full_abstract, doc_id = item

        if remove_empty_targets and (abstract is None or len(abstract) == 0):
            continue

        if article is None or len(article) == 0:
            continue

        if start_doc_id and doc_id < start_doc_id:
            continue

        if end_doc_id and doc_id > end_doc_id:
            continue

        ds_dict["doc_id"].append(doc_id)
        ds_dict["abstract"].append("\n".join(abstract))
        ds_dict["full_abstract"].append("\n".join(full_abstract))
        ds_dict["article"].append("\n".join(article))

        if max_samples and len(ds_dict["doc_id"]) == max_samples:
            break

    logger.info(f"Processed views data rows: {len(ds_dict['doc_id'])}")
    logger.info(f"Processed views unique doc ids: {max(ds_dict['doc_id']) + 1}")
    return ds_dict


def _dataset_split_to_dict(
    dataset,
    remove_empty_targets=True,
    start_doc_id=None,
    end_doc_id=None,
    max_samples=None,
):

    if "full_abstracts" in dataset:
        return _views_dataset_split_to_dict(
            dataset,
            max_samples=max_samples,
            remove_empty_targets=remove_empty_targets,
            start_doc_id=start_doc_id,
            end_doc_id=end_doc_id,
        )

    ds_dict = {
        "source": [],
        "target": [],
    }

    for source, target in zip(dataset["sources"], dataset["targets"]):

        if remove_empty_targets and (target is None or len(target) == 0):
            continue

        if source is None or len(source) == 0:
            continue

        ds_dict["target"].append("\n".join(target))
        ds_dict["source"].append("\n".join(source))

        if max_samples and len(ds_dict["source"]) == max_samples:
            break

    logger.info(f'Processed samples: {len(ds_dict["source"])}')
    return ds_dict


def _get_splits(splits=None):
    if splits is None:
        splits = ["train", "validation", "test"]
    elif type(splits) == str:
        splits = [splits]
    return splits


def _dataset_to_csv(
    dataset,
    save_to,
    split=None,
    remove_empty_abstracts=True,
    max_samples=None,
    start_doc_id=None,
    end_doc_id=None,
):

    if split and split not in dataset:
        dataset = {split: dataset}

    if split:
        splits = [split]
    else:
        splits = list(dataset.keys())

    for split in splits:
        ds_dict = _dataset_split_to_dict(
            dataset[split],
            max_samples=max_samples,
            remove_empty_targets=remove_empty_abstracts,
            start_doc_id=start_doc_id,
            end_doc_id=end_doc_id,
        )

        ds_df = pd.DataFrame(ds_dict)
        logger.info(f"Saving converted dataset to {save_to}")
        ds_df.to_csv(save_to, index=None)


def _maybe_fix_predictions(preds, expected_length, invalid_samples=None):
    if invalid_samples is None:
        invalid_samples = []

    if expected_length < len(preds):
        preds = [p for idx, p in enumerate(preds) if idx not in invalid_samples]
    assert expected_length == len(preds), f"{expected_length} != {len(preds)}"
    return preds


@lru_cache(maxsize=None)
def load_summaries(
    dataset_name,
    split,
    model_name,
    training_domain,
    data_dir,
    expected_sample_count=None,
):
    filename = (
        f"{dataset_name}-{split}-summary_preds-{model_name}-{training_domain}.pkl"
    )

    try:
        summaries = _load_pickle(filename, data_dir)
    except FileNotFoundError as err:
        logger.warning(f"File does not exist: {filename}")
        return

    if "pubmed-validation-summary_preds" in filename:
        # these pubmed samples from the validation set have empty articles
        invalid_samples = [2320, 4923]
    else:
        invalid_samples = []

    if expected_sample_count:
        summaries = _maybe_fix_predictions(
            summaries, expected_sample_count, invalid_samples=invalid_samples
        )
    logger.info(f"Loaded {len(summaries)} summaries from {filename}")
    return summaries


def load_summary_views(
    dataset_name,
    split,
    data_dir,
    training_domain=None,
    intrinsic_model_id=None,
):
    if training_domain is None:
        training_domain = dataset_name

    filename = f"{dataset_name}-{split}-summary_views-bart-{training_domain}-run={intrinsic_model_id}.pkl"
    summary_views = _load_pickle(filename, data_dir)
    logger.info(f"Loaded {len(summary_views)} summary views from {filename}")
    return summary_views


@lru_cache(maxsize=1)
def load_dataset(
    dataset_path,
    dataset_name,
    sample_type=None,
    sample_factor=None,
    views_per_doc=None,
    split=None,
    include_oracles=True,
    data_dir="data",
    cache_dir=None,
    seed=17,
):

    dataset = None
    splits = _get_splits(split)

    for _split in splits:
        split_data = _load_dataset_split(
            dataset_path,
            dataset_name,
            _split,
            sample_type=sample_type,
            sample_factor=sample_factor,
            views_per_doc=views_per_doc,
            include_oracles=include_oracles,
            data_dir=data_dir,
            cache_dir=cache_dir,
        )

        if dataset is None:
            # TODO: find a more elegant way to create an empty dataset with splits
            dataset = datasets.Dataset.from_dict({}).train_test_split(
                test_size=0.1, seed=seed
            )
            dataset[_split] = split_data
        else:
            dataset[_split] = split_data

    if split and len(splits) == 1:
        dataset = dataset[splits[0]]
    return dataset


def prepare_dataset(
    dataset_path,
    dataset_name,
    split=None,
    data_dir="data",
    sample_type=None,
    sample_factor=5,
    remove_empty_abstracts=True,
    views_per_doc=20,
    max_samples=None,
):

    splits = _get_splits(split)

    for split in splits:
        logger.info(f"Preparing {dataset_name} {split} set...")
        data_file = _data_split_filename(
            dataset_path,
            dataset_name,
            split,
            sample_type=sample_type,
            sample_factor=sample_factor,
            views_per_doc=views_per_doc,
        )
        data_file = data_file.replace(".pkl", ".csv")
        data_file = os.path.join(data_dir, data_file)

        if os.path.exists(data_file):
            logger.info(f"Data file already exists: {data_file}")
            logger.info("Skipping data preparation.")
            return

        dataset = load_dataset(
            dataset_path,
            dataset_name,
            split=split,
            data_dir=data_dir,
            sample_type=sample_type,
            sample_factor=sample_factor,
            views_per_doc=views_per_doc,
        )

        _dataset_to_csv(
            dataset,
            data_file,
            split=split,
            max_samples=max_samples,
            remove_empty_abstracts=remove_empty_abstracts,
        )


def download(dataset_name=None, data_dir="data"):

    if dataset_name is None:
        datasets = list(["arxiv", "punmed", "govreport"])
    else:
        datasets = [dataset_name]

    for dataset in datasets:
        params = model_params(dataset)
        url = params.get("dataset_url")
        local_path = Path(data_dir) / f"{dataset}.zip"
        if url:
            logger.info(f"Downloading {dataset} files from {url}...")
            download_resource(url, local_path)


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    fire.Fire()

    """
    Examples of CLI commands
    
    1. Generating CSV file for arXiv test set:
    > python -m factorsum.data prepare_dataset scientific_papers arxiv --split test

    2. Generating CSV file for the arXiv document/reference views (validation set):
    > python -m factorsum.data prepare_dataset scientific_papers arxiv --split validation \
        --sample_type random --sample_factor 5 --views_per_doc 20

    3. Generating CSV files for all arXiv splits:
    > python -m factorsum.data prepare_dataset scientific_papers arxiv
    """
