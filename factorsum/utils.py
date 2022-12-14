from pathlib import Path
from urllib.parse import urlparse
import logging
import textwrap
import os
import re
from zipfile import ZipFile

import requests
import wandb
import fire
import nltk
import gdown
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline,
)
import torch
from rich.text import Text

from .config import model_params

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)

start_with_non_alpha_pattern = re.compile("^(_|[^\w\@])*\s")
end_with_non_alpha_pattern = re.compile("\s(_|[^\w\@])*$")


def download_wandb_model(project, model_name):

    wandb.login()

    # Create a new run
    with wandb.init(project=project) as run:

        # Connect an Artifact to the run
        my_model_name = model_name
        my_model_artifact = run.use_artifact(my_model_name)

        # Download model weights to a folder and return the path
        model_dir = my_model_artifact.download()
        return model_dir


def is_url(url):
    return urlparse(url).scheme in ("http", "https")


def download_resource(url, local_path, extract_zip=True):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if url and not local_path.exists():
        logger.info(f"Model not found in path: {local_path}")

        if is_url(url):
            logger.info(f"Downloading resource from {url}...")
            r = requests.get(url)
            with open(local_path, "wb") as f:
                f.write(r.content)
        else:
            logger.info(f"Downloading resource {url} from Google Drive...")
            # if it is not a URL, assume it is a Google Drive ID
            gdown.download(id=url, output=str(local_path), quiet=True)

        if local_path.exists() and local_path.suffix == ".zip" and extract_zip:
            extract_folder = local_path.parent
            logger.info(f"Extracting {local_path} to {extract_folder}...")
            with ZipFile(local_path, "r") as zip_ref:
                zip_ref.extractall(extract_folder)


def get_model_path(model_type, model_id, model_dir="artifacts"):
    model_type = model_type.replace("-", "_")
    model_path = f"model-{model_id}"
    if ":v0" not in model_path:
        model_path = f"{model_path}:v0"
    model_path = Path(model_dir) / model_path
    return model_path


def _download_model(
    training_domain, model_dir, model_type="intrinsic_importance", params=None
):
    if params is None:
        params = model_params(training_domain)

    model_id = params[f"{model_type}_model_id"]
    model_path = get_model_path(model_type, model_id, model_dir)

    download_resource(params[f"{model_type}_model_url"], f"{model_path}.zip")
    return model_path


def load_model(
    model_domain_or_path,
    model_type="intrinsic_importance",
    model_dir="artifacts",
    params=None,
):
    if Path(model_domain_or_path).exists():
        logger.info(f"Model found in path: {model_domain_or_path}")
        model_path = model_domain_or_path
    else:
        model_path = _download_model(
            model_domain_or_path, model_dir, model_type=model_type, params=params
        )

    logger.info(f"Loading {model_type} model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True)

    model_type = model_type.replace("-", "_")
    pipeline_type = "summarization"
    if params:
        pipeline_type = params.get(f"{model_type}_pipeline_type", "summarization")

    if pipeline_type == "text-classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    try:
        device = torch.cuda.current_device()
    except:
        device = None

    summarizer = pipeline(
        pipeline_type, model=model, tokenizer=tokenizer, device=device
    )

    return summarizer, model_path


def apply_word_limit(text, max_words, return_list=False):
    if max_words is None:
        return text

    if type(text) == list:
        sents = text
    else:
        # nltk.word_tokenize drops \n information, so we need to do this sentence by sentence
        sents = nltk.sent_tokenize(text)

    truncated_sents = []
    total_words = 0

    for sent in sents:
        sent_words = nltk.word_tokenize(sent)
        if len(sent_words) + total_words >= max_words:
            words_to_keep = max_words - total_words
            sent_words = sent_words[:words_to_keep]

        truncated_sents.append(" ".join(sent_words))
        total_words += len(sent_words)

        if total_words >= max_words:
            break

    if return_list:
        return truncated_sents

    return "\n".join(truncated_sents)


def clean_summary_view(sent):
    sent = start_with_non_alpha_pattern.sub("", sent)
    sent = end_with_non_alpha_pattern.sub("", sent)
    sent = sent.replace("\n", " ")
    sent = sent.strip()
    return sent


def sent_tokenize_views(views, summary=None, min_words=5):
    sents = []

    for view in views:
        if summary is not None and view in summary:
            continue

        # try to put spaces after period to improve sentence tokenization
        view = re.sub(r"\.([\D]{2})", r" . \1", view)
        view_sents = nltk.sent_tokenize(view)
        for sent in view_sents:
            # avoid tokenization if any of the sentences is too short
            sent = clean_summary_view(sent)
            if len(sent.split()) < min_words:
                view_sents = [view]
                break

        sents.extend(view_sents)

    sents = [clean_summary_view(s) for s in sents]
    return sents


def show_summary(summary):
    info = [" ", " "]
    if type(summary) == list:
        for sent in summary:
            info.append(textwrap.fill(f"  - {sent}", 80))
    else:
        info.append(summary)
    logger.info("\n".join(info) + "\n", extra={"markup": True})


def download_models(training_domain=None, model_dir="artifacts"):

    if training_domain is None:
        training_domains = ["arxiv", "pubmed", "govreport"]
    else:
        training_domains = [training_domain]

    for domain in training_domains:
        _download_model(domain, model_dir)


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    fire.Fire()

    """
    To call download_wandb_model(project, model_name) from CLI use:
    > python -m factorsum.utils download_wandb_model <project> <model_name>
    """
