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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

from .config import model_params

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)

start_with_non_alpha_pattern = re.compile("^[^\w_\s]+\s")


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


def _download_intrinsic_model(training_domain, model_dir):
    params = model_params(training_domain)
    model_id = params["intrinsic_model_id"]
    model_path = f"model-{model_id}:v0"
    model_path = Path(model_dir) / model_path
    download_resource(params["google_drive_id"], f"{model_path}.zip")
    return model_path


def load_intrinsic_model(model_name_or_path, model_dir="artifacts"):

    if Path(model_name_or_path).exists():
        model_path = model_name_or_path
    else:
        model_path = _download_intrinsic_model(model_name_or_path, model_dir)

    logger.info(f"Loading intrinsic importance model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    try:
        device = torch.cuda.current_device()
    except:
        device = None
    summarizer = pipeline(
        "summarization", model=model, tokenizer=tokenizer, device=device
    )

    return summarizer


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


def sent_tokenize_views(views, view_idxs, summary_idxs=None):
    sents = []
    sent_idxs = []
    sent_to_view_idxs = []

    for idx in view_idxs:
        view = views[idx]
        if summary_idxs is not None and idx in summary_idxs:
            continue

        # try to put spaces after period to improve sentence tokenization
        view = re.sub(r"\.([\D]{2})", r" . \1", view)
        view_sents = nltk.sent_tokenize(view)
        sents.extend(view_sents)

        sent_to_view_idxs.extend([idx] * len(view_sents))
        sent_idxs.extend(range(len(sent_idxs), len(sent_idxs) + len(view_sents)))

    def clean_sent(sent):
        sent = start_with_non_alpha_pattern.sub("", sent)
        sent = sent.replace(" .", ".")
        return sent

    sents = [clean_sent(s) for s in sents]

    return sents, sent_idxs, sent_to_view_idxs


def show_summary(summary):
    if type(summary) == list:
        for sent in summary:
            print(textwrap.fill(f"- {sent}", 80))
    else:
        print(summary)


def download_models(training_domain=None, model_dir="artifacts"):

    if training_domain is None:
        training_domains = ["arxiv", "pubmed", "govreport"]
    else:
        training_domains = [training_domain]

    for domain in training_domains:
        _download_intrinsic_model(domain, model_dir)


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    fire.Fire()

    """
    To call download_wandb_model(project, model_name) from CLI use:
    > python -m factorsum.utils download_wandb_model <project> <model_name>
    """
