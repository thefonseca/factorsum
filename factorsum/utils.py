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
from optimum.bettertransformer import BetterTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline,
)

import torch

from .config import model_params, default_params

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
    # if ":v0" not in model_path:
    #     model_path = f"{model_path}:v0"
    model_path = Path(model_dir) / model_path
    return model_path


def get_model_info(model_type, training_domain=None):
    if training_domain:
        params = model_params(training_domain)
    else:
        params = default_params()

    model_id = params[f"{model_type}_model_id"]
    model_url = params[f"{model_type}_model_url"]
    return model_id, model_url


def _download_model(
    training_domain=None,
    model_dir="artifacts",
    model_type="intrinsic_importance",
    model_id=None,
    model_url=None,
):
    if model_id is None or model_url is None:
        model_id, model_url = get_model_info(
            model_type, training_domain=training_domain
        )
    model_path = get_model_path(model_type, model_id, model_dir)
    download_resource(model_url, f"{model_path}.zip")
    return model_path


def load_hf_model(
    model_path, model_type=None, pipeline_type=None, use_bettertransformer=True
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding=True, truncation=True)
    if model_type:
        model_type = model_type.replace("-", "_")

    if pipeline_type is None and model_type and model_type == "intrinsic_importance":
        pipeline_type = "summarization"

    if pipeline_type == "text-classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    if use_bettertransformer:
        try:
            model = BetterTransformer.transform(model, keep_original_model=True)
        except NotImplementedError:
            logger.warning(
                f"Model {model_path} not yet supported by BetterTransformer."
            )

    try:
        device = torch.cuda.current_device()
    except:
        device = None

    pipe = pipeline(pipeline_type, model=model, tokenizer=tokenizer, device=device)
    return pipe


def load_model(
    model_domain_or_path=None,
    model_type="intrinsic_importance",
    pipeline_type=None,
    model_id=None,
    model_url=None,
    model_dir="artifacts",
):
    if model_domain_or_path and Path(model_domain_or_path).exists():
        logger.info(f"Model found in path: {model_domain_or_path}")
        model_path = model_domain_or_path
    else:
        model_path = _download_model(
            model_domain_or_path,
            model_dir,
            model_type=model_type,
            model_id=model_id,
            model_url=model_url,
        )

    logger.info(f"Loading {model_type} model from {model_path}...")
    model = load_hf_model(model_path, model_type, pipeline_type=pipeline_type)
    return model, model_path


def apply_word_limit(text, max_words, return_list=False):
    if max_words is None:
        return text

    if type(text) == list:
        sents = text
    else:
        # nltk.word_tokenize drops \n information, so we need to do this sentence by sentence
        sents = sent_tokenize(text)

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


def clean_sentence(sent):
    sent = start_with_non_alpha_pattern.sub("", sent)
    # sent = end_with_non_alpha_pattern.sub(r"", sent)
    sent = sent.replace("\n", " ")
    sent = sent.strip()
    return sent


def fix_sent_tokenization(sents, min_words):
    """
    Apply heuristics to improve quality of sentence tokenization.
    Re-merge sentences resulting from common tokenization errors found empirically.
    """

    invalid_sents = [len(sent.split()) < min_words for sent in sents]
    # jj = 0
    while len(sents) > 1 and any(invalid_sents):
        new_sents = [sents[0]]
        mergeable = [False]

        for ii, sent in enumerate(sents[1:]):
            previous_sent = new_sents[-1].strip()

            prev_ends_with_ie = re.match(r".*i\s*\.\s*e\.$", previous_sent)
            prev_is_short_sentence = len(previous_sent.split()) < min_words
            # is_short_sentence = len(sent.split()) < min_words
            start_with_non_alpha = re.match(r"^([^\w]|\d)+.*", sent.strip())

            must_merge = (
                prev_ends_with_ie
                or prev_is_short_sentence
                # or is_short_sentence
                or start_with_non_alpha
            )
            mergeable.append(must_merge)

            if must_merge:
                new_sents[-1] = " ".join([previous_sent, sent])
            else:
                new_sents.append(sent)

        sents = new_sents

        # if debug:
        #     invalid_sents = [len(sent.split()) < min_words for sent in sents]
        #     # print(invalid_sents)
        #     # rich.print(sents)
        #     for ii in range(len(invalid_sents)):
        #         if invalid_sents[ii]:
        #             rich.print(ii-1, sents[ii-1])
        #             rich.print(ii, sents[ii])

        #     jj += 1
        #     if jj > 10:
        #         break

        invalid_sents = [
            len(sent.split()) < min_words and mergeable[ii]
            for ii, sent in enumerate(sents)
        ]

    return sents


def sent_tokenize(text, min_words=5):
    if type(text) == str:
        sents = nltk.sent_tokenize(text)
    else:
        sents = [s for x in text for s in nltk.sent_tokenize(x)]
    sents = fix_sent_tokenization(sents, min_words=min_words)
    sents = [clean_sentence(x) for x in sents if x != "\n"]
    return sents


def sent_tokenize_views(views, min_words=5, sent_tokenize_fn=None):
    sents = []

    if sent_tokenize_fn is None:
        sent_tokenize_fn = sent_tokenize

    for view in views:
        # try to put spaces after period to improve sentence tokenization
        view = re.sub(r"\.([\D]{2})", r" . \1", view)
        view_sents = sent_tokenize_fn(view, min_words=min_words)
        sents.extend(view_sents)

    return sents


def word_tokenize(text):
    if isinstance(text, list):
        text = " ".join(text)
    words = nltk.word_tokenize(text)
    return words


def log_summary(summary):
    info = [" ", " "]
    if type(summary) == list:
        for sent in summary:
            info.append(textwrap.fill(f"  - {sent}", 80))
    else:
        info.append(summary)
    logger.info("\n".join(info) + "\n", extra={"markup": True})


def log_guidance_scores(scores):
    info = ["Guidances scores:"]
    for key in scores.keys():
        if type(scores[key]) == dict:
            _scores = [f"{scores[key][x]:.3f}" for x in ["low", "mean", "high"]]
            _scores = ", ".join(_scores)
        else:
            _scores = f"{scores[key]:.3f}"
        info.append(f"{key}: {_scores}")
    logger.info("\n".join(info))


def log_reference_summary(target, sent_tokenize_fn=None):
    n_words = sum([len(nltk.word_tokenize(sent)) for sent in target])
    logger.info(f"Reference summary: ({n_words} words)")

    if type(target) == list:
        target = "\n".join(target)

    if sent_tokenize_fn:
        target = sent_tokenize_fn(target)
    else:
        target = sent_tokenize(target)

    log_summary(target)


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
