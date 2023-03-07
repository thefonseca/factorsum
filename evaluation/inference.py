from functools import lru_cache
import logging

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
)

from factorsum.memoizer import memoize
from .utils import (
    get_progress_bar,
    add_progress_task,
    sent_tokenize,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_tokenizer(model_name_or_path, cache_dir=None):
    logger.info(f"Loading tokenizer {model_name_or_path}...")
    if "google/pegasus-x-base" in model_name_or_path:
        model_name_or_path = "google/pegasus-x-base"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False, cache_dir=cache_dir
    )
    return tokenizer


@lru_cache(maxsize=1)
def load_model(model_name_or_path, cache_dir=None):
    logger.info(f"Loading model {model_name_or_path}...")
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path, config=config, cache_dir=cache_dir
    )
    try:
        device = torch.cuda.current_device()
        model = model.to(device)
    except:
        logger.warning("Failed to get cuda device")

    tokenizer = load_tokenizer(model_name_or_path)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )
    model.eval()
    return model, tokenizer


@memoize()
def predict_summary(
    model_name_or_path,
    text,
    max_source_length=None,
    truncation=True,
    memoizer_ignore_cache=False,
    **generation_kwargs,
):
    model, tokenizer = load_model(model_name_or_path)
    if isinstance(text, list):
        text = "\n".join(text)

    if (
        max_source_length is None
        and "google/bigbird-pegasus-large" in model_name_or_path
    ):
        max_source_length = 3072
    elif max_source_length is None and hasattr(model.config, "max_position_embeddings"):
        max_source_length = model.config.max_position_embeddings
    elif max_source_length is None and hasattr(
        model.config, "max_encoder_position_embeddings"
    ):
        max_source_length = model.config.max_encoder_position_embeddings
    elif max_source_length is None:
        max_source_length = 1024

    generation_config = GenerationConfig.from_pretrained(
        model_name_or_path, **generation_kwargs
    )

    with torch.no_grad():
        inputs = tokenizer(
            [text],
            padding="max_length",
            max_length=max_source_length,
            truncation=truncation,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        # print(len(inputs["input_ids"][0]))
        preds = model.generate(inputs["input_ids"], generation_config)
        summary = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return summary


def postprocess_summary(model_name_or_path, summary):
    # special newline postprocessing for some models
    if any(
        [x in model_name_or_path] for x in ["google/pegasus", "google/bigbird-pegasus"]
    ):
        summary = summary.replace(".<n> ", ".\n ")

    # rougeLSum expects newline after each sentence
    summary = "\n".join([s.strip() for s in sent_tokenize(summary)])
    return summary


def predict_summaries(
    model_name_or_path, sources, max_length=256, cache_start=0, cache_end=None
):
    summaries = []
    progress = get_progress_bar()
    task = add_progress_task(
        progress,
        f"Generating summaries for {model_name_or_path}...",
        total=len(sources),
        existing_ok=False,
    )
    cache_end = cache_end if cache_end is not None else len(sources)

    with progress:
        for idx, text in enumerate(sources):
            ignore_cache = idx < cache_start or idx >= cache_end
            summary = predict_summary(
                model_name_or_path,
                text,
                max_length=max_length,
                memoizer_ignore_cache=ignore_cache,
            )
            summary = postprocess_summary(model_name_or_path, summary)
            summaries.append(summary)
            progress.update(task, advance=1)

    return summaries
