# FactorSum

Supporting code for the paper [Factorizing Content and Budget Decisions in Abstractive Summarization of Long Documents](https://arxiv.org/abs/2205.12486).

## Abstract

We argue that disentangling content selection from the budget used to cover salient content improves the performance and applicability of abstractive summarizers. Our method, FactorSum, does this disentanglement by factorizing summarization into two steps through an energy function: 

1. Intrinsic importance model: generation of abstractive summary views.
2. Extrinsic importance model: combination of these views into a final summary, following a budget and content guidance. 

This extrinsic guidance may come from different sources, including from an advisor model such as [BART](https://arxiv.org/abs/1910.13461) or [BigBird](https://arxiv.org/abs/2007.14062), or in oracle mode -- from the reference. This factorization achieves significantly higher ROUGE scores on multiple benchmarks for long document summarization, namely PubMed, arXiv, and GovReport. Most notably, our model is effective for domain adaptation. When trained only on PubMed samples, it achieves a 46.29 ROUGE-1 score on arXiv, which indicates a strong performance due to more flexible budget adaptation and content selection less dependent on domain-specific textual structure.

## Getting started
Clone this repository and install the dependencies:
```bash
git clone https://github.com/thefonseca/factorsum.git
cd factorsum
# Optional: checkout the arXiv version 2205.12486v2 for reproducibility
git checkout 2205.12486v2
# Install dependencies
pip install -r requirements.txt
```

## Usage
Example: summarizing a single document using a budget guidance and source content guidance.
```python
training_domain = 'arxiv'
model = FactorSum(training_domain)
summary = model.summarize(
        document, # a document string
        budget_guidance=200, # budget guidance in tokens
        source_token_budget=budget_guidance, # number of tokens to use from source document as content guidance
        verbose=True,
    )
```

A command-line tool is provided to explore summary samples and parameters. For instance, 
to see the summary for the sample 230 from arXiv test set, use the following command (GPU recommended):
```shell
python -m factorsum.model --doc_id 230 --dataset_name arxiv --split test \
--budget_guidance=200 --content_guidance_type source
```
It will output target abstract, the generated summary, and the evaluation scores.

### Colab Playground
A Colab notebook is available for summary generation. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CuMi6V4ZMPwLKXPYcmpuObMk55NHZivz?usp=sharing)

## Reproducing the evaluation results
The evaluation procedure relies on the following data:
- The [arXiv](https://huggingface.co/datasets/scientific_papers), [PubMed](https://huggingface.co/datasets/scientific_papers), and [GovReport](https://huggingface.co/datasets/ccdv/govreport-summarization) summarization datasets.
- The document views dataset generated by the [sampling procedure](./factorsum/sampling.py) (refer to Section 2.1 "Sampling Document Views" in [the paper](https://arxiv.org/pdf/2205.12486.pdf)).
- The summary views predicted from the document views (see Section 2.1.1 in the paper).

For convenience, we provide all the preprocessed resources, which can be downloaded using this command:
```shell
python -m factorsum.data download
```
Alternatively, you can use the instructions below to prepare the resources from scratch.

### Prepare data from scratch
Preprocess the summarization datasets (test splits):
```shell
python -m factorsum.data prepare_dataset scientific_papers arxiv --split test

python -m factorsum.data prepare_dataset scientific_papers pubmed --split test

python -m factorsum.data prepare_dataset ccdv/govreport-summarization govreport --split test
```

Then generate the document views for each dataset:
```shell
python -m factorsum.data prepare_dataset scientific_papers arxiv --split test --sample_type random --sample_factor 5 --views_per_doc 20

python -m factorsum.data prepare_dataset scientific_papers pubmed --split test --sample_type random --sample_factor 5 --views_per_doc 20

python -m factorsum.data prepare_dataset ccdv/govreport-summarization govreport --split test --sample_type random --sample_factor 5 --views_per_doc 20
```

Download the intrinsic model importance checkpoints:
```shell
python -m factorsum.utils download_models --model_dir ./artifacts
```
Currently, the checkpoints are:
- arXiv: [model-rs86h5g0:v0](https://drive.google.com/file/d/1QxPBNm5Eqx89YsqQ8Jjpj_Oofo5T2whg/view?usp=sharing)
- PubMed: [model-cku41vkj:v0](https://drive.google.com/file/d/1ni-hVXM2jLMD71Ez2t7ND7v0eAlbeEgr/view?usp=sharing)
- GovReport: [model-2oklw1wt:v0](https://drive.google.com/file/d/1ONB41tDm2x_QFiW0qw4gil7RMB6Lzitz/view?usp=sharing)

Finally, generate summary views using the `run_summarization.py` script (slightly adapted from the original [huggingface script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py)). The following command generates summary views for the arXiv test set using the model checkpoint in `artifacts/model-rs86h5g0:v0`:
```bash
MODEL_PATH='artifacts/model-rs86h5g0:v0' \
DATASET='arxiv' SPLIT='test' \
python scripts/run_summarization.py \
    --model_name_or_path "${MODEL_PATH}" \
    --do_predict \
    --output_dir output/"${DATASET}-${SPLIT}-summary_views" \
    --per_device_eval_batch_size=8 \
    --overwrite_output_dir \
    --predict_with_generate \
    --validation_file "data/${DATASET}-random_k_5_samples_20_${SPLIT}.csv" \
    --test_file "data/${DATASET}-random_k_5_samples_20_${SPLIT}.csv" \
    --text_column source \
    --summary_column target \
    --generation_max_length 128 \
    --generation_num_beams 4
```
It will generate a `generated_predictions.pkl` in the `output_dir` folder. To use the summary views, this file has to be moved to the `data` folder according to this naming convention:
```shell
cp output/"${DATASET}-${SPLIT}-summary_views/generated_predictions.pkl" data/"${DATASET}-${SPLIT}-summary_views-bart-${TRAINING_DOMAIN}-run=${RUN_ID}.pkl"
```
For instance, for the arXiv test set in-domain summary views we would have:
```shell
cp output/arxiv-test-summary_views/generated_predictions.pkl data/arxiv-test-summary_views-bart-arxiv-run=rs86h5g0.pkl
```

To generate summary views in a cross-domain setting, just set the variables `MODEL_PATH` and `DATASET` accordingly.

### Hyperparameters
Refer to the file [config.py](./factorsum/config.py) for hyperparameter definitions. 

### In-domain evaluation
The in-domain summarization results in Table 2 in [the paper](https://arxiv.org/pdf/2205.12486.pdf) 
are obtained with the following command:
```shell
python -m evaluation.factorsum evaluate --dataset_name arxiv --split test --output_dir output
```
where `dataset_name` is `arxiv`, `pubmed`, or `govreport`. By default, scores and summary predictions 
are saved to the `./output` folder.

### Cross-domain evaluation
(Results in Table 3 of [the paper](https://arxiv.org/pdf/2205.12486.pdf))
To specify the training domain of the intrinsic model, use the `training_domain` option.
The following example performs cross-domain evaluation on the arXiv dataset, using summary
views generated by a model trained on PubMed.
```shell
python -m evaluation.factorsum evaluate --dataset_name arxiv --split test --training_domain pubmed
```

### Varying budget guidance
Results for the experiments with varying budget guidance (Appendix D in [the paper](https://arxiv.org/pdf/2205.12486.pdf)) are obtained with the following command:
```shell
python -m evaluation.budgets --dataset_name <dataset_name> --split test
```
where `dataset_name` is `arxiv`, `pubmed`, or `govreport`. 

### Baselines
PEGASUS predictions:

```shell
python scripts/run_summarization.py \
    --model_name_or_path google/pegasus-arxiv \
    --do_predict \
    --output_dir /output \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --generation_max_length 256 \
    --generation_num_beams 8 \
    --val_max_target_length 256 \
    --max_source_length 1024 \
    --dataset_name scientific_papers \
    --dataset_config arxiv \
    --predict_split test
```

BigBird predictions:
```shell
python scripts/run_summarization.py \
    --model_name_or_path google/bigbird-pegasus-large-arxiv \
    --do_predict \
    --output_dir /content/output \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --report_to none \
    --generation_max_length 256 \
    --generation_num_beams 5 \
    --val_max_target_length 256 \
    --max_source_length 3072 \
    --dataset_name scientific_papers \
    --dataset_config arxiv \
    --predict_split test
```

## Training the intrinsic importance model
First, make sure the data for all splits are available (processing of the training sets might take several minutes):
```bash
python -m factorsum.data prepare_dataset scientific_papers arxiv
python -m factorsum.data prepare_dataset scientific_papers pubmed
python -m factorsum.data prepare_dataset ccdv/govreport-summarization govreport
```

Then run the training script as follows:

```bash
DATASET='arxiv' \
python scripts/run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir output/"${DATASET}"-k_5_samples_20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --gradient_accumulation_steps 4 \
    --generation_max_length 128 \
    --generation_num_beams 4 \
    --val_max_target_length 128 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --fp16 \
    --save_total_limit 2 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --save_steps 5000 \
    --eval_steps 5000 \
    --max_steps 50000 \
    --learning_rate 5e-5 \
    --report_to wandb \
    --metric_for_best_model eval_rouge1_fmeasure \
    --load_best_model_at_end \
    --max_train_samples 4000000 \
    --max_eval_samples 10000 \
    --max_predict_samples 10000 \
    --train_file data/"${DATASET}"-random_k_5_samples_20_train.csv \
    --validation_file data/"${DATASET}"-random_k_5_samples_20_validation.csv \
    --test_file data/"${DATASET}"-random_k_5_samples_20_test.csv \
    --text_column source \
    --summary_column target \
    --seed 17
```

Note: to use mixed precision (`--fp16`) you need a compatible CUDA device.

## Citation
```bibtex
@inproceedings{fonseca2022factorizing,
 author = {Fonseca, Marcio and Ziser, Yftah and Cohen, Shay B.},
 booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
 location = {Abu Dhabi},
 publisher = {Association for Computational Linguistics},
 title = {Factorizing Content and Budget Decisions in Abstractive Summarization of Long Documents},
 year = {2022}
}
```