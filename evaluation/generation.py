import re
import textwrap

import nltk
import fire

from .utils import log_rouge_scores, get_eval_data
from model.score import extrinsic_scores
from model.inference import summarize
from model.config import model_params


try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)

CONCEPT_NAMES = ['Background', 'Conclusion', 'Method', 'Objective', 'Result']
non_alpha_pattern = re.compile('[^\w_\s]+')
start_with_non_alpha_pattern = re.compile('^[^\w_\s]+\s') 


def show_summary(summary):
    if type(summary) == dict:
        for concept in ['Objective', 'Background', 'Method', 'Result', 'Conclusion']: # sorted(summary.keys()):
            if concept not in summary:
                continue
            print('>>', concept)
            for sent in summary[concept]:
                print(textwrap.fill(f'  - {sent}', 80))
            print()
    elif type(summary) == list:
        for sent in summary:
            print(textwrap.fill(f'- {sent}', 80))
    else:
        print(summary)


def show_score(score):
    if 'kl' in score:
        print('> Concept scores')
        print('kl divergence:', score['kl'])
        print('kendall tau correlation:', score['kendalltau'])
        print('Concept distribution:', score['concept_distribution'])
        print('Concept score:', score['concept_score'])
    
    if 'rouge' in score:
        print('\n> ROUGE score:')
        log_rouge_scores(score['rouge'])
    
    if 'budget_error' in score:
        print('> Budget error:', score['budget_error'])


def generate(doc_id=617, data_dir='data', dataset='arxiv', split='test',
             training_domain=None, concept_guidance=False, budget_weight=None, 
             source_token_budget=None, token_budget=None, intrinsic_model_id=None, 
             views_per_doc=None, sample_factor=None, cache_dir=None):
    
    params = model_params(dataset, 
                          source_token_budget=source_token_budget,
                          budget_weight=budget_weight, 
                          token_budget=token_budget,
                          views_per_doc=views_per_doc,
                          sample_factor=sample_factor,
                          intrinsic_model_id=intrinsic_model_id)
    
    eval_data = get_eval_data(params['dataset_path'], dataset, split, 
                              data_dir, training_domain, concept_guidance, 
                              cache_dir, params)
        
    eval_views_dataset = eval_data['eval_views_dataset']
    doc = eval_views_dataset[eval_views_dataset.doc_id == doc_id]
    
    print('\n> DOC ID:', doc.doc_id.iloc[0])
    print()
    print('>> Abstract:')
    summary = doc.full_abstract.iloc[0]
    summary = nltk.sent_tokenize(doc.full_abstract.iloc[0].replace('\n', '. ').replace('..', '.'))
    
    for sent in summary:
        print(textwrap.fill(f'  - {sent}', 80))

    score = extrinsic_scores(summary, target_summary=doc.full_abstract.iloc[0])
    print('> Tokens:', sum([len(nltk.word_tokenize(sent)) for sent in summary]))
    print()
    show_score(score)

    print('\n>> No content guidance, fixed budget')
    summary, _, idxs = summarize(doc.summary.values, token_budget=params['token_budget'], verbose=True)
    score = extrinsic_scores(summary, target_summary=doc.full_abstract.iloc[0],
                             token_budget=params['token_budget'])
    print('>> Tokens:', sum([len(nltk.word_tokenize(sent)) for sent in summary]))
    show_score(score)

    for model in ['source', 'pegasus', 'bigbird-pegasus-large', 'bart-large']:
        model_summaries = eval_data.get(f'{model}_summaries')
        budget_adjust_key = f"{model}_content_fixed_budget_adjust"
        budget_adjust = params.get(budget_adjust_key, 0)
        
        if model_summaries:
            print(f'\n>> {model} content guidance, fixed budget')
            model_summary = model_summaries[doc_id]
            token_budget = params['token_budget'] + budget_adjust
            summary, _, idxs = summarize(doc.summary.values, 
                                         text_guidance=model_summary,
                                         token_budget=token_budget,
                                         verbose=True)

            score = extrinsic_scores(summary, target_summary=doc.full_abstract.iloc[0],
                                     token_budget=token_budget)
            print('> Tokens:', sum([len(nltk.word_tokenize(sent)) for sent in summary]))
            show_score(score)

            print(f'\n>> Vanilla {model}')
            if type(model_summary) == str:
                model_summary = nltk.sent_tokenize(model_summary.replace('<n>', ' '))
            for sent in model_summary:
                print(textwrap.fill(f'  - {sent}', 80))

            score = extrinsic_scores(model_summary, target_summary=doc.full_abstract.iloc[0])
            print('\n>> Tokens:', len(nltk.word_tokenize(' '.join(model_summary))))
            print()
            show_score(score)

    
if __name__ == '__main__':
    fire.Fire(generate)