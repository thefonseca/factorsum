from collections import defaultdict


def _get_params():

    params = defaultdict(int)
    params["content_weight"] = 1.0
    params["budget_weight"] = 1.0
    params["token_budget"] = 200
    params["views_per_doc"] = 20
    params["sample_factor"] = 5
    params["sample_type"] = "random"
    params["min_words_per_view"] = 5

    # we adjust budgets so that the average predicted summary words
    # is close to the validation set average reference summary words
    params["pubmed"] = {}
    params["pubmed"]["intrinsic_importance_model_id"] = "cku41vkj"
    params["pubmed"][
        "intrinsic_importance_model_url"
    ] = "https://www.dropbox.com/s/8birb30qhjxyqnr/model-cku41vkj%20v0.zip?dl=1"
    params["pubmed"][
        "dataset_url"
    ] = "https://www.dropbox.com/s/4vh8hfoio7r4vfl/factorsum-pubmed.zip?dl=1"
    params["pubmed"]["token_budget"] = 205
    params["pubmed"]["no_content_fixed_budget_adjust"] = 8
    params["pubmed"]["no_content_oracle_budget_adjust"] = 11
    params["pubmed"]["no_content_pegasus_budget_adjust"] = 7
    params["pubmed"]["no_content_bigbird_budget_adjust"] = 12
    params["pubmed"]["source_content_fixed_budget_adjust"] = 12
    params["pubmed"]["source_content_oracle_budget_adjust"] = 16
    params["pubmed"]["oracle_content_fixed_budget_adjust"] = 14
    params["pubmed"]["pegasus_content_fixed_budget_adjust"] = 30
    params["pubmed"]["bigbird_content_fixed_budget_adjust"] = 27
    params["pubmed"]["pegasus_content_pegasus_budget_adjust"] = 17
    params["pubmed"]["bigbird_content_bigbird_budget_adjust"] = 22
    params["pubmed"]["dataset_path"] = "scientific_papers"

    params["arxiv"] = {}
    params["arxiv"]["intrinsic_importance_model_id"] = "rs86h5g0"
    params["arxiv"][
        "intrinsic_importance_model_url"
    ] = "https://www.dropbox.com/s/apgdmfqqwz22p7e/model-rs86h5g0%20v0.zip?dl=1"
    params["arxiv"][
        "dataset_url"
    ] = "https://www.dropbox.com/s/tzlcptrgwu41un8/factorsum-arxiv.zip?dl=1"
    params["arxiv"]["token_budget"] = 165
    params["arxiv"]["no_content_fixed_budget_adjust"] = 2
    params["arxiv"]["no_content_oracle_budget_adjust"] = 4
    params["arxiv"]["no_content_pegasus_budget_adjust"] = 6
    params["arxiv"]["no_content_bigbird_budget_adjust"] = 5
    params["arxiv"]["source_content_fixed_budget_adjust"] = 2
    params["arxiv"]["source_content_oracle_budget_adjust"] = 4
    params["arxiv"]["oracle_content_fixed_budget_adjust"] = 7
    params["arxiv"]["pegasus_content_fixed_budget_adjust"] = 10
    params["arxiv"]["bigbird_content_fixed_budget_adjust"] = 10
    params["arxiv"]["pegasus_content_pegasus_budget_adjust"] = 15
    params["arxiv"]["bigbird_content_bigbird_budget_adjust"] = 12
    params["arxiv"]["dataset_path"] = "scientific_papers"

    params["govreport"] = {}
    params["govreport"]["intrinsic_importance_model_id"] = "2oklw1wt"
    params["govreport"][
        "intrinsic_importance_model_url"
    ] = "https://www.dropbox.com/s/tramsr8g27smuju/model-2oklw1wt%20v0.zip?dl=1"
    params["govreport"][
        "dataset_url"
    ] = "https://www.dropbox.com/s/vc5yyooa2o6euyj/factorsum-govreport.zip?dl=1"
    params["govreport"]["token_budget"] = 648
    params["govreport"]["no_content_fixed_budget_adjust"] = 8
    params["govreport"]["no_content_oracle_budget_adjust"] = 8
    params["govreport"]["source_content_fixed_budget_adjust"] = -24
    params["govreport"]["source_content_oracle_budget_adjust"] = -16
    params["govreport"]["oracle_content_fixed_budget_adjust"] = -10
    params["govreport"]["no_content_bart-large_budget_adjust"] = 50
    params["govreport"]["bart-large_content_fixed_budget_adjust"] = 10
    params["govreport"]["bart-large_content_oracle_budget_adjust"] = 10
    params["govreport"]["bart-large_content_bart-large_budget_adjust"] = 50
    params["govreport"]["dataset_path"] = "ccdv/govreport-summarization"

    return params


def model_params(domain_name, default_params=None, **kwargs):
    if default_params is None:
        params = _get_params()
    else:
        params = default_params

    params["dataset_path"] = params[domain_name].get("dataset_path", domain_name)

    domains = ["pubmed", "arxiv", "govreport"]
    default_param_keys = [key for key in params.keys() if key not in domains]

    _model_params = params[domain_name]
    _model_params["domain_name"] = domain_name

    for param_key in default_param_keys:
        param_value = params[domain_name].get(param_key, params[param_key])
        _model_params[param_key] = param_value

    # override params if values are not None
    for key, val in kwargs.items():
        if val is not None:
            _model_params[key] = val

    return _model_params
