from collections import defaultdict


def _get_params():

    params = defaultdict(int)
    params["content_weight"] = 1.0
    params["budget_weight"] = 1.0
    params["token_budget"] = 200
    params["views_per_doc"] = 20
    params["sample_factor"] = 5
    params["sample_type"] = "random"

    # we adjust budgets so that the average predicted summary words
    # is close to the validation set average reference summary words
    params["pubmed"] = {}
    params["pubmed"]["intrinsic_model_id"] = "cku41vkj"
    params["pubmed"]["google_drive_id"] = "1ni-hVXM2jLMD71Ez2t7ND7v0eAlbeEgr"
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
    params["arxiv"]["intrinsic_model_id"] = "rs86h5g0"
    params["arxiv"]["google_drive_id"] = "1QxPBNm5Eqx89YsqQ8Jjpj_Oofo5T2whg"
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
    params["govreport"]["intrinsic_model_id"] = "2oklw1wt"
    params["govreport"]["google_drive_id"] = "1ONB41tDm2x_QFiW0qw4gil7RMB6Lzitz"
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


def model_params(domain_name, **kwargs):
    params = _get_params()
    params["dataset_path"] = params[domain_name].get("dataset_path", domain_name)

    domains = ["pubmed", "arxiv", "govreport"]
    default_params = [key for key in params.keys() if key not in domains]

    _model_params = params[domain_name]
    _model_params["domain_name"] = domain_name

    for param_key in default_params:
        param_value = params[domain_name].get(param_key, params[param_key])
        _model_params[param_key] = param_value

    # override params if values are not None
    for key, val in kwargs.items():
        if val is not None:
            _model_params[key] = val

    return _model_params
