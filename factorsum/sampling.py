import random

import numpy as np
import matplotlib.pyplot as plt


def show_sampling_metrics(metrics):
    print(f'Statistics for {metrics["valid_docs"]} documents')
    print("Oracle coverage:", metrics["oracle_coverage"])
    print("Average sentences per document:", metrics["avg_sents_per_document"])
    print(
        "Average abstract sentences per document:",
        metrics["avg_target_sents_per_document"],
    )
    print(
        "Average abstract words per document:",
        metrics["avg_target_words_per_document"],
    )
    print("Average document coverage:", metrics["source_coverage"])
    print("Average sentences per sample:")
    for sample_idx, avg_sents in enumerate(metrics["avg_sents_per_document"]):
        print(f"- sample {sample_idx}:", avg_sents)

    all_oracle_pos = metrics["all_oracle_pos"]
    print("\nAverage oracle position (entire document):", np.mean(all_oracle_pos))
    print("Oracle position std (entire document):", np.std(all_oracle_pos))
    plt.figure(figsize=(12, 6))
    plt.xlim([min(all_oracle_pos), max(all_oracle_pos)])
    plt.hist(all_oracle_pos, bins=20, alpha=0.5)
    plt.title("Oracle position (entire document)")
    plt.xlabel("position")
    plt.ylabel("count")
    plt.show()

    all_oracle_pos_in_sample = metrics["all_oracle_pos_in_view"]
    print("\nAverage oracle position (inside view):", np.mean(all_oracle_pos_in_sample))
    print("Oracle position std (inside view):", np.std(all_oracle_pos_in_sample))
    plt.figure(figsize=(12, 6))
    plt.xlim([min(all_oracle_pos_in_sample), max(all_oracle_pos_in_sample)])
    plt.hist(all_oracle_pos_in_sample, bins=20, alpha=0.5)
    plt.title("Oracle position inside view")
    plt.xlabel("position")
    plt.ylabel("count")
    plt.show()

    oracle_freqs = metrics["oracle_freqs"]
    print("\nAverage oracle frequency per view:", np.mean(oracle_freqs))
    print("Oracle frequency std per view:", np.std(oracle_freqs))
    plt.figure(figsize=(12, 6))
    plt.xlim([min(oracle_freqs), max(oracle_freqs)])
    plt.hist(oracle_freqs, alpha=0.5)
    plt.title("Oracle sentences per view")
    plt.xlabel("number of oracle sentences")
    plt.ylabel("count")
    plt.show()

    plt.figure(figsize=(12, 6))

    paper_abstract_lengths = metrics["paper_abstract_lengths"]
    paper_abstract_lengths = np.array(paper_abstract_lengths)
    ratios = paper_abstract_lengths[:, 1] / paper_abstract_lengths[:, 0]

    print("Average abstract length:", metrics["avg_abstract_sents_per_document"])
    idx = paper_abstract_lengths[:, 0] < 50
    print("Average abstract length (< 50):", np.mean(paper_abstract_lengths[idx, 1]))
    idx = (50 <= paper_abstract_lengths[:, 0]) & (paper_abstract_lengths[:, 0] <= 100)
    print(
        "Average document/abstract length ratio (50-100):",
        np.mean(paper_abstract_lengths[idx, 1]),
    )
    idx = (100 <= paper_abstract_lengths[:, 0]) & (paper_abstract_lengths[:, 0] <= 200)
    print(
        "Average document/abstract length ratio (100-200):",
        np.mean(paper_abstract_lengths[idx, 1]),
    )
    idx = paper_abstract_lengths[:, 0] > 200
    print(
        "Average document/abstract length ratio (> 200):",
        np.mean(paper_abstract_lengths[idx, 1]),
    )
    plt.scatter(paper_abstract_lengths[:, 0], ratios)
    plt.title("Ratio number of sentences vs abstract sentences")
    plt.xlabel("total number of sentences")
    plt.ylabel("ratio")
    plt.show()


def get_document_views(
    source,
    sample_factor=5,
    views_per_doc=20,
    target=None,
    oracle=None,
    sample_fn=None,
    require_oracle=False,
    top_k=None,
    seed=17,
):

    source_views = []
    target_views = []
    oracle_views = []

    # Collect sentences in views to calculate source/oracle coverage
    source_sents_in_views = []
    oracle_sents_in_views = []

    oracle_sent_pos = []

    if top_k is None:
        split_length = len(source) // sample_factor
    else:
        split_length = top_k

    if oracle:
        for sent_pos, sent in enumerate(source):
            if sent in oracle:
                oracle_sent_pos.append(sent_pos)

    for view_idx in range(views_per_doc):
        source_view = []
        target_view = []
        oracle_view = []
        source_idxs = []
        oracle_idxs = []
        oracle_sent_pos_in_source_view = []

        if sample_fn is None:
            random.seed(view_idx + seed)
            sample = random.sample(source, len(source))
        else:
            sample = sample_fn(source, view_idx, sample_factor)

        for sent_pos, sentence in enumerate(sample):
            source_view.append(sentence)
            source_idxs.append(sent_pos)

            if target and oracle:
                for target_sent, oracle_sent in zip(target, oracle):
                    if sentence == oracle_sent:
                        oracle_view.append(oracle_sent)
                        target_view.append(target_sent)
                        oracle_idxs.append(sent_pos)

            split_len_ok = len(source_view) >= split_length
            oracle_ok = len(oracle_view) > 0 or not require_oracle

            if split_len_ok and oracle_ok:
                # reorder sentences according to original position in the document
                source_view = [source_view[ii] for ii in np.argsort(source_idxs)]
                oracle_view = [oracle_view[ii] for ii in np.argsort(oracle_idxs)]
                target_view = [target_view[ii] for ii in np.argsort(oracle_idxs)]

                source_sents_in_views.extend(source_view)
                oracle_sents_in_views.extend(oracle_view)
                break

        for idx, sent in enumerate(source_view):
            if sent in oracle_view:
                oracle_sent_pos_in_source_view.append(idx)

        source_views.append(source_view)
        target_views.append(target_view)
        oracle_views.append(oracle_view)

    # remove duplicated sentences
    source_sents_in_views = list(set(source_sents_in_views))
    oracle_sents_in_views = list(set(oracle_sents_in_views))

    result = dict(
        source_views=source_views, source_sents_in_views=source_sents_in_views
    )

    if target:
        result["target_views"] = target_views
    if oracle:
        result["oracle_views"] = oracle_views
        result["oracle_sents_in_views"] = oracle_sents_in_views
        result["oracle_sent_pos_in_source_view"] = oracle_sent_pos_in_source_view
        result["oracle_sent_pos"] = oracle_sent_pos

    return result


def sample_dataset(
    data,
    sample_factor=5,
    views_per_doc=20,
    sample_fn=None,
    require_oracle=False,
    top_k=None,
    verbose=True,
    seed=17,
):

    doc_ids = []
    all_source_views = []
    all_target_views = []
    all_oracle_views = []
    original_targets = []
    all_source_coverage = []
    all_oracle_coverage = []
    all_oracle_pos = []
    all_oracle_pos_in_view = []
    oracle_freqs = []
    source_view_sent_counts = []
    target_sent_counts = []
    target_word_counts = []

    sources = data["sources"]
    targets = data["targets"]
    oracles = []

    for ii, source in enumerate(sources):
        oracles.append([source[jj] for jj in data["oracles"][ii] if jj is not None])

    for doc_idx, (source, target, oracle) in enumerate(zip(sources, targets, oracles)):

        if len(source) == 0 or len(target) == 0:
            continue

        if len(oracle) == 0 and require_oracle:
            continue

        target_sent_counts.append(len(target))
        target_word_counts.append(sum([len(abs.split()) for abs in target]))

        doc_views = get_document_views(
            source,
            sample_factor=sample_factor,
            views_per_doc=views_per_doc,
            target=target,
            oracle=oracle,
            sample_fn=sample_fn,
            require_oracle=require_oracle,
            seed=seed,
            top_k=top_k,
        )

        n_views = len(doc_views["source_views"])
        doc_ids.extend([doc_idx] * n_views)
        original_targets.extend([target] * n_views)
        all_source_views.extend(doc_views["source_views"])
        all_target_views.extend(doc_views["target_views"])
        all_oracle_views.extend(doc_views.get("oracle_views", [""]))
        source_view_sent_counts.extend([len(x) for x in doc_views["source_views"]])
        all_source_coverage.append(
            len(doc_views["source_sents_in_views"]) / len(source)
        )

        if len(oracle) > 0:
            oracle_coverage = len(doc_views["oracle_sents_in_views"]) / len(oracle)
            all_oracle_coverage.append(oracle_coverage)
            all_oracle_pos.extend(doc_views["oracle_sent_pos"])
            all_oracle_pos_in_view.extend(doc_views["oracle_sent_pos_in_source_view"])

    metrics = {
        "valid_docs": len(doc_ids),
        "oracle_coverage": np.mean(all_oracle_coverage),
        "source_coverage": np.mean(all_source_coverage),
        "avg_sents_per_document": np.mean([len(x) for x in sources]),
        "avg_target_sents_per_document": np.mean(target_sent_counts),
        "avg_target_words_per_document": np.mean(target_word_counts),
        "avg_sents_per_source_view": np.mean(source_view_sent_counts),
        "all_oracle_sent_pos": all_oracle_pos,
        "all_oracle_sent_pos_in_view": all_oracle_pos_in_view,
        "oracle_freqs": oracle_freqs,
    }

    if verbose:
        show_sampling_metrics(metrics)

    return {
        "doc_ids": doc_ids,
        "sources": all_source_views,
        "targets": all_target_views,
        "oracles": all_oracle_views,
        "original_targets": original_targets,
    }, metrics
