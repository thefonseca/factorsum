import os
import textwrap


def get_sources(data):
    if "sources" in data:
        return data["sources"]
    return data["articles"]


def get_targets(data):
    if "targets" in data:
        return data["targets"]
    return data["abstracts"]


def log_rouge_scores(scores):
    for k, v in sorted(scores.items()):
        if hasattr(v, "low"):
            print("%s-R: %f,%f,%f" % (k, v.low.recall, v.mid.recall, v.high.recall))
            print(
                "%s-P: %f,%f,%f"
                % (k, v.low.precision, v.mid.precision, v.high.precision)
            )
            print(
                "%s-F: %f,%f,%f\n"
                % (k, v.low.fmeasure, v.mid.fmeasure, v.high.fmeasure)
            )
        else:
            print("%s-R: %f,%f,%f" % (k, v.recall, v.recall, v.recall))
            print("%s-P: %f,%f,%f" % (k, v.precision, v.precision, v.precision))
            print("%s-F: %f,%f,%f\n" % (k, v.fmeasure, v.fmeasure, v.fmeasure))


def log_summary(doc_id, pred, target, score, bad_score, good_score, score_key="rouge1"):
    if bad_score and score[score_key].fmeasure < bad_score:
        print("\n>> BAD SUMMARY ============")
        print("> DOC ID:", doc_id)
        for sent in pred.split("\n"):
            print(textwrap.fill(f"- {sent}", 80))
        print()
        print("> Abstract:")
        print(textwrap.fill(target, 80))
        print("\n> Scores:")
        log_rouge_scores(score)

    if good_score and score[score_key].fmeasure > good_score:
        print("\n>> GOOD SUMMARY ============")
        print("> DOC ID:", doc_id)
        for sent in pred.split("\n"):
            print(textwrap.fill(f"- {sent}", 80))
        print()
        print("> Abstract:")
        print(textwrap.fill(target, 80))
        print("\n> Scores:")
        log_rouge_scores(score)


def get_output_path(
    save_dir, dataset, split, content, budget, training_domain=None, timestr=None
):
    save_to = None
    suffix = f"{content}_content-{budget}_budget"

    if save_dir:
        save_dir = f"{save_dir}-{dataset}-{split}"
        if training_domain:
            save_dir = f"{save_dir}-{training_domain}"
        if timestr:
            save_dir = f"{save_dir}_{timestr}"
        save_to = f"{dataset}-{split}-{suffix}.csv"
        save_to = os.path.join(save_dir, save_to)
    return save_to
