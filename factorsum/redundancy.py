import re

import nltk
import textdistance
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)

non_alpha_pattern = re.compile("[^\w_\s]+")


def has_repeated_trigram(text, trigrams):
    words = nltk.word_tokenize(text)
    sample_trigrams = nltk.trigrams(words)
    try:
        sample_trigrams = list(sample_trigrams)
        if any([(t in trigrams) for t in sample_trigrams]):
            return True
        trigrams.extend(sample_trigrams)
        return False
    except RuntimeError:
        print("Failed to get trigrams")
        print(words)
        print(text)


def has_similar_content(text, sentences, threshold=0.4, max_tokens=30):

    text = non_alpha_pattern.sub("", text)
    text_words = text.split()
    len_text_words = len(text_words)

    for sent in sentences:
        sent = non_alpha_pattern.sub("", sent)

        if text in sent or sent in text:
            return True

        sent_words = sent.split()
        len_sent_words = len(sent_words)

        if (
            abs(len_sent_words - len_text_words)
            * 1.0
            / (len_sent_words + len_text_words)
            > threshold
        ):
            # try to save some computation, since we know that
            # levenshtein distance will be larger than threshold
            continue

        # we may limit the number of tokens to avoid an explosion in cost of distance computation
        distance = textdistance.levenshtein.distance(
            text_words[:max_tokens], sent_words[:max_tokens]
        )
        normalizer = min(max_tokens, len_sent_words) + min(max_tokens, len_text_words)
        normalized_distance = distance * 1.0 / normalizer

        if normalized_distance <= threshold:
            return True
    return False
