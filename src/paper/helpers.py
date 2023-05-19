import re


def translate_domain(domain):
    if domain == "news":
        return "News"
    elif domain == "pubmed":
        return "PubMed"
    elif domain == "random-words":
        return "Random Words"
    elif domain == "random-bytes":
        return "Random Bytes"
    elif domain == "binary":
        return "Multimedia"
    else:
        raise ValueError(domain)


def translate_model(model_name, pretrained=True):
    if model_name == "gpt2" and pretrained:
        return "GPT-2"
    if model_name == "gpt2" and not pretrained:
        return "GPT-2 (rand)"
    elif model_name == "gpt2-medium":
        return "335M"
    elif model_name == "EleutherAI/pythia-70m":
        return "Pythia 70M"
    elif model_name == "EleutherAI/pythia-70m-deduped":
        return "Pythia 70M, deduped"
    elif model_name == "EleutherAI/pythia-160m":
        return "Pythia"
    elif model_name == "EleutherAI/pythia-160m-deduped":
        return "Pythia 160M, deduped"
    elif model_name == "cerebras/Cerebras-GPT-111M":
        return "Cerebras"
    else:
        raise ValueError(model_name)


def translate_feature(feature):
    if feature == "l2-norm":
        return "L2"
    elif feature == "l1-norm":
        return "L1"
    elif feature == "std":
        return "Std"
    elif feature == "mean":
        return "Mean"
    elif feature == "max":
        return "Max"
    elif feature == "min":
        return "Min"
    else:
        raise ValueError(feature)


def translate_filename(filename):
    if filename == "data/pubmed/100-tokens/0.txt":
        return "PubMed (PM)"
    elif filename == "data/news/100-tokens/0.txt":
        return "News ($m1$)"
        # return "News (N0)"
    elif filename == "data/news/100-tokens/1.txt":
        return "News (N1)"
    elif filename == "data/random-bytes/100-tokens/0.txt":
        return "Rand. Bytes ($m2$)"
        # return "Rand. Words (RW)"
    elif filename == "data/random-words/100-tokens/0.txt":
        return "Rand. Words ($m2$)"
        # return "Rand. Words (RW)"
    else:
        raise ValueError(filename)


def parse_length(filename: str) -> int:
    # data/random-words/100-tokens/5.txt -> 100
    pattern = re.compile(r"data/[a-z\-]+/(\d+)-tokens/\d\.txt")
    match = pattern.match(filename)

    return int(match.group(1))


def parse_domain(filename: str) -> str:
    # data/random-words/100-tokens/5.txt -> "random-words"
    pattern = re.compile(r"data/([a-z\-]+)/\d+-tokens/\d\.txt")
    match = pattern.match(filename)

    return match.group(1)
