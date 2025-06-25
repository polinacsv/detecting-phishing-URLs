import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from collections import Counter
import math

def compute_entropy(string):
    """Calculate Shannon entropy of a string."""
    prob = [n_x / len(string) for x, n_x in Counter(string).items()]
    entropy = -sum([p * math.log2(p) for p in prob])
    return entropy

def extract_url_features(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    df = df.copy()
    urls = df[url_col].astype(str)

    df["url_length"] = urls.str.len()
    df["num_digits"] = urls.str.count(r"\d")
    df["num_special_chars"] = urls.str.count(r"[^a-zA-Z0-9]")
    df["has_ip"] = urls.str.contains(r"(?:\d{1,3}\.){3}\d{1,3}").astype(int)
    df["num_subdomains"] = urls.apply(lambda u: len(urlparse(u).netloc.split(".")) - 2)
    df["num_dots"] = urls.str.count(r"\.")
    df["entropy"] = urls.apply(compute_entropy)
    df["has_https"] = urls.str.startswith("https").astype(int)
    df["has_at_symbol"] = urls.str.contains("@").astype(int)
    df["has_dash"] = urls.str.contains("-").astype(int)
    df["path_length"] = urls.apply(lambda u: len(urlparse(u).path))

    return df
