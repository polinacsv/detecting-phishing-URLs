import pandas as pd
import numpy as np
import tldextract
from typing import Set
from urllib.parse import urlparse
import re


def extract_mld(url: str) -> str:
    """
    Extracts the main-level domain + public suffix using tldextract.
    E.g., from 'http://sub.mail.example.co.uk/login' returns 'example.co.uk'

    Parameters:
        url (str): A full URL string

    Returns:
        str: Main-level domain with public suffix
    """
    try:
        extracted = tldextract.extract(url)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}"
        return ''
    except Exception:
        return ''



def add_mld_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column 'mld_ps' to the DataFrame, which contains the extracted
    main-level domain + public suffix from each URL.

    Parameters:
        df (pd.DataFrame): A DataFrame with a 'url' column

    Returns:
        pd.DataFrame: A copy of the original DataFrame with a new 'mld_ps' column
    """
    df = df.copy()
    df['mld_ps'] = df['url'].apply(extract_mld)
    return df


def add_alexa_rank_feature(df: pd.DataFrame, alexa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two Alexa-based features:
    - 'is_in_alexa': 1 if the domain is found in Alexa Top 1M, else 0
    - 'ranking': Alexa rank (1 = most popular). If not found, impute with 1_000_001

    Parameters:
        df (pd.DataFrame): DataFrame that must include an 'mld_ps' column
        alexa_df (pd.DataFrame): Alexa dataset with ['alexa_domain', 'rank']

    Returns:
        pd.DataFrame: Original DataFrame with 'is_in_alexa' and 'ranking' columns added
    """
    df = df.copy()
    merged = df.merge(alexa_df, how='left', left_on='mld_ps', right_on='alexa_domain')
    
    # Create binary indicator
    df['is_in_alexa'] = merged['rank'].notnull().astype(int)
    
    # Fill missing ranks with a default (beyond Top 1M)
    df['ranking'] = merged['rank'].fillna(1_000_001).astype(int)

    return df


def tokenize(text: str) -> Set[str]:
    """
    Splits input text into lowercase alphanumeric tokens.
    Useful for comparing similarity between URL parts.

    Parameters:
        text (str): Input string (e.g., a URL path or domain)

    Returns:
        Set[str]: Set of lowercase alphanumeric tokens
    """
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Computes Jaccard similarity between two sets:
    J(A, B) = |A ∩ B| / |A ∪ B|

    Returns:
        float: Jaccard similarity score (0.0 to 1.0)
    """
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def extract_jaccard_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Jaccard similarity features to compare how much overlap there is
    between the tokens in the URL path and the tokens in:
    (1) the main domain (mld_ps), and
    (2) the subdomain.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'url' and 'mld_ps' columns

    Returns:
        pd.DataFrame: Original dataframe with two new features:
                      - jaccard_mld_path
                      - jaccard_subdomain_path
    """
    df = df.copy()

    # Ensure URLs are parseable by adding scheme if missing
    df['full_url'] = df['url'].apply(lambda u: u if u.startswith('http') else f'http://{u}')

    def compute_jaccards(url, mld_ps):
        try:
            parsed = urlparse(url)

            # Tokenize the URL path
            path_tokens = tokenize(parsed.path)

            # Tokenize the main domain (e.g., 'paypal', 'com')
            mld_tokens = tokenize(mld_ps) if pd.notnull(mld_ps) else set()

            # Extract and tokenize subdomain (e.g., 'accounts' in 'accounts.google.com')
            hostname_parts = parsed.hostname.split('.') if parsed.hostname else []
            subdomain_parts = hostname_parts[:-2] if len(hostname_parts) > 2 else []
            subdomain_tokens = tokenize('.'.join(subdomain_parts))

            # Calculate Jaccard similarities
            jaccard_path = jaccard_similarity(mld_tokens, path_tokens)
            jaccard_subdomain = jaccard_similarity(subdomain_tokens, path_tokens)

            return pd.Series([jaccard_path, jaccard_subdomain])
        except:
            return pd.Series([0.0, 0.0])  # Fallback in case of parse errors

    df[['jaccard_mld_path', 'jaccard_subdomain_path']] = df.apply(
        lambda row: compute_jaccards(row['full_url'], row['mld_ps']),
        axis=1
    )

    return df


def compute_shannon_entropy(text: str) -> float:
    """
    Computes Shannon entropy of a string, which measures the randomness
    or complexity of the characters.

    Parameters:
        text (str): Input string (typically a URL or part of it)

    Returns:
        float: Shannon entropy value
    """
    if not text:
        return 0.0

    from collections import Counter
    counts = Counter(text)
    probabilities = [count / len(text) for count in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy


def add_entropy_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a Shannon entropy feature to the DataFrame based on the full URL.

    Parameters:
        df (pd.DataFrame): Must contain a 'url' column

    Returns:
        pd.DataFrame: DataFrame with new column:
                      - 'url_entropy': Shannon entropy of the full URL
    """
    df = df.copy()

    # Ensure full URL has a scheme for consistent parsing
    df['full_url'] = df['url'].apply(lambda u: u if u.startswith('http') else f'http://{u}')

    df['url_entropy'] = df['full_url'].apply(compute_shannon_entropy)
    return df


def add_url_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds structural URL features:
    - num_digits
    - url_length
    - num_subdomains
    - num_path_segments
    - num_suspicious_keywords
    - num_special_chars
    - has_brand_conflict
    """
    df = df.copy()

    df['full_url'] = df['url'].apply(lambda u: u if u.startswith('http') else f'http://{u}')
    
    suspicious_keywords = {'login', 'secure', 'verify', 'account', 'update', 'banking',
                           'webscr', 'ebay', 'paypal', 'signin', 'submit', 'confirm'}

    known_brands = {'paypal', 'apple', 'amazon', 'google', 'facebook', 'bankofamerica', 'microsoft'}

    def extract_structure_features(url: str):
        try:
            parsed = urlparse(url)
            hostname_parts = parsed.hostname.split('.') if parsed.hostname else []

            num_digits = sum(c.isdigit() for c in url)
            url_length = len(url)
            num_subdomains = max(0, len(hostname_parts) - 2)
            num_path_segments = len(parsed.path.strip('/').split('/')) if parsed.path else 0
            url_tokens = re.findall(r"[a-zA-Z]+", url.lower())
            num_suspicious_keywords = sum(token in suspicious_keywords for token in url_tokens)
            num_special_chars = len(re.findall(r"[^a-zA-Z0-9]", url))

            # brand conflict check
            domain_part = ".".join(hostname_parts[-2:]) if len(hostname_parts) >= 2 else parsed.hostname
            path_part = parsed.path.lower()
            domain_brands = [b for b in known_brands if b in domain_part]
            path_brands = [b for b in known_brands if b in path_part]
            has_brand_conflict = int(
                len(domain_brands) > 0 and
                any(b not in domain_brands for b in path_brands)
            )

            return pd.Series([
                num_digits,
                url_length,
                num_subdomains,
                num_path_segments,
                num_suspicious_keywords,
                num_special_chars,
                has_brand_conflict
            ])
        except:
            return pd.Series([0, 0, 0, 0, 0, 0, False])

    df[[
        'num_digits', 'url_length', 'num_subdomains',
        'num_path_segments', 'num_suspicious_keywords',
        'num_special_chars', 'has_brand_conflict'
    ]] = df['full_url'].apply(extract_structure_features)

    return df