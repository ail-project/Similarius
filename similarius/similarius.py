import string

import requests
from requests import Response
import urllib3
from urllib3.exceptions import InsecureRequestWarning
from functools import lru_cache
from urllib.parse import urlparse

from typing import Tuple, Optional, List, Dict

from bs4 import BeautifulSoup
from bs4.element import Comment

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.feature_extraction import text as sklearn_text  # type: ignore

import nltk  # type: ignore
from nltk.corpus import stopwords  # type: ignore

urllib3.disable_warnings(InsecureRequestWarning)

# Shared session to reuse connections and reduce per-call allocations
_SESSION = requests.Session()

_SESSION.cookies.set_policy(requests.cookies.DefaultCookiePolicy(allowed_domains=[]))


##############
# Similarity #
##############

@lru_cache(maxsize=1)
def _build_stop_words() -> List[str]:
    """Load and cache combined stop words once."""
    _ensure_nltk_data()
    combined = set(stopwords.words('english')).union(set(sklearn_text.ENGLISH_STOP_WORDS)).union(set(string.punctuation))
    return list(combined)


def sk_similarity(doc1: str, doc2: str, max_features: int = 500) -> float:
    stop_words = _build_stop_words()
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
    tfidf = vectorizer.fit_transform([doc1, doc2])

    # Avoid dense conversion; compute cosine similarity directly on sparse vectors
    numerator = tfidf[0].multiply(tfidf[1]).sum()
    denom = (tfidf[0].power(2).sum() ** 0.5) * (tfidf[1].power(2).sum() ** 0.5)
    similarity = 0.0 if denom == 0 else float(numerator / denom)

    return round(similarity * 100)


##################
# Web treatment #
#################

def tag_visible(element) -> bool:
    """Identified element present in specific balise"""
    if element.parent and element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body: str) -> str:
    """Extract text from web page and remove text present in specific balise"""
    soup = BeautifulSoup(body, 'lxml')
    visible_texts = (t.strip() for t in soup.strings if tag_visible(t))
    filtered = [t for t in visible_texts if t]
    return " \n".join(filtered)


def find_list_resources(tag: str, attribute: str, soup: BeautifulSoup, max_items: int = 500) -> List[str]:
    """Find ressource in web page list in attribute, deduped and bounded."""
    list_resources: List[str] = []
    seen = set()
    for x in soup.find_all(tag):
        value = x.get(attribute)
        if not value or value in seen:
            continue
        seen.add(value)
        list_resources.append(value)
        if len(list_resources) >= max_items:
            break
    return list_resources


def website_ressource(soup: BeautifulSoup, max_items: int = 500) -> Dict[str, List[str]]:
    """Collect all website's ressources"""
    ressource_dict: Dict[str, List[str]] = dict()
    ressource_dict["image_scr"] = find_list_resources('img', "src", soup, max_items)
    ressource_dict["script_src"] = find_list_resources('script', "src", soup, max_items)
    ressource_dict["css_link"] = find_list_resources("link", "href", soup, max_items)
    ressource_dict["source_src"] = find_list_resources("source", "src", soup, max_items)
    ressource_dict["a_href"] = find_list_resources("a", "href", soup, max_items)
    ressource_dict["form_action"] = find_list_resources("form", "action", soup, max_items)

    return ressource_dict


def ressource_difference(original_ressource: Dict[str, List[str]], compare_ressource: Dict[str, List[str]]) -> str:
    """Difference between original website's ressources and comparaison website's ressources"""
    cp_total = 0
    cp_diff = 0

    for r_to_check in original_ressource:
        for r in original_ressource[r_to_check]:
            cp_total += 1
            if r_to_check in compare_ressource:
                if r not in compare_ressource[r_to_check]:
                    cp_diff += 1

    if cp_total == 0:
        return "0"

    return str(int((cp_diff / cp_total) * 100))


def ratio(ressource_diff: str, similarity: str) -> float:
    """Calculate the possibility of similarity based on ressource difference en tfidf similarity"""
    if int(ressource_diff) != 0:
        if int(ressource_diff) < int(similarity):
            return round((int(ressource_diff) / int(similarity)) * int(ressource_diff), 2)
        else:
            return round((int(similarity) / int(ressource_diff)) * int(similarity), 2)
    else:
        return 0.5


def get_website(website: str) -> Optional[Response]:
    """Request the website pass in parameter"""
    _SESSION.cookies.clear()
    candidates: List[str] = []
    parsed = urlparse(website)

    if parsed.scheme:
        candidates.append(website)
    else:
        candidates.append(f"https://{website}")
        candidates.append(f"http://{website}")

    for url in candidates:
        try:
            response = _SESSION.get(url, verify=True, timeout=3)
            return response
        except (urllib3.exceptions.NewConnectionError, requests.exceptions.ConnectionError, urllib3.exceptions.ReadTimeoutError, requests.exceptions.SSLError):
            continue
        except Exception:
            continue

    return None


def extract_text_ressource(website_text: str) -> Tuple[str, Dict[str, List[str]]]:
    """Extract the text from html and collect all ressources"""
    soup = BeautifulSoup(website_text, "lxml")

    text = text_from_html(website_text)
    ressource = website_ressource(soup)

    return text, ressource


def _ensure_nltk_data() -> None:
    """Ensure required NLTK datasets are available without downloading every import."""
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
    }

    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)
