import string

import requests
from requests import Response
import urllib3
from urllib3.exceptions import InsecureRequestWarning

from typing import Tuple, Optional, List, Dict

from bs4 import BeautifulSoup
from bs4.element import Comment

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.feature_extraction import text as sklearn_text  # type: ignore

import nltk  # type: ignore
from nltk.corpus import stopwords  # type: ignore

urllib3.disable_warnings(InsecureRequestWarning)
nltk.download('punkt')
nltk.download('stopwords')


##############
# Similarity #
##############

def sk_similarity(doc1: str, doc2: str) -> float:
    ENGLISH_STOP_WORDS = set(stopwords.words('english')).union(set(sklearn_text.ENGLISH_STOP_WORDS)).union(set(string.punctuation))
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), max_features=5000)
    tfidf = vectorizer.fit_transform([doc1, doc2])

    return round(((tfidf * tfidf.T).toarray())[0, 1] * 100)


##################
# Web treatment #
#################

def tag_visible(element) -> bool:
    """Identified element present in specific balise"""
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body: str) -> str:
    """Extract text from web page and remove text present in specific balise"""
    soup = BeautifulSoup(body, 'lxml')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" \n".join(t.strip() for t in visible_texts)


def find_list_resources(tag: str, attribute: str, soup: BeautifulSoup) -> List[str]:
    """Find ressource in web page list in attribute"""
    list_resources = []
    for x in soup.findAll(tag):
        try:
            list_resources.append(x[attribute])
        except KeyError:
            pass
    return list_resources


def website_ressource(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """Collect all website's ressources"""
    ressource_dict = dict()
    ressource_dict["image_scr"] = find_list_resources('img', "src", soup)
    ressource_dict["script_src"] = find_list_resources('script', "src", soup)
    ressource_dict["css_link"] = find_list_resources("link", "href", soup)
    ressource_dict["source_src"] = find_list_resources("source", "src", soup)
    ressource_dict["a_href"] = find_list_resources("a", "href", soup)
    ressource_dict["form_action"] = find_list_resources("form", "action", soup)

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

    return str(int((cp_diff / cp_total) * 100))


def ratio(ressource_diff: str, similarity: str) -> float:
    """Calculate the possibility of similarity based on ressource difference en tfidf similarity"""
    if int(ressource_diff) != 0:
        if int(ressource_diff) < int(similarity):
            return round((int(ressource_diff) / int(similarity)) * int(ressource_diff), 2)
        else:
            return round((int(similarity) / int(ressource_diff)) * int(similarity), 2)
    elif int(similarity) == 100:
        return 0.05
    else:
        return 0.5


def get_website(website: str) -> Optional[Response]:
    """Request the website pass in parameter"""
    url = f"http://{website}"

    try:
        return requests.get(url, verify=False, timeout=3)
    except urllib3.exceptions.NewConnectionError:
        return None
    except requests.exceptions.ConnectionError:
        return None
    except urllib3.exceptions.ReadTimeoutError:
        return None
    except Exception:
        # import traceback
        # traceback.print_exception(type(e), e, e.__traceback__)
        return None


def extract_text_ressource(website_text: str) -> Tuple[str, Dict[str, List[str]]]:
    """Extract the text from html and collect all ressources"""
    soup = BeautifulSoup(website_text, "lxml")

    text = text_from_html(website_text)
    ressource = website_ressource(soup)

    return text, ressource
