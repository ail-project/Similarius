import argparse
import string

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import urllib3

from bs4 import BeautifulSoup
from bs4.element import Comment

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sklearn_text

import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


##############
# Similarity #
##############

stop_words = set(stopwords.words('english') + list(string.punctuation))

ENGLISH_STOP_WORDS = set( stopwords.words('english') ).union( set(sklearn_text.ENGLISH_STOP_WORDS) ).union(set(string.punctuation))

def sk_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), max_features=5000)
    tfidf = vectorizer.fit_transform([doc1, doc2])

    return round(((tfidf * tfidf.T).toarray())[0,1] * 100)


##################
# Web treatment #
#################

def tag_visible(element):
    """Identified element present in specific balise"""
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    """Extract text from web page and remove text present in specific balise"""
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" \n".join(t.strip() for t in visible_texts)


def find_list_resources(tag, attribute, soup):
    """Find ressource in web page list in attribute"""
    list = []
    for x in soup.findAll(tag):
        try:
            list.append(x[attribute])
        except KeyError:
            pass
    return(list)

def website_ressource(soup):
    """Collect all website's ressources"""
    ressource_dict = dict()
    ressource_dict["image_scr"] = find_list_resources('img',"src",soup)
    ressource_dict["script_src"] = find_list_resources('script',"src",soup)
    ressource_dict["css_link"] = find_list_resources("link","href",soup)
    ressource_dict["source_src"] = find_list_resources("source","src",soup)
    ressource_dict["a_href"] = find_list_resources("a","href",soup)
    ressource_dict["form_action"] = find_list_resources("form","action",soup)

    return ressource_dict


def ressource_difference(original_ressource, compare_ressource):
    """Difference between original website's ressources and comparaison website's ressources"""
    cp_total = 0
    cp_diff = 0

    for r_to_check in original_ressource:
        for r in original_ressource[r_to_check]:
            cp_total += 1
            if r_to_check in compare_ressource:
                if r not in compare_ressource[r_to_check]:
                    cp_diff += 1

    return str(int((cp_diff/cp_total)*100))

def ratio(ressource_diff, similarity):
    """Calculate the possibility of similarity based on ressource difference en tfidf similarity"""
    if int(ressource_diff) != 0:
        if int(ressource_diff) < int(similarity):
            return round((int(ressource_diff)/int(similarity))*int(ressource_diff), 2)
        else:
            return round((int(similarity)/int(ressource_diff))*int(similarity), 2)
    elif int(similarity) == 100:
        return 0.05
    else:
        return 0.5

def get_website(website):
    """Request the website pass in parameter"""
    url = f"http://{website}"

    try:
        return requests.get(url, verify=False, timeout=3)
    except urllib3.exceptions.NewConnectionError:
        return 
    except requests.exceptions.ConnectionError:
        return 
    except urllib3.exceptions.ReadTimeoutError:
        return 
    except Exception as e :
        # import traceback
        # traceback.print_exception(type(e), e, e.__traceback__)
        return 

def extract_text_ressource(website_text):
    """Extract the text from html and collect all ressources"""
    soup = BeautifulSoup(website_text, "html.parser")

    text = text_from_html(website_text)
    ressource = website_ressource(soup)

    return text, ressource


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--original", help="Website to compare")
    parser.add_argument("-w", "--website", nargs="+", help="Website to compare")
    args = parser.parse_args()

    # Original
    original = get_website(args.original)

    if not original:
        print("[-] The original website is unreachable...")
        exit(1)

    original_text, original_ressource = extract_text_ressource(original.text)

    for website in args.website:
        print(f"\n********** {args.original} <-> {website} **********")

        # Compare
        compare = get_website(website)

        if not compare:
            print(f"[-] {website} is unreachable...")
            continue
    
        compare_text, compare_ressource = extract_text_ressource(compare.text)

        # Calculate
        sim = str(sk_similarity(compare_text, original_text))
        print(f"\nSimilarity: {sim}")

        ressource_diff = ressource_difference(original_ressource, compare_ressource)
        print(f"Ressource Difference: {ressource_diff}")

        ratio_compare = ratio(ressource_diff, sim)
        print(f"Ratio: {ratio_compare}")
