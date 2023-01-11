# Similarius

Similarius is a Python library to compare web page and evaluate the level of similarity.

The tool can be used as a stand-alone tool or to feed other systems.



# Requirements

- Python 3.8+
- [Requests](https://github.com/psf/requests)
- [Scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [Beautifulsoup4](https://pypi.org/project/beautifulsoup4/)
- [nltk](https://github.com/nltk/nltk)



# Installation

## Source install

**Similarius** can be install with poetry. If you don't have poetry installed, you can do the following `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python`.

~~~bash
$ poetry install
$ poetry shell
$ similarius -h
~~~

## pip installation

~~~bash
$ pip3 install similarius
~~~



# Usage

~~~bash
dacru@dacru:~/git/Similarius/similarius$ similarius --help
usage: similarius.py [-h] [-o ORIGINAL] [-w WEBSITE [WEBSITE ...]]

optional arguments:
  -h, --help            show this help message and exit
  -o ORIGINAL, --original ORIGINAL
                        Website to compare
  -w WEBSITE [WEBSITE ...], --website WEBSITE [WEBSITE ...]
                        Website to compare
~~~



# Usage example

~~~bash
dacru@dacru:~/git/Similarius/similarius$ similarius -o circl.lu -w europa.eu circl.eu circl.lu
~~~



# Used as a library

~~~python
import argparse
from similarius import get_website, extract_text_ressource, sk_similarity, ressource_difference, ratio

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--website", nargs="+", help="Website to compare")
parser.add_argument("-o", "--original", help="Website to compare")
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
~~~

