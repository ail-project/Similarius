import argparse

from .similarius import sk_similarity
from .similarius import text_from_html  # noqa
from .similarius import find_list_resources  # noqa
from .similarius import ressource_difference
from .similarius import ratio
from .similarius import get_website
from .similarius import extract_text_ressource


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--original", help="Website to compare")
    parser.add_argument("-w", "--website", nargs="+", help="Website(s) to compare")
    args = parser.parse_args()

    # Original
    original = get_website(args.original)

    if not original:
        parser.print_help()
        print("[-] The original website is not set")
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
