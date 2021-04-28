import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    links = corpus[page]
    n_links = len(links)
    pages = corpus.keys()
    n_pages = len(pages)

    if n_links == 0:
        return {k: 1/n_pages for k in pages}

    probability_link = damping_factor / n_links
    probability_page = (1-damping_factor) / n_pages

    probability_dict = dict()
    for p in pages:
        probability_dict[p] = probability_page
    for p in links:
        probability_dict[p] += probability_link

    return probability_dict


def get_corpus_with_corrected_links(corpus):
    """ returns a corpus in which a page without links is filled with all possible links including itself"""
    corpus_mod = corpus.copy()
    for k, v in corpus_mod.items():
        if not v:
            corpus_mod[k] = set(corpus.keys())

    return corpus_mod


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # first handle pages without links
    # corpus_mod = get_corpus_with_corrected_links(corpus)
    corpus_mod = corpus.copy()

    pages_all = list(corpus_mod.keys())
    page_current = random.choice(list(corpus_mod.keys()))

    page_counter = {k: 0 for k in pages_all}
    for _ in range(n):

        #add 1 to the page counter
        page_counter[page_current] += 1

        # get transition model
        tm = transition_model(corpus_mod, page_current, damping_factor)

        # choose randomly among transition model probabilities
        weightings = list(tm.values())
        page_current = random.choices(pages_all, weightings, k=1)[0]

    return {k: v/n for k, v in page_counter.items()}


def update_page_ranks(page_ranks, corpus, damping_factor):
    """updates all page ranks within page_ranks dictionary
       returns a list of error factors: new/old"""

    return [update_page_rank(page_ranks, page, corpus, damping_factor) for page in page_ranks]


def update_page_rank(page_ranks, page, corpus, damping_factor):
    """updates the page_rank dictionary once for a specified page
       returns the error between old and new: abs(rank_new / rank_old - 1) """

    # get all pages that link to "page"
    pages_that_link = []
    for k, v in corpus.items():
        if page in v: pages_that_link.append(k)

    # get number of links on each page
    num_links = {k: len(v) for k, v in corpus.items()}

    rank_old = page_ranks[page]

    # calculate all terms of the formula (use list comprehension for the sum)
    C1 = (1-damping_factor) / len(corpus)
    C2 = sum([page_ranks[pl] / num_links[pl] for pl in pages_that_link])
    rank_new = C1 + damping_factor * C2

    # update the page ranks dict
    page_ranks[page] = rank_new

    # return the error between old and new value
    return abs(rank_new / rank_old - 1)



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # first handle pages without links
    corpus_mod = get_corpus_with_corrected_links(corpus)

    # Initialize page rank dict
    page_ranks = {p: 1 / len(corpus_mod) for p in corpus_mod.keys()}

    max_error = 10
    while max_error > .001:
        errors = update_page_ranks(page_ranks, corpus_mod, damping_factor)
        max_error = max(errors)

    return page_ranks


if __name__ == "__main__":
    main()
