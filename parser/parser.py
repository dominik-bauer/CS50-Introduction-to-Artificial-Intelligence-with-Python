import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# NONTERMINALS = """
# S    -> NP VP | NP Conj NP | NP Conj VP
# NP   -> N | P Det NP | N Adv | Det AdjC NP | NP VP | Det N
# VP   -> V | V Adv | VP P NP | VP NP |
# AdjC -> Adj | AdjC Adj
# """
# NONTERMINALS = """
# S  -> NP VP | NP Conj NP | NP Conj VP
# NP -> N     | Det N | Adj NP | P Det NP | N Adv | Det NP | Det Adj NP | NP VP | NP Adv VP | NP P NP | NP VP
# VP -> V     | Adv V | V Adv | VP Det NP | VP P NP | P VP | VP NP
# """

# After brute forcing several rule combinations, I found a set of rules that work for me:
NONTERMINALS = """
S  -> NP VP | NP Conj NP | NP Conj VP
NP -> N | P Det N | N Adv | Det AdjC NP | NP VP | Det N
VP -> V | V Adv | VP P NP | VP NP | VP Adv
AdjC -> Adj   | AdjC Adj
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    # Initially some nltk resources must be downloaded
    # nltk.download('punkt')

    # Get list of words without any modifications
    words = nltk.word_tokenize(sentence)

    # This could be way simpler, but the specification was very clear about this
    # now words like "Super2000" will not be removed.
    return [word.lower() for word in words if any([char.isalpha() for char in word])]


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """

    list_of_np_chunks = []

    # go through all trees and its leaves
    for st in tree.subtrees():

        # skip if the label is not NP
        if st.label() != "NP":
            continue

        # skip if there is another NP below this node/leaf
        if has_np_below(st):
            continue

        # Now st must be of the type noun phrase and also
        # it does not contain any noun phrases within itself
        list_of_np_chunks.append(st)

    return list_of_np_chunks


def has_np_below(tree):
    labels = set()

    # fetch all labels below tree2
    for st in tree.subtrees():

        # skip the initial tree itself, as this can already be "NP" (i.e. skipping the first loop)
        # the following two lines simplified the code A LOT!!!
        if st == tree:
            continue

        # get a complete set of labels (easier for understanding and debugging)
        labels.add(st.label())

    return "NP" in labels


if __name__ == "__main__":
    main()
