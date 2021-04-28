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

NONTERMINALS = """

S -> NP VP | NP VP | NP Conj NP
NP -> N | Det N | Adj NP | P NP  | Det N | Det Adj NP | NP VP | NP Adv VP | NP P NP
VP -> V | VP Det NP | VP P NP | P VP

"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


sentences_list = """Holmes sat.
Holmes sat in the armchair.
Holmes lit a little red pipe.
Holmes sat in the red armchair and he chuckled.
I had a little moist red paint in the palm of my hand.
We arrived the day before Thursday.
My companion smiled an enigmatical smile.
Holmes chuckled to himself.
She never said a word until we were at the door here.
Holmes sat down and lit his pipe.
I had a country walk on Thursday and came home in a dreadful mess.
"ABV asklf asdfk483 asdflj...""".split("\n")


def main():
    if False:
        # If filename specified, read sentence from file
        if len(sys.argv) == 2:
            with open(sys.argv[1]) as f:
                s = f.read()

        # Otherwise, get sentence as input
        else:
            s = input("Sentence: ")
    for s in sentences_list:
        # Convert input into list of words
        s = preprocess(s)

        print(s)

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
    return []



if __name__ == "__main__":
    main()
