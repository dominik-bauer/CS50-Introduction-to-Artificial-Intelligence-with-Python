import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

# nltk.download('stopwords')


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    # remove trailing back/slashes
    if directory[-1] in ["\\", "/"]:
        directory = directory[:-1]

    # Loop all subdirectories
    corpus_dict = {}

    for file in os.scandir(directory):

        # skip non-file objects like subdirectories
        if not file.is_file():
            continue

        # load the contents of the text file into a string
        str_file_content = ''
        with open(file, 'r', encoding='utf-8') as x:
            str_file_content = ''.join(x.readlines())

        # skip empty text files
        if str_file_content:
            corpus_dict[file.name] = str_file_content

    return corpus_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    # make all characters lowercase
    document2 = document.lower()

    # remove punctuation (using maketrans, as it handles the conversion to unicode)
    punctuation_translate_table = str.maketrans("", "", string.punctuation)
    document2 = document2.translate(punctuation_translate_table)

    # directly tokenize to words, as the removal of punctuation
    # already eliminated any information about sentences
    words = nltk.word_tokenize(document2)

    # remove all stopwords (note: this would also remove a
    # lot of punctuation characters, if it had been done at the beginning)
    stopwords = nltk.corpus.stopwords.words("english")
    words = [w for w in words if w not in stopwords]

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    idfs = {}

    # get all words first
    unique_words = set().union(*documents.values())

    # number of documents
    nd = len(documents)

    # loop every word/term found in all the documents
    for term in unique_words:

        # determine in how many documents the word/term occurs
        ft = 0
        for words in documents.values():
            if term in words:
                ft = ft + 1

        # calculate idf
        idfs[term] = math.log10(nd/ft)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    ranking = {}
    for file, file_words in files.items():

        # starting at zero, as we will sum um the tf-idfs
        ranking[file] = 0

        # calculate the tf-idf for each term and add it to the ranking dict
        for term in query:

            # only consider words that occur in the file
            if term in file_words:
                # term frequency * inverse document frequency
                ranking[file] += file_words.count(term) * idfs[term]

    # return empty list if all ranking values are 0
    # if not any(ranking.items()):
    #     return []

    ranking_sorted = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

    # finally return top n files
    return [x[0] for x in ranking_sorted][:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    ranking = []
    for sentence, sentence_words in sentences.items():

        idf = 0
        n_matches = 0

        # sum up inverse document frequency
        for term in query:
            if term in sentence_words:
                idf += idfs[term]

            # calculate matches for query term density
            n_matches += sentence_words.count(term)

        # calculate query term density
        qtd = n_matches / len(sentence_words)

        # put sentence and its related idf and qtd into a list
        ranking.append([sentence, idf, qtd])

    # sort first by idf then by qtd
    ranking_sorted = sorted(ranking, key=lambda x: (x[1], x[2]), reverse=True)

    return [x[0] for x in ranking_sorted[:n]]


if __name__ == "__main__":
    main()
