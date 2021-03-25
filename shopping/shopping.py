import csv
import sys
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    #if len(sys.argv) != 2:
    #    sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data("shopping.csv")#sys.argv[1])

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    def cmonth(input_string):
        s = input_string.lower()
        if s in ["jan"]: return 0
        if s in ["feb"]: return 1
        if s in ["mar"]: return 2
        if s in ["apr"]: return 3
        if s in ["may"]: return 4
        if s in ["june"]: return 5
        if s in ["jul"]: return 6
        if s in ["aug"]: return 7
        if s in ["sep"]: return 8
        if s in ["oct"]: return 9
        if s in ["nov"]: return 10
        if s in ["dec"]: return 11

        raise Exception("Input String: {} is unknown".format(s))

    def cvisitortype(input_string):
        s = input_string.lower()
        if s in ["returning_visitor"]:
            return 1
        if s in ["new_visitor", "other"]:
            return 0

        raise Exception("Input String: {} is unknown".format(s))

    def cbln(input_string):
        s = input_string.lower()
        if s in ["true"]:
            return 1
        elif s in ["false"]:
            return 0

        raise Exception("Input String: {} is unknown".format(s))

    conversion_evidence= OrderedDict()
    conversion_evidence["Administrative"] = int
    conversion_evidence["Administrative_Duration"] = float
    conversion_evidence["Informational"] = int
    conversion_evidence["Informational_Duration"] = float
    conversion_evidence["ProductRelated"] = int
    conversion_evidence["ProductRelated_Duration"] = float
    conversion_evidence["BounceRates"] = float
    conversion_evidence["ExitRates"] = float
    conversion_evidence["PageValues"] = float
    conversion_evidence["SpecialDay"] = float
    conversion_evidence["Month"] = cmonth
    conversion_evidence["OperatingSystems"] = int
    conversion_evidence["Browser"] = int
    conversion_evidence["Region"] = int
    conversion_evidence["TrafficType"] = int
    conversion_evidence["VisitorType"] = cvisitortype
    conversion_evidence["Weekend"] = cbln

    evidence = []
    labels = []
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            evidence.append([conv(row[key]) for (key, conv) in conversion_evidence.items()])
            labels.append(cbln(row["Revenue"]))

    return evidence, labels



def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    return KNeighborsClassifier(n_neighbors=1).fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    true_positive_total = 0
    true_positive_matches = 0
    true_negative_total = 0
    true_negative_matches = 0

    for actual_label, predicted_label in zip(labels, predictions):

        if actual_label == 1:
            true_positive_total += 1
            if actual_label == predicted_label:
                true_positive_matches += 1
        else:
            true_negative_total += 1
            if actual_label == predicted_label:
                true_negative_matches += 1

    # In Python 3 integer division is handled as follows:
    # // => used for integer output
    # / => used for double output
    sensitivity = true_positive_matches / true_positive_total
    specificity = true_negative_matches / true_negative_total
    return sensitivity, specificity




if __name__ == "__main__":
    main()
