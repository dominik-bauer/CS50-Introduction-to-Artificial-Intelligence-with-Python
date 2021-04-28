import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def has_parents(people_dict, name):
    person_info = people_dict[name]
    if not person_info['mother'] and not person_info['father']:
        # no parents at all
        return False

    elif person_info['mother'] and person_info['father']:
        # person with two parents
        return True

    else:
        # case with only one parent
        raise NotImplementedError("Case with only one Parent is not supported")


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    def get_index_gene(name):
        if name in one_gene:
            return 1
        elif name in two_genes:
            return 2
        else:
            return 0

    def get_index_trait(name):
        return name in have_trait

    def get_passing_probability(number_of_genes):
        if number_of_genes == 0:
            return PROBS["mutation"]
        if number_of_genes == 1:
            return 0.5
        if number_of_genes == 2:
            return 1-PROBS["mutation"]

    def combine_parents_passing_probabilities(passing_probability_father,
                                              passing_probability_mother,
                                              persons_number_of_genes):

        if persons_number_of_genes == 2:
            # both parents must pass a gene
            return passing_probability_mother * passing_probability_father

        if index_gene == 1:
            # One parents passes the other does not
            s1 = (1 - passing_probability_mother) * passing_probability_father
            s2 = (1 - passing_probability_father) * passing_probability_mother
            return s1 + s2

        if index_gene == 0:
            # no one is passing
            return (1-passing_probability_mother) * (1-passing_probability_father)

        raise Exception("Invalid number of genes: ", persons_number_of_genes)

    # container for all probabilities to be multiplied
    probabilities = []

    for person in people:

        index_gene = get_index_gene(person)
        index_trait = get_index_trait(person)
        father = people[person]["father"]
        mother = people[person]["mother"]

        # 1) universally add the trait probability
        probabilities.append(PROBS["trait"][index_gene][index_trait])

        # 2) now determine the gene probability
        if not has_parents(people, person):
            # 2.1) if no parents are present add unconditional probability
            probabilities.append(PROBS["gene"][index_gene])

        else:
            # 2.2) if parents are present add conditional probability

            # Passing Probabilities of each parent
            ppf = get_passing_probability(get_index_gene(father))
            ppm = get_passing_probability(get_index_gene(mother))

            # Based on parents passing probabilities and the "target" number of genes of the person
            probability_with_parents = combine_parents_passing_probabilities(ppf, ppm, index_gene)

            probabilities.append(probability_with_parents)

    probability = 1
    for val in probabilities:
        probability *= val

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """

    for person, distributions in probabilities.items():
        for dist_name, probability_dict in distributions.items():
            normalization_factor = 1. / sum(probability_dict.values())
            for k in probability_dict:
                probability_dict[k] *= normalization_factor


if __name__ == "__main__":
    main()
