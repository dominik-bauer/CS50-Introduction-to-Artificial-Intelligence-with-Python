import sys
from collections import deque
from crossword import *
import copy


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var, domain in self.domains.items():
            # avoiding the repeated call of remove() by directly generating
            # a new set of words with correct length via set comprehension
            words_with_correct_length = set(word for word in domain if len(word) == var.length)
            self.domains[var] = words_with_correct_length

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        bln_revised = False

        # loop a copy in order to directly modify the domain
        for word in copy.copy(self.domains[x]):
            if not self.are_there_matching_words_in_y_domain(x, y, word):
                self.domains[x].remove(word)
                bln_revised = True

        return bln_revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        # "begin with initial list of all arcs in the problem"
        if arcs is None:
            arcs = self.crossword.overlaps

        arcs_queue = deque()
        for arc in arcs:
            arcs_queue.append(arc)

        while arcs_queue:
            v1, v2 = arcs_queue.pop()

            if not self.crossword.overlaps[(v1, v2)]:
                continue  # there is no arc

            if self.revise(v1, v2):
                if not self.domains[v1]:  # domain is empty
                    return False

                neighbor_variables = self.crossword.neighbors(v1) - {v2}
                for vn in neighbor_variables:
                    arcs_queue.append((vn, v1))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """

        # check if number of assignments matches exactly the number of words in the cross-word puzzle
        if len(self.crossword.variables) != len(assignment):
            return False

        # some additional checking
        for var, word in assignment.items():
            if not word:
                return False
            if var not in self.crossword.variables:
                return False

        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        # check if all values are distinct, i.e. that there are no duplicate words within the cross-word puzzle
        words = list(assignment.values())
        if len(words) != len(set(words)):
            return False

        # every value is the correct length?
        for var, word in assignment.items():
            if var.length != len(word):
                return False

        # Checking for conflicts between neighboring variables
        # Kept the inefficient loop, as the nested loops will be small and readability is prioritized
        for v1 in assignment:
            for v2 in self.crossword.neighbors(v1):

                # skip any variable that is not present in the assignment as the
                # specification states "[...] that the assignment may not be complete."
                if v2 not in assignment:
                    continue

                # Now that v1 and v2 are in the assignment, the overlapping character can be checked:
                i1, i2 = self.crossword.overlaps[(v1, v2)]
                if assignment[v1][i1] != assignment[v2][i2]:
                    return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        # for readability and consistency in naming
        v1 = var

        # get the variables that are still not assigned and overlap with the input var
        neighbor_variables = self.crossword.neighbors(v1) - set(assignment.keys())
        
        # create dictionary in which removals are counted
        removal_count = {word: 0 for word in self.domains[v1]}
        
        # now loop relevant neighbour variables
        for v2 in neighbor_variables:

            # get overlap indices
            n1, n2 = self.crossword.overlaps[v1, v2]

            # loop all word combinations
            for word1 in self.domains[v1]:
                for word2 in self.domains[v2]:
                    if word1[n1].upper() != word2[n2].upper():
                        removal_count[word1] += 1

        foo = sorted(removal_count.items(), key=lambda x:x[1])

        return [x[0] for x in foo]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_variables = self.crossword.variables - set(assignment.keys())

        remaining_values = dict()
        for var in unassigned_variables:
            domain_length = len(self.domains[var])
            if domain_length in remaining_values:
                remaining_values[domain_length].append(var)
            else:
                remaining_values[domain_length] = [var]

        m = min(remaining_values.keys())
        if len(remaining_values[m]) == 1:
            return remaining_values[m][0]

        highest_degree = dict()
        for var in remaining_values[m]:
            d = len(self.crossword.neighbors(var))
            if d in highest_degree:
                highest_degree[d].append(var)
            else:
                highest_degree[d] = [var]

        m = max(highest_degree.keys())
        return highest_degree[m][0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):

            foo = dict(assignment)
            foo[var] = value
            if self.consistent(foo):
                assignment[var] = value
                result = self.backtrack(assignment)
                if result:
                    return result
                assignment.pop(var)
        return None

    def are_there_matching_words_in_y_domain(self, x, y, word):
        """ Takes an x and y Variable object and a target word of the domain of the x variable. Then it checks if the
            words of the y variable domain satisfy the overlap condition for that single target word. As soon as a
            matching word is found within the domain of the y Variable the method returns true, otherwise false. """

        # get overlap indexes
        indexes = self.crossword.overlaps[(x, y)]

        # just in case
        if not indexes:
            raise Exception("No overlap")

        nx, ny = indexes
        return is_character_in_words(self.domains[y], word[nx], ny)


def is_character_in_words(input_strings, target_character, target_index):
    """ Takes a list of strings (input_strings) and checks if the target_character is present at the target_index
        for each word. As soon as there is at least one match the function returns true, otherwise false"""

    for word in input_strings:
        if word[target_index].upper() == target_character.upper():
            return True
    return False


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
