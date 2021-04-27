import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """

        if len(self.cells) == self.count:
            return self.cells
        else:
            return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """

        if self.count == 0:
            return self.cells
        else:
            return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """

        if cell in self.cells:
            self.cells.remove(cell)
            self.count = self.count - 1  # mine count must be reduced accordingly

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """

        if cell in self.cells:
            self.cells.remove(cell)
            # self.count stays the same as only mines are counted


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def _get_set_of_nearby_cells(self, cell):
        nearby_cells = set()

        # get the surrounding rows and columns index values
        row_start = cell[0] - 1
        row_end = cell[0] + 2
        col_start = cell[1] - 1
        col_end = cell[1] + 2

        for r in range(row_start, row_end):
            for c in range(col_start, col_end):

                cell_current = (r, c)

                # Ignore the cell itself
                if cell_current == cell: continue

                # check for board boundaries and add to the set
                if 0 <= r < self.height and 0 <= c < self.width:
                    nearby_cells.add(cell_current)

        if not (3 <= len(nearby_cells) <= 8):
            raise Exception("to many or to little nearby cells found")

        return nearby_cells

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.
        """

        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)

        # 2) mark the cell as safe
        self.mark_safe(cell)

        # 3) add a new sentence to the AI's knowledge base based on the value of `cell` and `count`
        neighboring_cells = self._get_set_of_nearby_cells(cell)
        sentence_new = Sentence(neighboring_cells, count)

        # update the new sentence with the current knowledge of mines
        for cell in self.mines:
            sentence_new.mark_mine(cell)

        # update the sentence with the current knowledge of safe cells
        for cell in self.safes:
            sentence_new.mark_safe(cell)

        self.knowledge.append(sentence_new)

        # 4) mark any additional cells as safe or as mines if it can be concluded based on the AI's knowledge base
        tmp_safes = set()
        tmp_mines = set()
        for sentence in self.knowledge:
            tmp_mines.update(sentence.known_mines())
            tmp_safes.update(sentence.known_safes())

        for c in tmp_mines:
            self.mark_mine(c)
        for c in tmp_safes:
            self.mark_safe(c)

        # 5) add any new sentences to the AI's knowledge base if they can be inferred from existing knowledge
        for sentence_new in self.get_subset_sentences():
            self.knowledge.append(sentence_new)

    def get_subset_sentences(self):

        # combine all sentences in the knowledge base
        for sa, sb in itertools.product(self.knowledge, repeat=2):

            # Skip unnecessary combinations
            if sa.cells == set() or sb.cells == set() or sa == sb:
                continue

            # check for subset (only one direction, as itertools will combine the other way around
            if sa.cells.issubset(sb.cells):
                new_sentence = Sentence(sb.cells - sa.cells, sb.count - sa.count)

                # and only return sentence if it is unknown
                if new_sentence not in self.knowledge:
                    yield new_sentence

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """

        # get all cells
        candidates = self.safes - self.moves_made

        if candidates:
            return candidates.pop()
        else:
            return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """

        # at first get all cells
        all_cells = set()
        for r in range(self.height):
            for c in range(self.width):
                all_cells.add((r, c))

        # determine unknown cells by removing all known cells
        unknown_cells = list(all_cells - self.mines - self.moves_made - self.safes)

        if unknown_cells:
            return random.choice(unknown_cells)
        else:
            return None
