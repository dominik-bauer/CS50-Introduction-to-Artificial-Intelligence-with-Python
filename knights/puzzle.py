from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    Or(AKnave, AKnight),                             # A can only be one: a Knight or a Knave
    Implication(AKnave, Not(And(AKnave, AKnight))),  # Whatever a Knave says is wrong --> NOT
    Implication(AKnight,    And(AKnave, AKnight))    # Whatever a Knight says is correct
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    Or(AKnave, AKnight),  # A can only be a Knight or a Knave
    Or(BKnave, BKnight),  # B can only be a Knight or a Knave
    Implication(AKnave, Not(And(AKnave, BKnave))),  # Only B says something, here as a Knave
    Implication(AKnight,    And(AKnave, BKnave))    # Only B says something, here as a Knight
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."

# helper sentence
being_same = Or(And(AKnave,  BKnave),
                And(AKnight, BKnight))
being_different = Or(And(AKnave, BKnight),
                     And(AKnight, BKnave))

knowledge2 = And(
    Or(AKnave, AKnight),  # A can only be a Knight or a Knave
    Or(BKnave, BKnight),  # B can only be a Knight or a Knave

    # A expressing that both are of the same kind
    Implication(AKnave, Not(being_same)),
    Implication(AKnight,    being_same),

    # B expressing that both are of different kind
    Implication(BKnave,   Not(being_different)),
    Implication(BKnight,     being_different)
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."

# Helper sentence
A2 = And(Implication(AKnave, Not(AKnave)),
         Implication(AKnight,    AKnave))

knowledge3 = And(
    Or(AKnave, AKnight),  # A can only be a Knight or a Knave
    Or(BKnave, BKnight),  # B can only be a Knight or a Knave
    Or(CKnave, CKnight),  # C can only be a Knight or a Knave

    # B says some about A, as Knave and as a Knight, ...
    Implication(BKnave, Not(A2)),
    Implication(BKnight,    A2),

    # B also says ,as Knave and Knight, ...
    Implication(BKnave, Not(CKnave)),
    Implication(BKnight,    CKnave),

    # C speaks as Knave and Knight:
    Implication(CKnave, Not(AKnight)),
    Implication(CKnight,    AKnight)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
