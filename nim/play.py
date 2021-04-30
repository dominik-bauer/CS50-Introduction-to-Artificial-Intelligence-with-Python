from nim import train, play

ai = train(10000)

# output some states and actions sorted by their q-value
for q, sa in zip(sorted(ai.q, key=ai.q.__getitem__), sorted(ai.q.values())):
    print(q, sa)

play(ai)
