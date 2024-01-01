from locs import *
from tkinter import *

sq = 5
squares = []
for i in range(25, 55, sq):
    for j in range(65, 130, sq):
        squares.append((i, -j))

occurrences = {square: 0 for square in squares}

for loc in locs:
    lat, long, heading, pitch, state = loc
    square = (int(lat//sq*sq)+sq, int(long//sq*sq))
    if square in squares:
        occurrences[square] += 1

for key in list(occurrences.keys()):
    if occurrences[key] < 1:
        occurrences.pop(key)

print(occurrences)
print(list(occurrences.keys()))
print(len(occurrences))
