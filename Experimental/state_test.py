from locs import *

state_file = open('states.txt', 'r')
states = state_file.read().split('\n')
state_file.close()

occurrences = {state: 0 for state in states}

for loc in locs:
    lat, long, heading, pitch, state = loc
    if state in states:
        occurrences[state] += 1

print(occurrences)
