from locs import *
from squares import *
from sklearn.cluster import KMeans
from numpy import array

state_file = open('states.txt', 'r')
states = state_file.read().split('\n')
state_file.close()

state_avg = [[0, 0, 0] for i in range(len(states))]
for loc in locs:
    lat, long, heading, pitch, state = loc
    if state != 'Error':
        i = states.index(state)
        state_avg[i][0] += lat
        state_avg[i][1] += long
        state_avg[i][2] += 1
state_avg = [(avg[0]/avg[2], avg[1]/avg[2]) for avg in state_avg]

sq = 5
square_avg = [[0, 0, 0] for i in range(len(squares) + 2)]
for loc in locs:
    lat, long, heading, pitch, state = loc
    square = (lat//sq*sq+sq, long//sq*sq)
    if state == "Alaska":
        i = len(squares)
    elif state == "Hawaii":
        i = len(squares) + 1
    else:
        i = squares.index(square)
    square_avg[i][0] += lat
    square_avg[i][1] += long
    square_avg[i][2] += 1
square_avg = [(avg[0]/avg[2], avg[1]/avg[2]) for avg in square_avg]
grid_avg = square_avg

num_clusters = 50
coords = []
for loc in locs:
    coords.append([loc[0], loc[1]])
data = array(coords)
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(data)

cluster_avg = [[0, 0, 0] for i in range(num_clusters)]
for index, loc in enumerate(locs):
    lat, long, heading, pitch, state = loc
    i = kmeans.labels_[index]
    cluster_avg[i][0] += lat
    cluster_avg[i][1] += long
    cluster_avg[i][2] += 1
cluster_avg = [(avg[0]/avg[2], avg[1]/avg[2]) for avg in cluster_avg]

print(state_avg)
print()
print(square_avg)
print()
print(cluster_avg)
