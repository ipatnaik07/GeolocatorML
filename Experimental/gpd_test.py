from sklearn.cluster import KMeans
from numpy import array
from locs import *
from squares import *

import pandas as pd
import folium
from random import shuffle

coords = []
for loc in locs:
    coords.append([loc[0], loc[1]])
data = array(coords)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
          'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
          'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
          'gray', 'black', 'lightgray']
colors = ['red', 'green', 'blue', 'orange', 'purple', 'lightgray']
shuffle(colors)

my_map = folium.Map(location=(50, -100), zoom_start=3)
folium.TileLayer('CartoDB dark_matter').add_to(my_map)

'''
kmeans = KMeans(n_clusters=50, random_state=0, n_init='auto').fit(data)
print(len(kmeans.labels_))
print(min(kmeans.labels_), max(kmeans.labels_))
for i in range(5000):
    folium.Marker(coords[i], icon=folium.Icon(color=colors[kmeans.labels_[i] % len(colors)], icon='')).add_to(my_map)
'''

'''
sq = 5
for i in range(5000):
    lat, long, heading, pitch, state = locs[i]
    square = (lat//sq*sq+sq, long//sq*sq)
    if state == "Alaska":
        index = len(squares)
    elif state == "Hawaii":
        index = len(squares) + 1
    else:
        index = squares.index(square)
    folium.Marker(coords[i], icon=folium.Icon(color=colors[index % len(colors)], icon='')).add_to(my_map)
'''

#'''
state_file = open('states.txt', 'r')
states = state_file.read().split('\n')
state_file.close()
for i in range(5000):
    index = states.index(locs[i][4])
    folium.Marker(coords[i], icon=folium.Icon(color=colors[index % len(colors)], icon='')).add_to(my_map)
#'''

my_map.save('map5.html')
