from torch.utils.data import Dataset
from torchvision.io import read_image
from os import listdir
from locs import *
from sklearn.cluster import KMeans
from numpy import array

width = 640
height = 256
num_categories = 50

class GeoDataLoader(Dataset):
    def __init__(self, folder, num_images):
        self.folder = folder
        files = listdir(folder)
        self.files = files[:num_images]

        coords = []
        for loc in locs:
            coords.append([loc[0], loc[1]])
        data = array(coords)
        kmeans = KMeans(n_clusters=num_categories, random_state=0, n_init='auto').fit(data)
        self.labels = kmeans.labels_

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        image = read_image(f'{self.folder}/{file}') / 255
        index = int(file[:file.index('_')]) + 40000*(self.folder=="Testing")
        return image, self.labels[index-1]
    
    def get_coords(self, index):
        file = self.files[index]
        index = int(file[:file.index('_')]) + 40000*(self.folder=="Testing")
        lat, long, heading, pitch, state = locs[index-1]
        return lat, long