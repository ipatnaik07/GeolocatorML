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
        self.images = []
        self.labels = []
        files = listdir(folder)
        files = files[:num_images]

        coords = []
        for loc in locs:
            coords.append([loc[0], loc[1]])
        data = array(coords)
        kmeans = KMeans(n_clusters=num_categories, random_state=0, n_init='auto').fit(data)

        print()
        print('... loading images ...')
        for i in range(len(files)):
            file = files[i]
            image = read_image(f'{folder}/{file}') / 255
            self.images.append(image)
            index = int(file[:file.index('_')]) + 40000*(folder=="Testing")
            self.labels.append(kmeans.labels_[index-1])
            if (i+1) % 100 == 0:
                print(f'{i+1}/{len(files)}')
        print()
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label
