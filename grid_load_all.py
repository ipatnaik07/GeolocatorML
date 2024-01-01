from torch.utils.data import Dataset
from torchvision.io import read_image
from os import listdir
from locs import *
from squares import *

sq = 5
width = 640
height = 256
num_categories = len(squares) + 2
print(num_categories)

class GeoDataLoader(Dataset):
    def __init__(self, folder, num_images):
        self.images = []
        self.labels = []
        files = listdir(folder)
        files = files[:num_images]

        print()
        print('... loading images ...')
        for i in range(len(files)):
            file = files[i]
            image = read_image(f'{folder}/{file}') / 255
            self.images.append(image)
            index = int(file[:file.index('_')]) + 40000*(folder=="Testing")
            lat, long, heading, pitch, state = locs[index-1]
            square = (lat//sq*sq+sq, long//sq*sq)
            
            if state == "Alaska":
                self.labels.append(len(squares))
            elif state == "Hawaii":
                self.labels.append(len(squares) + 1)
            else:
                self.labels.append(squares.index(square))
            
            if (i+1) % 100 == 0:
                print(f'{i+1}/{len(files)}')
        print()
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label