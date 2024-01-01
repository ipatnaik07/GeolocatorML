from torch.utils.data import Dataset
from torchvision.io import read_image
from os import listdir
from locs import *
from squares import *

sq = 5
width = 640
height = 256
num_categories = len(squares) + 2

class GeoDataLoader(Dataset):
    def __init__(self, folder, num_images):
        self.folder = folder
        files = listdir(folder)
        self.files = files[:num_images]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        image = read_image(f'{self.folder}/{file}') / 255
        index = int(file[:file.index('_')]) + 40000*(self.folder=="Testing")
        lat, long, heading, pitch, state = locs[index-1]
        square = (lat//sq*sq+sq, long//sq*sq)
        
        if state == "Alaska":
            return image, len(squares)
        elif state == "Hawaii":
            return image, len(squares) + 1
        else:
            return image, squares.index(square)
    
    def get_coords(self, index):
        file = self.files[index]
        index = int(file[:file.index('_')]) + 40000*(self.folder=="Testing")
        lat, long, heading, pitch, state = locs[index-1]
        return lat, long
