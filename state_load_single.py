from torch.utils.data import Dataset
from torchvision.io import read_image
from os import listdir
from locs import *

state_file = open('states.txt', 'r')
states = state_file.read().split('\n')
state_file.close()

width = 640
height = 256
num_categories = len(states)

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
        state = file[file.index('_')+1:file.index('.')]
        label = states.index(state)
        return image, label

    def get_coords(self, index):
        file = self.files[index]
        index = int(file[:file.index('_')]) + 40000*(self.folder=="Testing")
        lat, long, heading, pitch, state = locs[index-1]
        return lat, long
