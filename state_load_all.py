from torch.utils.data import Dataset
from torchvision.io import read_image
from os import listdir

state_file = open('states.txt', 'r')
states = state_file.read().split('\n')
state_file.close()

width = 640
height = 256
num_categories = len(states)

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
            state = file[file.index('_')+1 : file.index('.')]
            self.images.append(image)
            self.labels.append(states.index(state))
            if (i+1) % 100 == 0:
                print(f'{i+1}/{len(files)}')
        print()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label
