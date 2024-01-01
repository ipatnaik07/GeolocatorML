import io
import requests
import PIL.Image

import torch
from torch import nn
from torchvision.io import read_image
from geopy.geocoders import Nominatim

def get_loc():
    print()
    game_id = input('Game ID: ')
    url = f'https://www.geoguessr.com/api/v3/games/{game_id}'
    response = requests.get(url)
    json = response.json()
    #print(json)
    loc = json['rounds'][-1]
    print(json)
    return loc['lat'], loc['lng'], loc['heading'], loc['pitch']

def load_data():
    lat, long, heading, pitch = get_loc()
    info = geolocator.reverse(f'{lat},{long}')
    state = info.raw['address']['state']
    label = states.index(state)

    url = 'https://maps.googleapis.com/maps/api/streetview'
    key = 'AIzaSyA9Sa0lmHQ9TP-ST1bUgcLDEmU1BoPhwGc'
    payload = {'size': f'{width}x{height}',
               'location': f'{lat},{long}',
                'heading': heading,
                'pitch': pitch,
                'key': key}

    response = requests.get(url, params=payload)
    image = PIL.Image.open(io.BytesIO(response.content))
    image.save("test.png")
    image = read_image("test.png") / 255
    return image, label

class NeuralNetwork(nn.Module):
    def __init__(self, num_categories):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(width*height*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_categories)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

geolocator = Nominatim(user_agent="USA50K")
state_file = open('states.txt', 'r')
states = state_file.read().split('\n')
state_file.close()
width = 640
height = 256

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = NeuralNetwork(len(states)).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

x, y = load_data()
x = torch.unsqueeze(x, 0)
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    index = pred[0].tolist().index(torch.max(pred[0]).item())
    predicted, actual = states[index], states[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    print()