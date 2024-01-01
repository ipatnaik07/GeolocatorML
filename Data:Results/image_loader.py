import io
import json
import random
import requests
import PIL.Image
from urlsigner import sign_url
#from geopy.geocoders import Nominatim

def save_image(loc, index, folder):
    lat, long, heading, pitch, state = loc
    url = base_url + (f'?size={width}x{height}' +
                      f'&location={lat},{long}' +
                      f'&heading={heading}' +
                      f'&pitch={pitch}' +
                      f'&key={key}')

    try:
        signed_url = sign_url(input_url=url, secret=secret)
        response = requests.get(signed_url)
        image = PIL.Image.open(io.BytesIO(response.content))
        image.save(f'{folder}/{index}_{state}.png')
    except:
        print(index, 'image error')

def get_state(lat, long, index):
    url = base_url + f'?coordinates={lat},{long}&publishableKey={key}'
    try:
        response = requests.get(url)
        info = response.json()
        return info['addresses'][0]['state']
    except:
        print(index, 'address error')
        return 'Error'

states = open('USA50K.json', 'r')
data = json.load(states)
states.close()

coords = data['customCoordinates']
base_url = 'https://api.radar.io/v1/geocode/reverse'
key = 'prj_live_pk_d174033143925c7bef1f8597cd00d013178da79f'

locs = []
for i in range(len(coords)):
    c = coords[i]
    state = get_state(c['lat'], c['lng'], i)
    locs.append((c['lat'], c['lng'], c['heading'], c['pitch'], state))
    print(state)
random.shuffle(locs)

base_url = 'https://maps.googleapis.com/maps/api/streetview'
key = 'AIzaSyA9Sa0lmHQ9TP-ST1bUgcLDEmU1BoPhwGc'
secret = 'ZCYcGgoZRn9Wr9Zf3_HNPu-ByME='
width = 640
height = 256

for i in range(len(locs) * 4 // 5):
    save_image(locs[i], i + 1, 'Training')

for j in range(i + 1, len(locs)):
    save_image(locs[j], j - i, 'Testing')
