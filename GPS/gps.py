import requests
from config import api_key
url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={api_key}'

response = requests.post(url)

if response.status_code == 200:
    data = response.json()
    latitude = data['location']['lat']
    longitude = data['location']['lng']
    accuracy = data['accuracy']
    print(f'Latitude: {latitude}, Longitude: {longitude}, Accuracy: {accuracy} meters')
else:
    print('Failed to retrieve location data. Status code:', response.status_code)