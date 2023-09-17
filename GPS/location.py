import googlemaps
from datetime import datetime
from config import api_key

gmaps = googlemaps.Client(key=api_key)

current_location = gmaps.geolocate()

# Access the latitude and longitude of the current location
latitude = current_location['location']['lat']
longitude = current_location['location']['lng']

# Print the current location
print(f"Latitude: {latitude}, Longitude: {longitude}")
