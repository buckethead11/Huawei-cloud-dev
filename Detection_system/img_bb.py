import requests
import base64

API_KEY = "8af6298d8f104173e1320d50ea241a12"
IMAGE_PATH = f"snapshot.jpeg"
ALBUM_ID = ""

with open(IMAGE_PATH, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read())

url = "https://api.imgbb.com/1/upload"
payload = {
"key": API_KEY,
"image": encoded_image,
}
response = requests.post(url, payload)
response = response.json()
display_url = response["data"]["display_url"]

