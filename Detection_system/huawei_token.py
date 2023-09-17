import requests
import json

# Define the URL for authentication
url = "https://iam.ap-southeast-3.myhuaweicloud.com/v3/auth/tokens"

# Define the headers for the request
headers = {
    "Content-Type": "application/json"
}

# Define the authentication payload as a dictionary
auth_payload = {
    "auth": {
        "identity": {
            "methods": ["password"],
            "password": {
                "user": {
                    "name": "Aintnowei_1",
                    "password": "Aintnowei69",
                    "domain": {
                        "name": "jaxcodey"
                    }
                }
            }
        },
        "scope": {
            "project": {
                "name": "ap-southeast-3"
            }
        }
    }
}

# Convert the payload to JSON format
json_payload = json.dumps(auth_payload)

# Send the POST request to obtain the token
try:
    response = requests.post(url, headers=headers, data=json_payload)

    # Check if the request was successful (status code 201 indicates success)
    if response.status_code == 201:
        # The X-Subject-Token header contains the token
        x_subject_token = response.headers["X-Subject-Token"]
        print("X-Subject-Token:", x_subject_token)
    else:
        print(f"Request failed with status code: {response.status_code}")
        print("Error message:", response.text)  # Print the error message
except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")
