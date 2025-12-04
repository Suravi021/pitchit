import requests
import json

url = "http://localhost:5000/detect"
payload = {
    "text": "This is a simple example text written by a human."
}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Response JSON:")
print(json.dumps(response.json(), indent=4))
