import requests

API_KEY = "AIzaSyBGGemWaCifOCNsyerKNYDDY111KqjqLxo"

url = "https://translation.googleapis.com/language/translate/v2"

params = {
    "key": API_KEY
}

data = {
    "q": "Привет мир",
    "target": "en",
    "source": "ru",
    "format": "text"
}

response = requests.post(url, params=params, json=data)

print("STATUS:", response.status_code)
print("RESPONSE:")
print(response.json())