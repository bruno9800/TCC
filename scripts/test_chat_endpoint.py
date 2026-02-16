import httpx
import json

url = "http://localhost:8000/chat/"
headers = {"Content-Type": "application/json"}
data = {
    "message": "Quais são os direitos dos discentes?",
    "history": [],
    "top_k": 5,
    "filter_revoked": True
}

try:
    response = httpx.post(url, json=data, timeout=30.0)
    response.raise_for_status()
    print("Status Code:", response.status_code)
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
except httpx.RequestError as e:
    print(f"Request Error: {e}")
except httpx.HTTPStatusError as e:
    print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
