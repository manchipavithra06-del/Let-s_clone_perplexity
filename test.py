import urllib.request
import json
url = "http://127.0.0.1:8000/chat"
data = json.dumps({"message": "Say exactly the word 'testing'"}).encode('utf-8')
headers = {'Content-Type': 'application/json'}
req = urllib.request.Request(url, data=data, headers=headers)
try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode('utf-8'))
except Exception as e:
    print("Error:", e)
