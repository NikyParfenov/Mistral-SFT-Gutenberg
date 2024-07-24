import http.client
import json
import os

def run_google_search(prompt):
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
    "q": prompt,
    "page": 1,
    "num": 20
    })
    headers = {
    'X-API-KEY': os.environ['SERPER_KEY'],
    'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    
    return str([f"TITLE: {item['title']}; SNIPPET: {item['snippet']}" for item in json.loads(data.decode("utf-8"))['organic']])