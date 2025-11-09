
import json
import os
import requests

# create empty response object
resp = []

# query requests config
url = "http://localhost:8080/query"
tableName = "product"
headers = {
    "Content-Type": "application/json"
}

queries_json_file = "queries_synth_train.json"

with open(queries_json_file, "r") as f:
    data = json.load(f)

queries = [item["query"] for item in data]
query_ids = [item["query_id"] for item in data]

for i, query in enumerate(queries):
    print(f"Query {i}")
    payload = {
        "tableName": tableName,
        "query": query
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
    except:
        print(f"Missed query: {query}")
        continue
    
    # collect response, should be an array list of 30 strings
    resp_list = response.json()
    # loop through 30 strings, with enumeration j
    for j, product_id in enumerate(resp_list):
        ans = {"query_id": query_ids[i], "product_id": product_id, "rank": j+1}
        resp.append(ans)

# dump global json response object to submission.json
with open("submission.json", "w") as f:
    json.dump(resp, f, indent=2)

print("✅ Saved results to submission.json")