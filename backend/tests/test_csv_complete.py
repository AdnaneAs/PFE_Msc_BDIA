import requests
import time

# Test CSV upload and processing
with open("test.csv", "w") as f:
    f.write("name,age,city\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago\n")

# Upload
files = {"file": open("test.csv", "rb")}
response = requests.post("http://localhost:8000/api/documents/upload", files=files)
print(f"Upload Status: {response.status_code}")
print(f"Upload Response: {response.text}")

if response.status_code == 200:
    doc_id = response.json()["document_id"]
    
    # Wait a bit for processing
    time.sleep(3)
    
    # Check status
    status_response = requests.get(f"http://localhost:8000/api/documents/{doc_id}/status")
    print(f"Status Response: {status_response.text}")
    
    # Check documents list
    list_response = requests.get("http://localhost:8000/api/documents/")
    print(f"Documents List: {list_response.text}")
