import requests

# Test simple CSV upload
with open("test.csv", "w") as f:
    f.write("name,age,city\nJohn,25,NYC\nJane,30,LA\n")

files = {"file": open("test.csv", "rb")}
response = requests.post("http://localhost:8000/api/documents/upload", files=files)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
