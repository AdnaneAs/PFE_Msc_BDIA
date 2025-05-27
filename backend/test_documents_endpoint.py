import requests
import json

print("=== Testing Documents Endpoint ===")

try:
    response = requests.get("http://localhost:8000/api/documents")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Documents endpoint working!")
        print(f"   Total documents: {len(data.get('documents', []))}")
        
        # Count by status
        documents = data.get('documents', [])
        status_counts = {}
        file_type_counts = {}
        
        for doc in documents:
            status = doc.get('status', 'unknown')
            file_type = doc.get('file_type', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
        
        print(f"   Status breakdown:")
        for status, count in status_counts.items():
            print(f"     {status}: {count}")
            
        print(f"   File type breakdown:")
        for file_type, count in file_type_counts.items():
            print(f"     {file_type}: {count}")
        
        # Show first few documents with details
        print(f"   Recent documents:")
        for i, doc in enumerate(documents[:5]):
            original_name = doc.get('original_name', doc.get('filename', 'Unknown'))
            chunk_count = doc.get('chunk_count', 0)
            print(f"     [{i+1}] {original_name}")
            print(f"         Status: {doc.get('status', 'Unknown')} | Chunks: {chunk_count} | Type: {doc.get('file_type', 'Unknown')}")
            
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()
