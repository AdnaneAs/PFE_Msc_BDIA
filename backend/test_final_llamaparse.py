import requests
import time

print('=== Final LlamaParse Integration Test ===')

# Upload a different PDF to ensure fresh processing
try:
    with open('data/uploads/837f3bd9-ca85-4157-813c-29733d9dd14a.pdf', 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:8000/api/documents/upload', files=files)
        print(f'Upload Status: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            doc_id = result['doc_id']
            print(f'New Document ID: {doc_id}')
            
            # Monitor processing
            for i in range(20):
                time.sleep(1)
                try:
                    status_response = requests.get(f'http://localhost:8000/api/documents/status/{doc_id}')
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get('status', 'unknown')
                        progress = status_data.get('progress', 0)
                        message = status_data.get('message', '')
                        
                        print(f'[{i+1:2d}] {status:12} {progress:3d}% - {message}')
                        
                        if status in ['completed', 'failed']:
                            if status == 'completed':
                                print('✅ PDF processed successfully with LlamaParse!')
                                chunks = status_data.get('chunk_count', 'unknown')
                                print(f'   📊 Total chunks extracted: {chunks}')
                            break
                    else:
                        print(f'Status error: {status_response.status_code}')
                except Exception as e:
                    print(f'Status check error: {e}')
                    
        else:
            print(f'Upload failed: {response.text}')
            
except Exception as e:
    print(f'Error: {e}')
