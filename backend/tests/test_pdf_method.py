import requests
import time
import json

print('=== Testing PDF Processing Method ===')

# Upload a PDF
try:
    with open('data/uploads/32efb456-8097-466e-8946-3a17c7a07840.pdf', 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:8000/api/documents/upload', files=files)
        print(f'Upload Status: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            doc_id = result['doc_id']
            print(f'Document ID: {doc_id}')
            
            # Monitor processing
            completed = False
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
                        
                        # Look for clues about which method was used
                        if 'fallback' in message.lower():
                            print('🔄 USING FALLBACK METHOD')
                        elif 'llamaparse' in message.lower():
                            print('🚀 USING LLAMAPARSE')
                        
                        if status in ['completed', 'failed']:
                            completed = True
                            if status == 'completed':
                                print('✅ PDF processed successfully!')
                                chunks = status_data.get('chunk_count', 'unknown')
                                print(f'   Chunks: {chunks}')
                            else:
                                print('❌ PDF processing failed!')
                            break
                    else:
                        print(f'Status error: {status_response.status_code}')
                except Exception as e:
                    print(f'Status check error: {e}')
            
            if not completed:
                print('⚠️  Processing timeout')
                
        else:
            print(f'Upload failed: {response.text}')
            
except Exception as e:
    print(f'Error: {e}')
