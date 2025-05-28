import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database.db_setup import get_all_documents
from app.services.vector_db_service import get_all_vectorized_documents

async def debug_documents():
    print("=== Debugging Documents Issue ===")
    
    try:
        # Test SQLite documents
        print("1. Testing SQLite documents...")
        sqlite_docs = await get_all_documents()
        print(f"   SQLite documents type: {type(sqlite_docs)}")
        print(f"   SQLite documents count: {len(sqlite_docs)}")
        
        if sqlite_docs:
            print(f"   First document type: {type(sqlite_docs[0])}")
            print(f"   First document keys: {sqlite_docs[0].keys() if hasattr(sqlite_docs[0], 'keys') else 'No keys'}")
            print(f"   Sample document: {sqlite_docs[0]}")
        
        # Test vector documents
        print("\n2. Testing vector documents...")
        vector_docs = get_all_vectorized_documents()
        print(f"   Vector documents type: {type(vector_docs)}")
        print(f"   Vector documents count: {len(vector_docs)}")
        
        if vector_docs:
            print(f"   First vector doc type: {type(vector_docs[0])}")
            print(f"   Sample vector doc: {vector_docs[0]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_documents())
