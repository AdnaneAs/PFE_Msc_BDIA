import asyncio
from app.database.db_setup import check_duplicate_file, init_db, get_all_documents

async def test_duplicate():
    await init_db()
    
    # Test with a non-existent file
    result = await check_duplicate_file('test_nonexistent.pdf')
    print(f'Non-existent file check: {result}')
    
    # Test with existing files from the database
    docs = await get_all_documents()
    if docs:
        print(f'Total documents in DB: {len(docs)}')
        for doc in docs[:3]:  # Show first 3 as examples
            print(f'  - {doc["original_name"]} (status: {doc["status"]})')
        
        # Test duplicate check with first document
        if docs:
            first_doc = docs[0]
            duplicate_result = await check_duplicate_file(first_doc['original_name'])
            print(f'Duplicate check for "{first_doc["original_name"]}": Found={duplicate_result is not None}')
    else:
        print('No documents found in database')

asyncio.run(test_duplicate())
