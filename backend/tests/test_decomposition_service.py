#!/usr/bin/env python3
"""
Test the query decomposition service directly
"""

import sys
import os
import asyncio

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.query_decomposition_service import QueryDecomposer

async def test_decomposition_service():
    """Test the decomposition service directly"""
    
    # Initialize the decomposer
    decomposer = QueryDecomposer()
    
    # Test queries
    test_queries = [
        "What is the total revenue?",  # Simple query
        "What are the main financial risks mentioned in the audit report and how do they affect the company's performance?",  # Complex query
        "Can you analyze the company's cash flow, debt levels, and profitability trends over the past three years?",  # Complex query
    ]
    
    # Model config for testing
    model_config = {
        "provider": "ollama",
        "model": "llama3.2:latest"
    }
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query}")
        print('='*60)
        
        try:
            # Test decomposition
            is_complex, sub_queries = await decomposer.decompose_query(query, model_config)
            
            print(f"Is Complex: {is_complex}")
            print(f"Sub-queries ({len(sub_queries)}):")
            for j, sub_query in enumerate(sub_queries, 1):
                print(f"  {j}. {sub_query}")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Testing Query Decomposition Service")
    print("===================================")
    
    try:
        asyncio.run(test_decomposition_service())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
