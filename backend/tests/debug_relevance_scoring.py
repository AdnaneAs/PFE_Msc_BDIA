#!/usr/bin/env python3
"""
Relevance Score Diagnostic Test
===============================

This test investigates the relevance scoring bug where the "best match" 
appears to be lower than the average, which is mathematically impossible.
"""

import requests
import json

def test_relevance_scoring():
    """Test and debug the relevance scoring calculation"""
    
    print("=" * 70)
    print("  RELEVANCE SCORING DIAGNOSTIC TEST")
    print("=" * 70)
    
    try:
        # Make a query to get relevance scores
        query_data = {
            "question": "What information is available about employees?",
            "config_for_model": {
                "max_tokens": 100,
                "model": "llama3.2:3b"
            },
            "search_strategy": "semantic",
            "max_sources": 5
        }
        
        print("🔍 Querying with semantic search...")
        response = requests.post(
            "http://localhost:8000/api/query/",
            json=query_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract relevance information
            sources = result.get('sources', [])
            average_relevance = result.get('average_relevance')
            top_relevance = result.get('top_relevance')
            
            print(f"\n📊 RELEVANCE ANALYSIS:")
            print(f"   Average Relevance: {average_relevance}")
            print(f"   Top Relevance: {top_relevance}")
            print(f"   Number of Sources: {len(sources)}")
            
            # Check individual source relevance scores
            individual_scores = []
            print(f"\n📋 INDIVIDUAL SOURCE SCORES:")
            for i, source in enumerate(sources):
                score = source.get('relevance_score')
                if score is not None:
                    individual_scores.append(score)
                    filename = source.get('filename', 'unknown')
                    chunk_idx = source.get('chunk_index', 'N/A')
                    print(f"   {i+1}. {filename}:{chunk_idx} → {score:.4f}")
                else:
                    print(f"   {i+1}. No relevance score found")
            
            # Manual calculation verification
            if individual_scores:
                manual_average = sum(individual_scores) / len(individual_scores)
                manual_max = max(individual_scores)
                manual_min = min(individual_scores)
                
                print(f"\n🧮 MANUAL VERIFICATION:")
                print(f"   Manual Average: {manual_average:.4f}")
                print(f"   Manual Maximum: {manual_max:.4f}")
                print(f"   Manual Minimum: {manual_min:.4f}")
                print(f"   Score Range: {manual_min:.4f} - {manual_max:.4f}")
                
                print(f"\n✅ CONSISTENCY CHECK:")
                avg_match = abs(manual_average - (average_relevance or 0)) < 0.001
                max_match = abs(manual_max - (top_relevance or 0)) < 0.001
                
                print(f"   Average matches: {'✅ YES' if avg_match else '❌ NO'}")
                print(f"   Maximum matches: {'✅ YES' if max_match else '❌ NO'}")
                
                # The critical check - max should be >= average
                logical_check = manual_max >= manual_average
                print(f"   Logic check (max >= avg): {'✅ YES' if logical_check else '❌ NO - BUG DETECTED!'}")
                
                if not logical_check:
                    print(f"\n🐛 BUG DETECTED!")
                    print(f"   Maximum ({manual_max:.4f}) < Average ({manual_average:.4f})")
                    print(f"   This is mathematically impossible!")
                    print(f"   Individual scores: {[round(s, 4) for s in individual_scores]}")
                
                # Check for sorting issues
                sorted_scores = sorted(individual_scores, reverse=True)
                if individual_scores != sorted_scores:
                    print(f"\n⚠️  SORTING ISSUE DETECTED:")
                    print(f"   Current order: {[round(s, 4) for s in individual_scores]}")
                    print(f"   Should be:     {[round(s, 4) for s in sorted_scores]}")
                
            else:
                print(f"\n❌ NO INDIVIDUAL SCORES FOUND")
                
        else:
            print(f"❌ Query failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")

def test_multiple_strategies():
    """Test all search strategies to compare relevance scoring"""
    
    print("\n" + "=" * 70)
    print("  MULTI-STRATEGY RELEVANCE COMPARISON")
    print("=" * 70)
    
    strategies = ["semantic", "hybrid", "keyword"]
    question = "What types of documents are available?"
    
    for strategy in strategies:
        print(f"\n🔍 Testing {strategy.upper()} strategy...")
        
        try:
            query_data = {
                "question": question,
                "config_for_model": {"max_tokens": 50, "model": "llama3.2:3b"},
                "search_strategy": strategy,
                "max_sources": 3
            }
            
            response = requests.post(
                "http://localhost:8000/api/query/",
                json=query_data,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                avg_rel = result.get('average_relevance')
                top_rel = result.get('top_relevance')
                num_sources = len(result.get('sources', []))
                
                print(f"   Sources: {num_sources}")
                print(f"   Average: {avg_rel}")
                print(f"   Best:    {top_rel}")
                print(f"   Logic:   {'✅ OK' if top_rel and avg_rel and top_rel >= avg_rel else '❌ BUG'}")
                
            else:
                print(f"   ❌ Failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def main():
    """Run all relevance scoring diagnostic tests"""
    
    print("🔬 RELEVANCE SCORING DIAGNOSTIC SUITE")
    print("Investigating the 'best match < average' bug...")
    
    test_relevance_scoring()
    test_multiple_strategies()
    
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("If bugs were detected, check the vector_db_service.py")
    print("and query.py files for relevance calculation issues.")

if __name__ == "__main__":
    main()
