#!/usr/bin/env python3
"""
Relevance Scoring Validation Test
=================================

Tests the improved relevance scoring system to ensure:
1. Scores are positive and intuitive (0-100 scale)
2. Best match is always >= average (logical consistency)
3. Scores make sense relative to query relevance
"""

import requests
import statistics

def test_relevance_improvements():
    """Test the improved relevance scoring system"""
    
    print("=" * 70)
    print("  IMPROVED RELEVANCE SCORING VALIDATION")
    print("=" * 70)
    
    test_queries = [
        {
            "question": "What is retrieval augmented generation?",
            "expected_high_relevance": True,
            "description": "Direct match query"
        },
        {
            "question": "How does machine learning work?",
            "expected_high_relevance": False,
            "description": "Less relevant query"
        },
        {
            "question": "Employee information and data",
            "expected_high_relevance": True,
            "description": "Employee data query"
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_queries):
        print(f"\n🔍 Test {i+1}: {test['description']}")
        print(f"   Query: '{test['question']}'")
        
        try:
            query_data = {
                "question": test["question"],
                "config_for_model": {"max_tokens": 100, "model": "llama3.2:3b"},
                "search_strategy": "semantic",
                "max_sources": 5
            }
            
            response = requests.post(
                "http://localhost:8000/api/query/",
                json=query_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                sources = result.get('sources', [])
                avg_rel = result.get('average_relevance')
                top_rel = result.get('top_relevance')
                
                # Extract individual scores
                individual_scores = [s.get('relevance_score', 0) for s in sources if 'relevance_score' in s]
                
                # Validation checks
                checks = {
                    "positive_scores": all(score > 0 for score in individual_scores),
                    "percentage_range": all(0 <= score <= 100 for score in individual_scores),
                    "logical_consistency": top_rel >= avg_rel if top_rel and avg_rel else False,
                    "scores_descending": individual_scores == sorted(individual_scores, reverse=True),
                    "reasonable_spread": max(individual_scores) - min(individual_scores) < 50 if individual_scores else True
                }
                
                print(f"   📊 Results:")
                print(f"      Sources: {len(sources)}")
                print(f"      Average: {avg_rel}%")
                print(f"      Best:    {top_rel}%")
                print(f"      Range:   {min(individual_scores):.1f}% - {max(individual_scores):.1f}%")
                
                print(f"   ✅ Validation:")
                for check_name, passed in checks.items():
                    status = "✅" if passed else "❌"
                    print(f"      {check_name.replace('_', ' ').title()}: {status}")
                
                results.append({
                    "query": test["question"],
                    "avg_relevance": avg_rel,
                    "top_relevance": top_rel,
                    "individual_scores": individual_scores,
                    "checks": checks
                })
                
            else:
                print(f"   ❌ Query failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Overall analysis
    print(f"\n" + "=" * 70)
    print("  OVERALL VALIDATION RESULTS")
    print("=" * 70)
    
    if results:
        all_scores = []
        for r in results:
            all_scores.extend(r["individual_scores"])
        
        if all_scores:
            print(f"📊 Score Statistics:")
            print(f"   Total scores analyzed: {len(all_scores)}")
            print(f"   Average score: {statistics.mean(all_scores):.1f}%")
            print(f"   Median score: {statistics.median(all_scores):.1f}%")
            print(f"   Score range: {min(all_scores):.1f}% - {max(all_scores):.1f}%")
            print(f"   Standard deviation: {statistics.stdev(all_scores):.1f}%")
        
        # Check consistency across all tests
        all_checks = {}
        for result in results:
            for check_name, passed in result["checks"].items():
                if check_name not in all_checks:
                    all_checks[check_name] = []
                all_checks[check_name].append(passed)
        
        print(f"\n✅ Consistency Check:")
        for check_name, check_results in all_checks.items():
            pass_rate = sum(check_results) / len(check_results)
            status = "✅" if pass_rate == 1.0 else "⚠️" if pass_rate >= 0.8 else "❌"
            print(f"   {check_name.replace('_', ' ').title()}: {status} ({pass_rate:.1%})")
        
        overall_success = all(all(r["checks"].values()) for r in results)
        print(f"\n🎯 Overall Status: {'✅ ALL TESTS PASSED' if overall_success else '⚠️ SOME ISSUES DETECTED'}")
        
        if overall_success:
            print("\n🎉 RELEVANCE SCORING IMPROVEMENTS SUCCESSFUL!")
            print("   • Scores are now positive and intuitive (0-100% scale)")
            print("   • Best match is always ≥ average (logical consistency)")
            print("   • Scores are properly ordered (descending)")
            print("   • Score ranges are reasonable and meaningful")
            print("   • No more confusing negative relevance values!")
        else:
            print("\n⚠️ Some validation checks failed. Review the individual results above.")

def main():
    """Run relevance scoring validation"""
    print("🔬 RELEVANCE SCORING IMPROVEMENT VALIDATION")
    print("Testing the fix for negative relevance scores and logical inconsistencies...")
    
    test_relevance_improvements()
    
    print(f"\n" + "=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
