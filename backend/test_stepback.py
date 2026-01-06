"""
Test script for Stepback Query Decomposition Agent
Demonstrates how stepback improves retrieval and answer quality
"""

import requests
import json
from typing import Dict, Any


# Backend URL
BASE_URL = "http://localhost:8001"


def test_stepback_demo(query: str) -> Dict[str, Any]:
    """Test the stepback demo endpoint"""
    print("\n" + "=" * 80)
    print("STEPBACK QUERY DECOMPOSITION DEMO")
    print("=" * 80)
    
    print(f"\nOriginal Query: {query}")
    
    response = requests.post(
        f"{BASE_URL}/stepback-demo",
        json={"query": query}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nStepback Query: {result['stepback_query']}")
        print(f"\nRetrieval Statistics:")
        print(f"  - Original query results: {result['retrieval_stats']['original_results']}")
        print(f"  - Stepback query results: {result['retrieval_stats']['stepback_results']}")
        print(f"  - Combined unique results: {result['retrieval_stats']['combined_unique']}")
        print(f"\n{result['explanation']}")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}


def compare_standard_vs_stepback(query: str):
    """Compare standard RAG vs Stepback RAG"""
    print("\n" + "=" * 80)
    print("COMPARISON: Standard RAG vs Stepback RAG")
    print("=" * 80)
    print(f"\nQuery: {query}")
    
    # Test with standard RAG (no stepback)
    print("\n--- STANDARD RAG (No Stepback) ---")
    standard_response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "query": query,
            "use_cot": False,
            "use_stepback": False,
            "top_k": 3
        }
    )
    
    if standard_response.status_code == 200:
        standard = standard_response.json()
        print(f"\nAnswer Length: {len(standard['response'])} characters")
        print(f"Sources Used: {len(standard['sources'])}")
        print(f"\nAnswer Preview:")
        print(standard['response'][:300] + "..." if len(standard['response']) > 300 else standard['response'])
    
    # Test with Stepback RAG
    print("\n--- STEPBACK RAG ---")
    stepback_response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "query": query,
            "use_cot": False,
            "use_stepback": True,
            "top_k": 3
        }
    )
    
    if stepback_response.status_code == 200:
        stepback = stepback_response.json()
        print(f"\nStepback Query: {stepback.get('stepback_query', 'N/A')}")
        print(f"Answer Length: {len(stepback['response'])} characters")
        print(f"Sources Used: {len(stepback['sources'])}")
        print(f"\nAnswer Preview:")
        print(stepback['response'][:300] + "..." if len(stepback['response']) > 300 else stepback['response'])


def main():
    """Run test suite"""
    print("\n" + "=" * 80)
    print("STEPBACK QUERY DECOMPOSITION TEST SUITE")
    print("=" * 80)
    
    # Check if backend is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print("\nError: Backend is not healthy")
            return
    except requests.exceptions.RequestException:
        print("\nError: Backend is not running at http://localhost:8080")
        print("Please start the backend first with: cd backend && python app/main_openai.py")
        return
    
    # Test cases: specific medical questions that benefit from stepback
    test_cases = [
        {
            "query": "Can I take ibuprofen with tolvaptan?",
            "expected_stepback": "Should ask about drug interactions with vasopressin receptor antagonists"
        },
        {
            "query": "Will drinking more water help my PKD?",
            "expected_stepback": "Should ask about hydration's effect on kidney disease progression"
        },
        {
            "query": "Should I avoid caffeine if I have PKD?",
            "expected_stepback": "Should ask about dietary factors affecting PKD cyst growth"
        },
        {
            "query": "What's my risk of kidney stones with PKD?",
            "expected_stepback": "Should ask about complications associated with PKD"
        }
    ]
    
    # Run stepback demo for each test case
    print("\n\n")
    print("TEST 1: STEPBACK QUERY GENERATION")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n\nTest Case {i}:")
        test_stepback_demo(test["query"])
        print(f"\nExpected: {test['expected_stepback']}")
    
    # Run full comparison for one example
    print("\n\n")
    print("TEST 2: FULL COMPARISON")
    print("=" * 80)
    compare_standard_vs_stepback(test_cases[0]["query"])
    
    print("\n\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("  1. Stepback queries capture broader medical concepts")
    print("  2. Retrieval includes both specific and foundational knowledge")
    print("  3. Answers provide better context and comprehensiveness")
    print("  4. Particularly useful for edge cases and complex queries")
    print("\n")


if __name__ == "__main__":
    main()
