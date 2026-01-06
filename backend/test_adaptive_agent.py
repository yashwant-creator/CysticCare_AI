"""
Test script for Adaptive Agent Selector
Demonstrates how the system automatically selects the best agent
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8001"


def test_agent_selection(query: str) -> Dict[str, Any]:
    """Test the agent selection for a query"""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    response = requests.post(
        f"{BASE_URL}/analyze-query",
        json={"query": query}
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n✓ Recommended Agent: {result['recommendation'].upper()}")
        print(f"  Reason: {result['reason']}")
        print(f"\n  Configuration:")
        print(f"    - use_stepback: {result['agent_config']['use_stepback']}")
        print(f"    - use_cot: {result['agent_config']['use_cot']}")
        print(f"\n  Explanation:")
        print(f"  {result['explanation']}")
        
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}


def test_adaptive_chat(query: str) -> None:
    """Test a full chat with adaptive agent selection"""
    print(f"\n{'='*80}")
    print("ADAPTIVE CHAT TEST")
    print(f"{'='*80}")
    print(f"\nQuery: {query}")
    print("\nContacting backend with adaptive agent enabled...")
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "query": query,
            "use_adaptive_agent": True  # Enable adaptive selection
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Response received")
        print(f"  Answer length: {len(result['response'])} characters")
        print(f"  Sources used: {len(result['sources'])}")
        
        if result.get('stepback_query'):
            print(f"  Stepback query used: {result['stepback_query']}")
        
        if result.get('reasoning_chain'):
            print(f"  Reasoning steps: {len(result['reasoning_chain'])}")
        
        print(f"\n  Answer preview:")
        preview = result['response'][:200]
        print(f"  {preview}...")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def main():
    """Run comprehensive adaptive agent tests"""
    print("\n" + "="*80)
    print("ADAPTIVE AGENT SELECTOR TEST SUITE")
    print("="*80)
    
    # Check if backend is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print("\nError: Backend is not healthy")
            return
    except requests.exceptions.RequestException:
        print("\nError: Backend is not running at http://localhost:8001")
        return
    
    # Test cases for different query types
    test_cases = [
        {
            "query": "Can I take ibuprofen with tolvaptan?",
            "expected": "stepback",
            "type": "Drug Interaction"
        },
        {
            "query": "What are the side effects if I combine ibuprofen with my PKD medications?",
            "expected": "stepback",
            "type": "Drug Interaction (verbose)"
        },
        {
            "query": "How does PKD progress and what treatments can slow it down?",
            "expected": "cot",
            "type": "Multi-part Question"
        },
        {
            "query": "Compare tolvaptan versus ACE inhibitors for PKD treatment",
            "expected": "cot",
            "type": "Comparison"
        },
        {
            "query": "What is PKD?",
            "expected": "standard_rag",
            "type": "Simple Definition"
        },
        {
            "query": "What are the symptoms of ADPKD?",
            "expected": "standard_rag",
            "type": "Straightforward Question"
        },
        {
            "query": "Explain the mechanism of action of vasopressin V2-receptor antagonists",
            "expected": "stepback",
            "type": "Foundational Knowledge"
        },
        {
            "query": "Should I avoid caffeine if I have PKD? What about salt intake?",
            "expected": "stepback",
            "type": "Multi-part with Drug Focus"
        }
    ]
    
    print("\n\nTEST 1: AGENT SELECTION ANALYSIS")
    print("="*80)
    
    results = []
    for test in test_cases:
        result = test_agent_selection(test["query"])
        
        # Check if selection matches expected
        match = "✓" if result.get("recommendation") == test["expected"] else "✗"
        print(f"\n{match} Expected: {test['expected']} | Got: {result.get('recommendation')}")
        
        results.append({
            "type": test["type"],
            "expected": test["expected"],
            "actual": result.get("recommendation"),
            "match": result.get("recommendation") == test["expected"]
        })
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total = len(results)
    correct = sum(1 for r in results if r["match"])
    accuracy = correct / total * 100
    
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
    print("\nResults by category:")
    
    for test_type in set(r["type"] for r in results):
        type_results = [r for r in results if r["type"] == test_type]
        type_correct = sum(1 for r in type_results if r["match"])
        print(f"  {test_type}: {type_correct}/{len(type_results)}")
    
    # Test adaptive chat with one example
    print("\n\n" + "="*80)
    print("TEST 2: ADAPTIVE CHAT IN ACTION")
    print("="*80)
    
    test_adaptive_chat("Can I take ibuprofen with tolvaptan?")
    
    print("\n\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)
    print("\nKey Features:")
    print("  1. Automatic agent selection based on query analysis")
    print("  2. Keyword matching for drug interactions, multi-step questions, and foundational knowledge")
    print("  3. Simple, transparent logic - easy to understand and trust")
    print("  4. Seamless integration with /chat endpoint via use_adaptive_agent flag")
    print("\n")


if __name__ == "__main__":
    main()
