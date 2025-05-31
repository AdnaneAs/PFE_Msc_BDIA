"""
Quick debug script to understand HotpotQA data structure
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from datasets import load_dataset

def debug_hotpot_structure():
    print("Loading HotpotQA dataset...")
    dataset = load_dataset("hotpot_qa", "distractor")
    dataset_split = dataset["validation"]
    
    print(f"Dataset size: {len(dataset_split)}")
    
    # Look at first sample
    sample = dataset_split[0]
    
    print("\n=== Sample Structure ===")
    for key, value in sample.items():
        print(f"{key}: {type(value)}")
        if key == "context":
            print(f"  Context keys: {list(value.keys()) if isinstance(value, dict) else 'Not a dict'}")
        elif key == "supporting_facts":
            print(f"  Supporting facts: {value}")
    
    print("\n=== Context Analysis ===")
    context = sample["context"]
    print(f"Context type: {type(context)}")
    print(f"Context content: {context}")
    
    if isinstance(context, dict):
        for key, value in context.items():
            print(f"\nContext key '{key}':")
            print(f"  Type: {type(value)}")
            print(f"  Length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
            if isinstance(value, list) and value:
                print(f"  First item: {value[0]}")
                if isinstance(value[0], list) and len(value[0]) > 1:
                    print(f"    Title: {value[0][0]}")
                    print(f"    Sentences: {value[0][1]}")

if __name__ == "__main__":
    debug_hotpot_structure()
