"""
Debug script to understand HotpotQA dataset structure
"""
from datasets import load_dataset
import json

# Load dataset
dataset = load_dataset("hotpot_qa", "distractor")
sample = dataset['validation'][0]

print("HotpotQA Sample Structure:")
print("=" * 50)
for key, value in sample.items():
    print(f"{key}: {type(value)}")
    if key == 'context':
        print(f"  Context structure: {type(value)}")
        if isinstance(value, dict):
            print(f"  Context keys: {list(value.keys())}")
            for ctx_key in value.keys():
                print(f"    {ctx_key}: {type(value[ctx_key])}")
                if isinstance(value[ctx_key], list) and len(value[ctx_key]) > 0:
                    print(f"      First item: {value[ctx_key][0]}")
                    break  # Just show first one
        else:
            print(f"  Context: {value}")
    elif key == 'supporting_facts':
        print(f"  Supporting facts: {value}")
    else:
        print(f"  Value: {value}")
