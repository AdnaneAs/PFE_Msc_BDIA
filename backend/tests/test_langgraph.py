#!/usr/bin/env python3

print("Testing LangGraph imports...")

try:
    from langgraph.graph import StateGraph, END
    print("✅ StateGraph and END imported successfully")
    print(f"StateGraph: {StateGraph}")
    print(f"END: {END}")
except ImportError as e:
    print(f"❌ Failed to import StateGraph/END: {e}")

try:
    from langgraph.graph.message import add_messages
    print("✅ add_messages imported successfully")
    print(f"add_messages: {add_messages}")
except ImportError as e:
    print(f"❌ Failed to import add_messages: {e}")

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    print("✅ SqliteSaver imported successfully")
    print(f"SqliteSaver: {SqliteSaver}")
except ImportError as e:
    print(f"❌ Failed to import SqliteSaver: {e}")

print("Test completed!")
