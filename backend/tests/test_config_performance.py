#!/usr/bin/env python3
"""
Quick test script to verify configuration optimizations work
"""

import sys
import os
import asyncio
import time

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

async def test_minimal_config():
    """Test the minimal configuration endpoint"""
    print("🧪 Testing minimal configuration endpoint...")
    
    try:
        from app.api.v1.config import get_minimal_system_configuration
        
        start_time = time.time()
        config = await get_minimal_system_configuration()
        end_time = time.time()
        
        duration_ms = (end_time - start_time) * 1000
        
        print(f"⚡ Minimal config loaded in {duration_ms:.1f}ms")
        print(f"📊 Config sections: {list(config.keys())}")
        
        if 'model_selection' in config and 'search_configuration' in config:
            print("✅ All required configuration sections present")
        else:
            print("❌ Missing configuration sections")
            
        return True
        
    except Exception as e:
        print(f"❌ Minimal config test failed: {e}")
        return False

async def test_fast_config():
    """Test the fast configuration endpoint"""
    print("🧪 Testing fast configuration endpoint...")
    
    try:
        from app.api.v1.config import get_fast_system_configuration
        
        start_time = time.time()
        config = await get_fast_system_configuration()
        end_time = time.time()
        
        duration_ms = (end_time - start_time) * 1000
        
        print(f"✅ Fast config loaded in {duration_ms:.1f}ms")
        print(f"📊 Config sections: {list(config.keys())}")
        
        if 'model_selection' in config and 'search_configuration' in config:
            print("✅ All required configuration sections present")
        else:
            print("❌ Missing configuration sections")
            
        return True
        
    except Exception as e:
        print(f"❌ Fast config test failed: {e}")
        return False

async def test_regular_config():
    """Test the regular configuration endpoint"""
    print("\n🧪 Testing regular configuration endpoint...")
    
    try:
        from app.api.v1.config import get_system_configuration
        
        start_time = time.time()
        config = await get_system_configuration()
        end_time = time.time()
        
        duration_ms = (end_time - start_time) * 1000
        
        print(f"✅ Regular config loaded in {duration_ms:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Regular config test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Configuration Performance Tests")
    print("=" * 40)
    
    minimal_success = await test_minimal_config()
    fast_success = await test_fast_config()
    regular_success = await test_regular_config()
    
    print("\n📋 Test Summary:")
    print(f"Minimal Config: {'✅ PASS' if minimal_success else '❌ FAIL'}")
    print(f"Fast Config: {'✅ PASS' if fast_success else '❌ FAIL'}")
    print(f"Regular Config: {'✅ PASS' if regular_success else '❌ FAIL'}")
    
    if minimal_success and fast_success and regular_success:
        print("\n🎉 All tests passed! Configuration optimizations are working.")
    else:
        print("\n⚠️  Some tests failed. Check the implementation.")

if __name__ == "__main__":
    asyncio.run(main())
