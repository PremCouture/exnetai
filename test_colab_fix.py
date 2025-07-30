#!/usr/bin/env python3
"""
Test script to verify Code_to_Optimize.py execution fix for Colab
"""

def test_colab_execution():
    """Test that the pipeline executes successfully in Colab environment"""
    print("🔧 TESTING COLAB EXECUTION FIX")
    print("="*50)
    
    try:
        import sys
        import os
        sys.path.append('/home/ubuntu/exnetai')
        import Code_to_Optimize as pipeline
        print("✅ Import successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        print("🚀 Running main() function...")
        pipeline.main()
        print("✅ Pipeline executed successfully!")
        return True
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_colab_execution()
    if success:
        print("\n🎉 COLAB EXECUTION FIX SUCCESSFUL!")
    else:
        print("\n💥 COLAB EXECUTION STILL FAILING!")
