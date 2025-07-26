#!/usr/bin/env python3
"""
Test script to verify the fixed Code_to_Optimize.py pipeline actually runs
"""

import sys
import subprocess
import time
from datetime import datetime

def test_pipeline_execution():
    """Test if the pipeline runs without parameter errors"""
    print("Testing Code_to_Optimize.py pipeline execution...")
    print("="*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, "Code_to_Optimize.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        execution_time = time.time() - start_time
        
        print(f"Execution completed in {execution_time:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Pipeline executed successfully!")
        else:
            print("❌ Pipeline execution failed!")
            
        print("\nSTDOUT:")
        print("-" * 40)
        print(result.stdout[:2000])  # First 2000 chars
        
        if result.stderr:
            print("\nSTDERR:")
            print("-" * 40)
            print(result.stderr[:1000])  # First 1000 chars
            
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print("❌ Pipeline execution timed out after 5 minutes")
        return False, "", "Timeout"
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False, "", str(e)

if __name__ == "__main__":
    success, stdout, stderr = test_pipeline_execution()
    
    if success:
        print("\n🎯 Pipeline execution test PASSED!")
    else:
        print("\n💥 Pipeline execution test FAILED!")
        
    print("="*60)
