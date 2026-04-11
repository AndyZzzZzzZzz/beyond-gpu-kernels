import requests
import time
import os

def run_direct_baseline():
    url = "http://127.0.0.1:2000/generate"
    
    # We use these paths exactly as they are relative to the 'src' directory
    workloads = [
        ("Math_Direct_Baseline", "Evaluate the math file at '../workloads/single_step/math_stress_payload.txt'"),
        ("DB_Direct_Baseline", "Query the database at '../workloads/single_step/mock_db_payload.db'"),
        ("FS_Direct_Baseline", "Walk the directory at '../workloads/single_step/mock_fs_payload'")
    ]
    
    print("\nSTARTING EXPERIMENT A: DIRECT BASELINES")
    print("Bypassing LLM. Executing C++ kernels directly...\n")
    
    for test_name, prompt in workloads:
        payload = {
            "prompt": prompt,
            "max_new_tokens": 10, # Doesn't matter, LLM is bypassed
            "test_name": f"{test_name}_{int(time.time())}",
            "pipeline_mode": "direct", # <--- CRITICAL: Tells server to bypass LLM
            "max_tool_steps": 1
        }
        
        try:
            print(f"Executing: {test_name}...")
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print(f"Success. Output:\n{response.json().get('response')}\n")
        except requests.exceptions.RequestException as e:
            print(f"Error on {test_name}: {e}")


if __name__ == "__main__":
    run_direct_baseline()