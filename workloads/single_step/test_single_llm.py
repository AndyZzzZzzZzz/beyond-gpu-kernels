import requests
import time
import os

def run_llm_baseline():
    url = "http://127.0.0.1:2000/generate"
    
    # We use explicit prompts to force the 7B model to behave
    workloads = [
        ("Math_LLM_Baseline", """You are a system diagnostic agent. 
        Evaluate the math file at '../workloads/single_step/math_stress_payload.txt' using the evaluate_math_file tool. 
        Do not do anything else."""),
        
        ("DB_LLM_Baseline", """You are a system diagnostic agent. 
        Query the database at '../workloads/single_step/mock_db_payload.db' using the query_database tool. 
        Do not do anything else."""),
        
        ("FS_LLM_Baseline", """You are a system diagnostic agent. 
        Walk the directory at '../workloads/single_step/mock_fs_payload' using the walk_directory tool. 
        Do not do anything else.""")
    ]
    
    print("\nSTARTING EXPERIMENT A: LLM BASELINES")
    print("Routing tasks through Qwen 2.5 7B orchestration...\n")
    
    for test_name, prompt in workloads:
        payload = {
            "prompt": prompt,
            "max_new_tokens": 100,
            "test_name": f"{test_name}_{int(time.time())}",
            "pipeline_mode": "llm",
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
    run_llm_baseline()