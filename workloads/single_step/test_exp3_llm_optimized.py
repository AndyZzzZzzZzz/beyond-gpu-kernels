import requests
import time
import os

def run_llm_optimized():
    url = "http://127.0.0.1:2000/generate"
    
    prompt = """
        USER TASK:
        Evaluate the math file at '../workloads/single_step/math_stress_payload.txt'. 
        You MUST return a prompt that specify to use the tool. Do not provide a text summary until the tool returns data.
        """

    payload = {
        "prompt": prompt,
        "max_new_tokens": 100,
        "test_name": "Exp3_LLM_Optimized",
        "pipeline_mode": "llm", # AI is in the loop
        "tool_variants": {
            "evaluate_math_file": "avx2" # Forces the AVX2 kernel
        }
    }
    
    print("Running LLM + OPTIMIZED (AVX2) Kernel...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Run complete.")
        
        # Rename the results for Experiment 3
        os.rename("../../results/cpu_profiling_log.csv", "../../results/Exp3_LLM_Optimized_Summary.csv")
        os.rename("../../results/detailed_step_log.csv", "../../results/Exp3_LLM_Optimized_Detailed.csv")
        print("Data saved to: results/Exp3_LLM_Optimized_*.csv")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_llm_optimized()