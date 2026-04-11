import requests
import time
import os

def run_db_vanilla():
    url = "http://127.0.0.1:2000/generate"
    
    prompt = "Query the database at '../workloads/single_step/mock_db_payload.db' to find critical errors."

    payload = {
        "prompt": prompt,
        "max_new_tokens": 100,
        "test_name": "Exp4_DB_Vanilla",
        "pipeline_mode": "llm",
        "force_tool_call": "query_database", # Ensures deterministic tool invocation
        "tool_variants": None                # Defaults to baseline kernel
    }
    
    print("🚀 Running LLM + VANILLA DB Kernel...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("✅ Vanilla DB run complete.")
        
        # Isolate results for cleaning plotting
        os.rename("../../results/cpu_profiling_log.csv", "../../results/Exp4_DB_Vanilla_Summary.csv")
        os.rename("../../results/detailed_step_log.csv", "../../results/Exp4_DB_Vanilla_Detailed.csv")
        print("📁 Data saved to: results/Exp4_DB_Vanilla_*.csv")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_db_vanilla()