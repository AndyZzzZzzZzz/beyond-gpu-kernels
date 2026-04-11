import requests
import time
import os

def run_db_optimized():
    url = "http://127.0.0.1:2000/generate"
    
    prompt = "Query the database at '../workloads/single_step/mock_db_payload.db' to find critical errors."

    payload = {
        "prompt": prompt,
        "max_new_tokens": 100,
        "test_name": "Exp4_DB_Optimized",
        "pipeline_mode": "llm",
        "force_tool_call": "query_database", 
        "tool_variants": {
            "query_database": "prefetch32" # Forces the prefetch-optimized kernel
        }
    }
    
    print("🚀 Running LLM + OPTIMIZED (Prefetch32) DB Kernel...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("✅ Optimized DB run complete.")
        
        # Isolate results
        os.rename("../../results/cpu_profiling_log.csv", "../../results/Exp4_DB_Optimized_Summary.csv")
        os.rename("../../results/detailed_step_log.csv", "../../results/Exp4_DB_Optimized_Detailed.csv")
        print("📁 Data saved to: results/Exp4_DB_Optimized_*.csv")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_db_optimized()