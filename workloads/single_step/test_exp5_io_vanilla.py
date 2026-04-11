import requests
import os

def run_io_vanilla():
    url = "http://127.0.0.1:2000/generate"
    prompt = "Crawl the directory at '../workloads' to count the files."

    payload = {
        "prompt": prompt,
        "max_new_tokens": 100,
        "test_name": "Exp5_IO_Vanilla",
        "pipeline_mode": "llm",
        "force_tool_call": "walk_directory", 
        "tool_variants": None # Defaults to baseline
    }
    
    print("🚀 Running LLM + VANILLA IO Kernel...")
    response = requests.post(url, json=payload)
    print("✅ Run complete.")
    
    os.rename("../../results/cpu_profiling_log.csv", "../../results/Exp5_IO_Vanilla_Summary.csv")
    os.rename("../../results/detailed_step_log.csv", "../../results/Exp5_IO_Vanilla_Detailed.csv")

if __name__ == "__main__":
    run_io_vanilla()