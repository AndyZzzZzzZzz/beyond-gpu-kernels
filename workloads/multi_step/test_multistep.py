# test_multistep.py
import requests
import time

def run_multistep_test():
    url = "http://127.0.0.1:2000/generate"
    test_name = f"MultiStep_Gauntlet_{int(time.time())}"
    
    prompt2 = prompt = """
    You are a system diagnostic agent. You have access to three tools: walk_directory, query_database, and evaluate_math_file.

    Your task is to run a full diagnostic by executing all three tools sequentially. 
    
    CRITICAL RULES:
    1. You must ONLY use the exact file paths provided below. Do not make up paths.
       - Directory Path: "../workloads/single_step/mock_fs_payload"
       - Database Path: "../workloads/single_step/mock_db_payload.db"
       - Math File Path: "../workloads/single_step/math_stress_payload.txt"
    2. TAKE ONE STEP AT A TIME. Output exactly ONE JSON tool call per response. Do NOT output multiple JSON blocks.
    3. Wait for the 'System Output' to be provided to you before calling the next tool.
    4. Once you have received the results from all three tools, provide a final text summary.

    Begin by calling the first tool.
    """
    
    prompt = "say hi"

    payload = {
        "prompt": prompt2.strip(),
        "max_new_tokens": 300,
        "temperature": 0.0,
        "test_name": test_name
    }
    
    print(f"\n🚀 Sending Request: {test_name}")
    print("⏳ Waiting for Agent to complete the multi-step loop...")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("\n✅ Final Agent Response:\n")
        print(response.json().get("response"))
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error connecting to server: {e}")
        if response is not None:
            print(f"Server specifically said: {response.text}")

if __name__ == "__main__":
    run_multistep_test()