import requests
import time

API_URL = "http://localhost:2000/generate"

def run_test(test_name, prompt, max_tokens):
    print(f"\n--- Running {test_name} ---")
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": 0.0,
        "test_name": test_name
    }
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        print(f"Response: {response.json()['response'][:100]}...")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# ==========================================
# TEST 1: The Baseline (Minimal CPU Overhead)
# ==========================================
# print("Sending warmup request to clear FastAPI cold-start noise...")
# requests.post(API_URL, json={"prompt": "warmup", "max_new_tokens": 1, "temperature": 0.0})
# time.sleep(1)

# short_prompt = "Say the word 'Hi'."
# run_test("Test 1: Low CPU Baseline", short_prompt, max_tokens=5)

# # Give the server a second to breathe and flush caches
# time.sleep(2)

# ==========================================
# TEST 2: The Stress Test (High CPU Overhead)
# ==========================================
# Generate a massive string simulating a huge server log retrieval (approx 15,000 characters)
# massive_log_data = "Error 404: File not found at /var/www/html/index.php. " * 300
# long_prompt = f"Summarize this log data:\n\n{massive_log_data}\n\nSummary:"

# run_test("Test 2: High CPU Tokenizer Stress", long_prompt, max_tokens=50)

# print("\nTests complete. Check your server console or results/cpu_profiling_log.csv!")

# ==========================================
# TEST 3: Math evaluator tool call
# ==========================================
# agentic_prompt = "I have a math stress workload saved at 'workloads/single_step/math_stress_payload.txt'. Please evaluate it for me."

# run_test("Test 3: Agentic Tool Call", agentic_prompt, max_tokens=100)

# ==========================================
# TEST 4: File system I/O walk 
# ==========================================
# io_prompt = "I need to analyze an I/O bound directory. Please walk the directory at 'workloads/single_step/mock_fs_payload' and tell me what you find."

# run_test("Test 4: I/O Walker Tool Call", io_prompt, max_tokens=100)

db_prompt = "I need to analyze our server logs for issues. Please query the database located at 'workloads/single_step/mock_db_payload.db' and find the critical errors."

run_test("Test 5: DB Memory-Bound Tool Call", db_prompt, max_tokens=100)