import requests
import time

API_URL = "http://localhost:8000/generate"

def run_test(test_name, prompt, max_tokens):
    print(f"\n--- Running {test_name} ---")
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": 0.0
    }
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        print(f"Success! End-to-End Latency: {(t1-t0)*1000:.2f} ms")
        print(f"Response: {response.json()['response'][:100]}...") # Print just the start
    else:
        print(f"Error: {response.status_code} - {response.text}")

# ==========================================
# TEST 1: The Baseline (Minimal CPU Overhead)
# ==========================================
print("Sending warmup request to clear FastAPI cold-start noise...")
requests.post(API_URL, json={"prompt": "warmup", "max_new_tokens": 1, "temperature": 0.0})
time.sleep(1)

short_prompt = "Say the word 'Hi'."
run_test("Test 1: Low CPU Baseline", short_prompt, max_tokens=5)

# Give the server a second to breathe and flush caches
time.sleep(2)

# ==========================================
# TEST 2: The Stress Test (High CPU Overhead)
# ==========================================
# Generate a massive string simulating a huge server log retrieval (approx 15,000 characters)
massive_log_data = "Error 404: File not found at /var/www/html/index.php. " * 300
long_prompt = f"Summarize this log data:\n\n{massive_log_data}\n\nSummary:"

run_test("Test 2: High CPU Tokenizer Stress", long_prompt, max_tokens=50)

print("\nTests complete. Check your server console or results/cpu_profiling_log.csv!")