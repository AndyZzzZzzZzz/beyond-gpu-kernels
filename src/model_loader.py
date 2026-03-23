import torch                            
import uvicorn      # server engine to host API
import time
import os
import csv
import subprocess
import json
import re
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# PAPI low-level C-bindings to talk directly to the CPU's performance counters
# Python binding for the performance application programming interface
from pypapi import events, papi_low as papi
from pypapi.exceptions import PapiNoEventError
# Initialize PAPI library globally to access the kernel's perf_event subsystem
papi.library_init()

model = None
tokenizer = None

# disable background threading in the tokenizer
# force the CPU to do all work on one thread, so PAPI can measure it accurately
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Note: PAPI_L3_TCM represents Last Level Cache (LLC) misses on most modern x86 architectures.
EVENTS_TO_TRACK = [
    events.PAPI_TOT_CYC,  # Total CPU Cycles
    events.PAPI_TOT_INS,  # Total Instructions Completed
    events.PAPI_BR_MSP,   # Branch Mispredictions
    events.PAPI_L3_TCM    # Level 3 (Last Level) Cache Misses
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything in here runs ONCE when the server starts
    It loads the model into the GPU and keeps it there until you stop the server
    """
    global model, tokenizer
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 14GB+ neural network into the GPU's VRAM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,                   
        torch_dtype=torch.float16,  # half-precision to save 50% VRAM
        device_map="auto",          # splits model across multiple GPUs
        attn_implementation="sdpa"  # optimized scaled dot product attention
    )
    
    # Warm up the model
    # prevents the initial first-run allocation lag from ruining data rows
    print("Warming up the GPU...")
    dummy_input = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**dummy_input, max_new_tokens=10)
    print(f"Model successfully loaded on {model.device} and ready for queries!")
    yield   # server stays alive and accepts requetss


    print("Shutting down and clearing VRAM...")
    torch.cuda.empty_cache()

# Create the results directory if it doesn't exist
os.makedirs("results", exist_ok=True)
CSV_FILE = "results/cpu_profiling_log.csv"

# If the file is brand new, write the column headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "prompt_length_chars", "max_new_tokens", 
            "cpu_time_ms", "total_cycles", "total_instructions", 
            "ipc", "branch_mispredictions", "llc_misses"
        ])
app = FastAPI(title="Qwen Local Agentic Profiler", lifespan=lifespan)

# define what the incoming JSON should look like
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.0
    test_name: str = "Unknown"


# The schema defining C++ tool for the LLM
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_math_file",
            "description": "Evaluates a massive, complex mathematical expression stored in a local text file. Use this when the user asks to process the math stress workload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The local path to the text file (e.g., 'workloads/single_step/math_stress_payload.txt')"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "walk_directory",
            "description": "Recursively crawls a directory tree to count files and calculate total size. Use this for I/O bound directory analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "The local path to the directory (e.g., 'workloads/single_step/mock_fs_payload')"
                    }
                },
                "required": ["dir_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Executes a memory-bound full-table scan on a massive SQLite database to count critical errors. Use this when the user asks to query or analyze the database payload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "The local path to the SQLite database file (e.g., 'workloads/single_step/mock_db_payload.db')"
                    }
                },
                "required": ["db_path"]
            }
        }
    }
]

@app.post("/generate")
def generate_text(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    evs = papi.create_eventset()
    try:
        papi.add_events(evs, EVENTS_TO_TRACK)
    except PapiNoEventError:
        raise HTTPException(status_code=500, detail="PAPI Event Hardware Mismatch")

    # =================================================================
    # [CPU WINDOW 1]: Prompt Assembly & Tokenization
    # =================================================================
    papi.start(evs)
    t0 = time.perf_counter()
    
    messages = [{"role": "user", "content": req.prompt}]
    
    # [NEW] Pass the tool schema to the tokenizer
    text = tokenizer.apply_chat_template(
        messages, 
        tools=AGENT_TOOLS, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    t1 = time.perf_counter()
    pre_results = papi.stop(evs)

    # --- GPU EXECUTION 1 ---
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=False
        )

    # =================================================================
    # [CPU WINDOW 2]: Serialization & Tool Interception
    # =================================================================
    papi.start(evs)
    t2 = time.perf_counter()
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    
    # IMPORTANT: skip_special_tokens=False so we can see Qwen's <tool_call> tags
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    t3 = time.perf_counter()
    post_results = papi.stop(evs) # Stop Python PAPI before running C++

    # --- Metrics Accumulators ---
    total_cycles = pre_results[0] + post_results[0]
    total_instructions = pre_results[1] + post_results[1]
    total_branch_misses = pre_results[2] + post_results[2]
    total_llc_misses = pre_results[3] + post_results[3]
    
    # [NEW] Agentic Interception Logic
    tool_output_str = ""
    cpp_metrics = None

    if '{"name":' in response_text or "<tool_call>" in response_text:
        print("\n>>> [AGENT TRIGGERED] Tool call detected! Parsing request...")
        
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                tool_req = json.loads(json_match.group(0))
                tool_name = tool_req.get("name", "")
                
                # Extract arguments based on different possible LLM output structures
                args = tool_req.get("arguments", {})
                if isinstance(args, str):
                     args = json.loads(args)
                
                binary_path = ""
                target_path = ""
                
                # --- Routing Logic ---
                if "evaluate_math_file" in tool_name or "evaluate_math_file" in response_text:
                    binary_path = "./tools/calculator/eval_baseline"
                    target_path = args.get("file_path", "")
                    print(f">>> [DEBUG] Routing to MATH TOOL with path: '{target_path}'")
                    
                elif "walk_directory" in tool_name or "walk_directory" in response_text:
                    binary_path = "./tools/io_walker/io_walker_baseline"
                    target_path = args.get("dir_path", "")
                    print(f">>> [DEBUG] Routing to I/O WALKER with path: '{target_path}'")
                elif "query_database" in tool_name or "query_database" in response_text:
                    binary_path = "./tools/db_retrieval/db_lookup_baseline"
                    target_path = args.get("db_path", "")
                    print(f">>> [DEBUG] Routing to DATABASE TOOL with path: '{target_path}'")
                # --- Execution ---
                if binary_path and target_path:
                    result = subprocess.run(
                        [binary_path, target_path], 
                        capture_output=True, 
                        text=True
                    )
                    
                    if result.stderr:
                        print(f">>> [C++ STDERR ERROR]: {result.stderr.strip()}")
                        
                    tool_output_str = result.stdout.strip()
                    
                    if not tool_output_str:
                        print(">>> [TOOL ERROR] C++ stdout was completely empty.")
                    else:
                        try:
                            cpp_data = json.loads(tool_output_str)
                            if cpp_data.get("status") == "success":
                                cpp_metrics = cpp_data.get("metrics", {})
                                print(f">>> [TOOL SUCCESS] Result: {cpp_data.get('result')}")
                        except json.JSONDecodeError:
                            print(f">>> [TOOL ERROR] Failed to parse C++ output: {tool_output_str}")

        except Exception as e:
            print(f">>> [TOOL EXECUTION FAILED]: {e}")

    papi.cleanup_eventset(evs)
    papi.destroy_eventset(evs)

    # --- Merge Python and C++ Metrics ---
    cpu_time_ms = ((t1 - t0) + (t3 - t2)) * 1000
    
    if cpp_metrics:
        total_cycles += cpp_metrics.get("cycles", 0)
        total_instructions += cpp_metrics.get("instructions", 0)
        total_branch_misses += cpp_metrics.get("branch_misses", 0)
        total_llc_misses += cpp_metrics.get("l3_misses", 0)

    ipc = total_instructions / total_cycles if total_cycles > 0 else 0

    print("\n--- COMBINED AGENTIC CPU PROFILE ---")
    print(f"Total CPU Time (Python ms): {cpu_time_ms:.2f}")
    print(f"Total IPC:                  {ipc:.3f}")
    print(f"Total Branch Misses:        {total_branch_misses:,}")
    print(f"Total LLC Cache Misses:     {total_llc_misses:,}")
    print("------------------------------------\n")

    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            req.test_name,
            len(req.prompt),
            req.max_new_tokens,
            round(cpu_time_ms, 2),
            total_cycles,
            total_instructions,
            round(ipc, 3),
            total_branch_misses,
            total_llc_misses
        ])
        
    final_response = response_text if not tool_output_str else f"Tool executed successfully.\nRaw output:\n{tool_output_str}"
    
    return {"response": final_response}

if __name__ == "__main__":
    uvicorn.run("model_loader:app", host="0.0.0.0", port=2000, reload=False)