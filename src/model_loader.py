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
# Create the results directory if it doesn't exist
os.makedirs("results", exist_ok=True)
CSV_FILE = "results/cpu_profiling_log.csv"
DETAILED_CSV_FILE = "results/detailed_step_log.csv"

# Initialize global summary log headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "test_name", "prompt_length_chars", "max_new_tokens", 
            "cpu_time_ms", "total_cycles", "total_instructions", 
            "ipc", "branch_mispredictions", "llc_misses"
        ])

# Initialize detailed step-by-step log headers
if not os.path.exists(DETAILED_CSV_FILE):
    with open(DETAILED_CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "test_name", "step_number", "phase", "tool_name",
            "cpu_time_ms", "cycles", "instructions", "ipc", 
            "branch_misses", "llc_misses"
        ])

app = FastAPI(title="Qwen Local Agentic Profiler", lifespan=lifespan)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.0
    test_name: str = "Unknown"

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_math_file",
            "description": "Evaluates a massive, complex mathematical expression stored in a local text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "walk_directory",
            "description": "Recursively crawls a directory tree to count files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string"}
                },
                "required": ["dir_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Executes a memory-bound full-table scan on a SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_path": {"type": "string"}
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

    # --- AGENTIC LOOP STATE ---
    MAX_STEPS = 5  
    current_step = 0
    messages = [{"role": "user", "content": req.prompt}]
    final_response = ""

    # --- GLOBAL METRIC ACCUMULATORS ---
    agg_cpu_time_ms = 0
    agg_cycles = 0
    agg_instructions = 0
    agg_branch_misses = 0
    agg_llc_misses = 0

    print(f"\n========== STARTING AGENTIC TASK: {req.test_name} ==========")

    while current_step < MAX_STEPS:
        current_step += 1
        print(f"\n--- [AGENT STEP {current_step}] ---")

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
        
        text = tokenizer.apply_chat_template(
            messages, 
            tools=AGENT_TOOLS, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        t1 = time.perf_counter()
        pre_results = papi.stop(evs)

        # --- GPU EXECUTION ---
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                do_sample=False
            )

        # =================================================================
        # [CPU WINDOW 2]: Serialization & Decoding
        # =================================================================
        papi.start(evs)
        t2 = time.perf_counter()
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        t3 = time.perf_counter()
        post_results = papi.stop(evs)
        papi.cleanup_eventset(evs)
        papi.destroy_eventset(evs)

        # --- STEP-SPECIFIC FRAMEWORK METRICS ---
        step_cpu_time_ms = ((t1 - t0) + (t3 - t2)) * 1000
        step_cycles = pre_results[0] + post_results[0]
        step_instructions = pre_results[1] + post_results[1]
        step_branch_misses = pre_results[2] + post_results[2]
        step_llc_misses = pre_results[3] + post_results[3]
        step_ipc = step_instructions / step_cycles if step_cycles > 0 else 0

        # Log Framework Overhead
        with open(DETAILED_CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                req.test_name, current_step, "LLM_Framework_Overhead", "None",
                round(step_cpu_time_ms, 2), step_cycles, step_instructions, 
                round(step_ipc, 3), step_branch_misses, step_llc_misses
            ])

        # Accumulate Framework Metrics
        agg_cpu_time_ms += step_cpu_time_ms
        agg_cycles += step_cycles
        agg_instructions += step_instructions
        agg_branch_misses += step_branch_misses
        agg_llc_misses += step_llc_misses

        # Remember what the LLM just said
        messages.append({"role": "assistant", "content": response_text})

        # =================================================================
        # [NEW]: Tool Interception & Feedback Routing
        # =================================================================
        tool_triggered = False

        if '{"name":' in response_text or "<tool_call>" in response_text:
            print(">>> Tool call decided by LLM. Executing...")
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    tool_req = json.loads(json_match.group(0))
                    tool_name = tool_req.get("name", "")
                    args = tool_req.get("arguments", {})
                    if isinstance(args, str):
                         args = json.loads(args)
                    
                    binary_path, target_path = "", ""
                    
                    if "evaluate_math_file" in tool_name or "evaluate_math_file" in response_text:
                        binary_path = "./tools/calculator/eval_baseline"
                        target_path = args.get("file_path", "")
                    elif "walk_directory" in tool_name or "walk_directory" in response_text:
                        binary_path = "./tools/io_walker/io_walker_baseline"
                        target_path = args.get("dir_path", "")
                    elif "query_database" in tool_name or "query_database" in response_text:
                        binary_path = "./tools/db_retrieval/db_lookup_baseline"
                        target_path = args.get("db_path", "")
                    
                    if binary_path and target_path:
                        result = subprocess.run([binary_path, target_path], capture_output=True, text=True)
                        tool_output_str = result.stdout.strip()
                        
                        if tool_output_str:
                            try:
                                cpp_data = json.loads(tool_output_str)
                                if cpp_data.get("status") == "success":
                                    tool_triggered = True
                                    print(f">>> Tool Success: {cpp_data.get('result')}")
                                    
                                    # --- STEP-SPECIFIC C++ METRICS ---
                                    cpp_metrics = cpp_data.get("metrics", {})
                                    cpp_cycles = cpp_metrics.get("cycles", 0)
                                    cpp_instructions = cpp_metrics.get("instructions", 0)
                                    cpp_ipc = cpp_instructions / cpp_cycles if cpp_cycles > 0 else 0
                                    
                                    # Log Tool Execution
                                    with open(DETAILED_CSV_FILE, mode='a', newline='') as f:
                                        writer = csv.writer(f)
                                        writer.writerow([
                                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            req.test_name, current_step, "Tool_Execution", tool_name,
                                            0.0, # No python time for C++
                                            cpp_cycles, cpp_instructions, round(cpp_ipc, 3), 
                                            cpp_metrics.get("branch_misses", 0), 
                                            cpp_metrics.get("l3_misses", 0)
                                        ])

                                    # Accumulate C++ Metrics
                                    agg_cycles += cpp_cycles
                                    agg_instructions += cpp_instructions
                                    agg_branch_misses += cpp_metrics.get("branch_misses", 0)
                                    agg_llc_misses += cpp_metrics.get("l3_misses", 0)

                                    # FEEDBACK LOOP: Inject the result back into the prompt!
                                    messages.append({
                                        "role": "user", 
                                        "content": f"System Output:\n{tool_output_str}\nProceed with the next step or provide your final answer."
                                    })
                                    
                            except json.JSONDecodeError:
                                print(">>> Failed to parse C++ output.")
            except Exception as e:
                print(f">>> Tool execution failed: {e}")

        if not tool_triggered:
            print(">>> No tool called. Final answer reached.")
            final_response = response_text
            break

    # =================================================================
    # Log Final Aggregated Data
    # =================================================================
    global_ipc = agg_instructions / agg_cycles if agg_cycles > 0 else 0

    print("\n========== MULTI-STEP PROFILE COMPLETE ==========")
    print(f"Total Steps Taken:          {current_step}")
    print(f"Total CPU Time (Python ms): {agg_cpu_time_ms:.2f}")
    print(f"Total IPC:                  {global_ipc:.3f}")
    print(f"Total Branch Misses:        {agg_branch_misses:,}")
    print(f"Total LLC Cache Misses:     {agg_llc_misses:,}")
    print("=================================================\n")

    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            req.test_name,
            len(req.prompt),
            req.max_new_tokens,
            round(agg_cpu_time_ms, 2),
            agg_cycles,
            agg_instructions,
            round(global_ipc, 3),
            agg_branch_misses,
            agg_llc_misses
        ])
        
    return {"response": final_response}

if __name__ == "__main__":
    uvicorn.run("model_loader:app", host="0.0.0.0", port=2000, reload=False)