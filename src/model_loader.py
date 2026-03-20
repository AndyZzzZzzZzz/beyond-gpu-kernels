import torch                            
import uvicorn      # server engine to host API
import time
import os
import csv
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

@app.post("/generate")
def generate_text(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    # Set up the PAPI EventSet (a bundle of hardware counters) for this specific request
    evs = papi.create_eventset()
    try:
        papi.add_events(evs, EVENTS_TO_TRACK)
    except PapiNoEventError:
        print("Hardware doesn't support an event. Try removing PAPI_L3_TCM from EVENTS_TO_TRACK.")
        raise HTTPException(status_code=500, detail="PAPI Event Hardware Mismatch")

    # =================================================================
    # [CPU WINDOW 1]: Start tracking Prompt Assembly & Tokenization
    # =================================================================
    papi.start(evs)             # start hardware clock
    t0 = time.perf_counter()    # start wall clock
    
    # wrap the prompt in Qwen's specific chat format template
    messages = [{"role": "user", "content": req.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Execute BPE tokenization (convert string to integer tensors)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    t1 = time.perf_counter()        # record wall clock when CPU work is done
    pre_results = papi.stop(evs)    # stop hardware clock when CPU work is done

    # --- GPU EXECUTION (Un-profiled by CPU counters) ---
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=False
        )

    # =================================================================
    # [CPU WINDOW 2]: Start tracking Serialization & Decoding
    # =================================================================
    papi.start(evs)
    t2 = time.perf_counter()
    
    # CPU Step: separate the new generated text from the original prompt
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    # CPU Step: decoding integers back into human strings and filtering special tokens
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    t3 = time.perf_counter()
    post_results = papi.stop(evs)

    # Clean up the PAPI memory for this request
    papi.cleanup_eventset(evs)
    papi.destroy_eventset(evs)

    # --- Metrics Calculation ---
    total_cycles = pre_results[0] + post_results[0]
    total_instructions = pre_results[1] + post_results[1]
    total_branch_misses = pre_results[2] + post_results[2]
    total_llc_misses = pre_results[3] + post_results[3]
    
    cpu_time_ms = ((t1 - t0) + (t3 - t2)) * 1000
    ipc = total_instructions / total_cycles if total_cycles > 0 else 0

    print("\n--- INFERENCE CPU PROFILE ---")
    print(f"Total CPU Time (ms):   {cpu_time_ms:.2f}")
    print(f"IPC:                   {ipc:.3f}")
    print(f"Branch Mispredictions: {total_branch_misses:,}")
    print(f"LLC Cache Misses:      {total_llc_misses:,}")
    print("-----------------------------\n")

    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            len(req.prompt),
            req.max_new_tokens,
            round(cpu_time_ms, 2),
            total_cycles,
            total_instructions,
            round(ipc, 3),
            total_branch_misses,
            total_llc_misses
        ])
    return {"response": response_text}

if __name__ == "__main__":
    uvicorn.run("model_loader:app", host="0.0.0.0", port=8000, reload=False)