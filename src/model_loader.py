import torch                            
import uvicorn      
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
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from pypapi import events, papi_low as papi
from pypapi.exceptions import PapiNoEventError

# Initialize Linux PAPI
papi.library_init()

model = None
tokenizer = None
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EVENTS_TO_TRACK = [
    events.PAPI_TOT_CYC,  
    events.PAPI_TOT_INS,  
    events.PAPI_BR_MSP,   
    events.PAPI_L3_TCM    
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,                   
        torch_dtype=torch.float16,  
        device_map="auto",          
        attn_implementation="sdpa"  
    )
    print("Warming up the GPU...")
    dummy_input = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**dummy_input, max_new_tokens=10)
    print(f"Model successfully loaded on {model.device} and ready for queries!")
    yield   
    print("Shutting down and clearing VRAM...")
    torch.cuda.empty_cache()

# --- File Setup ---
os.makedirs("../results", exist_ok=True)
CSV_FILE = "../results/cpu_profiling_log.csv"               # Overall LLM Runs
DETAILED_CSV_FILE = "../results/detailed_step_log.csv"      # Granular LLM Steps
# PIPELINE_CSV_FILE = "../results/cpu_profiling_pipeline.csv" # Baseline Direct Runs

def _init_csv(file_path: str, headers: List[str]):
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

_init_csv(CSV_FILE, ["timestamp", "test_name", "prompt_length", "tokens", "cpu_time_ms", "cycles", "instructions", "ipc", "branch_misses", "llc_misses"])
_init_csv(DETAILED_CSV_FILE, ["timestamp", "test_name", "step_number", "phase", "tool_name", "cpu_time_ms", "cycles", "instructions", "ipc", "branch_misses", "llc_misses"])
# _init_csv(PIPELINE_CSV_FILE, ["timestamp", "test_name", "pipeline_mode", "cpu_time_ms", "tool_wall_ms", "cycles", "instructions", "ipc", "branch_misses", "llc_misses"])

app = FastAPI(title="Qwen Local Agentic Profiler", lifespan=lifespan)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.0
    test_name: str = "Unknown"
    max_tool_steps: int = 5
    pipeline_mode: str = "llm" # "llm" or "direct"
    tool_variants: Optional[Dict[str, str]] = None # Allow testing AVX2, block4k, etc.

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_math_file",
            "description": "Evaluates a massive, complex mathematical expression stored in a local text file.",
            "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "walk_directory",
            "description": "Recursively crawls a directory tree to count files.",
            "parameters": {"type": "object", "properties": {"dir_path": {"type": "string"}}, "required": ["dir_path"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Executes a memory-bound full-table scan on a SQLite database.",
            "parameters": {"type": "object", "properties": {"db_path": {"type": "string"}}, "required": ["db_path"]}
        }
    }
]

# Allows you to test different C++ architectures/optimizations in your paper
TOOL_VARIANT_BINARIES = {
    "evaluate_math_file": {"baseline": "calculator/eval_baseline", "avx2": "calculator/eval_avx2"},
    "query_database": {"baseline": "db_retrieval/db_lookup_baseline", "scan": "db_retrieval/db_lookup_scan_baseline", "prefetch32": "db_retrieval/db_lookup_scan_prefetch32"},
    "walk_directory": {"baseline": "io_walker/io_walker_baseline", "omp": "io_walker/io_walker_omp"}
}

def build_tool_plan_from_prompt(prompt: str) -> List[Dict[str, Dict[str, str]]]:
    """Extract a multi-step tool plan directly from the prompt for the 'direct' control baseline."""
    plan = []
    if match := re.search(r"([\w./-]+mock_fs_payload[\w./-]*)", prompt): plan.append({"name": "walk_directory", "args": {"dir_path": match.group(1)}})
    if match := re.search(r"([\w./-]+\.db)", prompt): plan.append({"name": "query_database", "args": {"db_path": match.group(1)}})
    if match := re.search(r"([\w./-]+\.txt)", prompt): plan.append({"name": "evaluate_math_file", "args": {"file_path": match.group(1)}})
    return plan

def run_tool(tool_name: str, args: Dict[str, str], tool_variants: Optional[Dict[str, str]]) -> Tuple[str, Optional[Dict[str, int]], float]:
    """Helper to execute the C++ binary and parse the metrics."""
    variant = tool_variants.get(tool_name, "baseline") if tool_variants else "baseline"
    
    # Resolve Path
    relative_path = TOOL_VARIANT_BINARIES.get(tool_name, {}).get(variant, f"{tool_name}_baseline")
    
    binary_path = os.path.abspath(f"../tools/{relative_path}")
    target_path = args.get("file_path") or args.get("dir_path") or args.get("db_path", "")

    if not os.path.exists(binary_path):
        return f'{{"status": "error", "error": "Missing binary {binary_path}"}}', None, 0.0

    print(f">>> [DEBUG] Running Tool: {binary_path} {target_path}")
    t0 = time.perf_counter()
    try:
        result = subprocess.run([binary_path, target_path], capture_output=True, text=True, timeout=120)
        tool_output_str = result.stdout.strip()
    except Exception as e:
        tool_output_str = f'{{"status": "error", "error": "{e}"}}'
    tool_wall_ms = (time.perf_counter() - t0) * 1000

    cpp_metrics = None
    if tool_output_str:
        try:
            parsed = json.loads(tool_output_str)
            if parsed.get("status") == "success":
                cpp_metrics = parsed.get("metrics")
        except json.JSONDecodeError:
            pass

    return tool_output_str, cpp_metrics, tool_wall_ms


@app.post("/generate")
async def generate_text(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    pipeline_mode = req.pipeline_mode.strip().lower()
    print(f"\n========== STARTING TASK: {req.test_name} | MODE: {pipeline_mode} ==========")

    # =====================================================================
    # MODE 1: DIRECT BASELINE (Control Experiment without LLM)
    # =====================================================================
    if pipeline_mode == "direct":
        request_start = time.perf_counter()
        plan = build_tool_plan_from_prompt(req.prompt)
        
        if not plan:
            raise HTTPException(status_code=400, detail="Direct mode requires paths in the prompt.")

        tot_cycles, tot_inst, tot_br, tot_l3, tot_ms = 0, 0, 0, 0, 0.0
        responses = []

        for step in plan[:req.max_tool_steps]:
            out_str, metrics, ms = run_tool(step["name"], step["args"], req.tool_variants)
            tot_ms += ms
            responses.append(out_str)
            if metrics:
                tot_cycles += metrics.get("cycles", 0)
                tot_inst += metrics.get("instructions", 0)
                tot_br += metrics.get("branch_misses", 0)
                tot_l3 += metrics.get("l3_misses", 0)

        cpu_time_ms = (time.perf_counter() - request_start) * 1000
        ipc = tot_inst / tot_cycles if tot_cycles > 0 else 0

        with open(PIPELINE_CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), req.test_name, pipeline_mode, 
                             round(cpu_time_ms, 2), round(tot_ms, 2), tot_cycles, tot_inst, round(ipc, 3), tot_br, tot_l3])

        return {"response": "\n".join(responses), "metrics_valid": True}

    # =====================================================================
    # MODE 2: FULL AGENTIC LLM LOOP (The "Agentic Tax" Benchmark)
    # =====================================================================
    request_start = time.perf_counter()

    try:
        evs = papi.create_eventset()
        papi.add_events(evs, EVENTS_TO_TRACK)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PAPI Init Error: {e}")

    current_step = 0
    messages = [{"role": "user", "content": req.prompt}]
    final_response = ""

    agg_cpu_ms, agg_cyc, agg_inst, agg_br, agg_l3 = 0, 0, 0, 0, 0

    try:
        while current_step < req.max_tool_steps:
            current_step += 1
            print(f"\n--- [AGENT STEP {current_step}] ---")

            # [CPU WINDOW 1]: Tokenization
            papi.start(evs)
            t0 = time.perf_counter()
            text = tokenizer.apply_chat_template(messages, tools=AGENT_TOOLS, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            t1 = time.perf_counter()
            pre_res = papi.stop(evs) 

            # --- GPU ---
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=req.max_new_tokens, temperature=req.temperature, do_sample=False)

            # [CPU WINDOW 2]: Serialization
            papi.start(evs) 
            t2 = time.perf_counter()
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
            response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
            t3 = time.perf_counter()
            post_res = papi.stop(evs)

            # --- Framework Metrics Logging ---
            step_ms = ((t1 - t0) + (t3 - t2)) * 1000
            s_cyc = pre_res[0] + post_res[0]
            s_inst = pre_res[1] + post_res[1]
            s_br = pre_res[2] + post_res[2]
            s_l3 = pre_res[3] + post_res[3]
            s_ipc = s_inst / s_cyc if s_cyc > 0 else 0

            with open(DETAILED_CSV_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), req.test_name, current_step, 
                                        "LLM_Framework_Overhead", "None", round(step_ms, 2), s_cyc, s_inst, round(s_ipc, 3), s_br, s_l3])

            agg_cpu_ms += step_ms; agg_cyc += s_cyc; agg_inst += s_inst; agg_br += s_br; agg_l3 += s_l3

            # Record Assistant Thought
            messages.append({"role": "assistant", "content": response_text})

            # --- Tool Routing ---
            if '{"name":' in response_text or "<tool_call>" in response_text:
                print(">>> Tool call decided by LLM. Executing...")
                tool_triggered = False
                
                try:
                    # =================================================================
                    # Safely isolate exactly ONE JSON object
                    # =================================================================
                    json_str = ""
                    
                    # Attempt 1: Look for Qwen's official <tool_call> XML tags
                    tag_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response_text, re.DOTALL)
                    if tag_match:
                        json_str = tag_match.group(1)
                    else:
                        # Attempt 2: Fallback to reading line-by-line to grab the first valid JSON
                        for line in response_text.splitlines():
                            if line.strip().startswith("{") and '"name"' in line:
                                json_str = line.strip()
                                break
                    
                    if json_str:
                        req_data = json.loads(json_str)
                        args = req_data.get("arguments", {})
                        if isinstance(args, str): args = json.loads(args)

                        out_str, metrics, _ = run_tool(req_data.get("name", ""), args, req.tool_variants)
                        
                        if metrics:
                            tool_triggered = True
                            m_cyc, m_inst = metrics.get("cycles", 0), metrics.get("instructions", 0)
                            m_br, m_l3 = metrics.get("branch_misses", 0), metrics.get("l3_misses", 0)
                            
                            with open(DETAILED_CSV_FILE, mode='a', newline='') as f:
                                csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), req.test_name, current_step, 
                                                        "Tool_Execution", req_data.get("name"), 0.0, m_cyc, m_inst, 
                                                        round(m_inst/m_cyc if m_cyc > 0 else 0, 3), m_br, m_l3])

                            agg_cyc += m_cyc; agg_inst += m_inst; agg_br += m_br; agg_l3 += m_l3

                            # Proper HuggingFace Tool Context Injection
                            messages.append({"role": "tool", "name": req_data.get("name"), "content": out_str})
                except Exception as e:
                    print(f">>> Tool execution failed: {e}")

                if not tool_triggered: break # Tool failed, abort loop to prevent infinite crash loop
            else:
                print(">>> No tool called. Final answer reached.")
                final_response = response_text
                break
    finally:
        papi.cleanup_eventset(evs)
        papi.destroy_eventset(evs)

    # --- Final Logging ---
    global_ipc = agg_inst / agg_cyc if agg_cyc > 0 else 0
    total_request_ms = (time.perf_counter() - request_start) * 1000 # <--- CALCULATE TOTAL TIME

    with open(CSV_FILE, mode='a', newline='') as f:
        csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), req.test_name, len(req.prompt), req.max_new_tokens,
                                round(total_request_ms, 2), agg_cyc, agg_inst, round(global_ipc, 3), agg_br, agg_l3])
        
    return {"response": final_response, "metrics_valid": True}

if __name__ == "__main__":
    uvicorn.run("model_loader:app", host="0.0.0.0", port=2000, reload=False)