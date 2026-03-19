import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables to keep the model and tokenizer resident in VRAM
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This runs once when the server starts. It handles downloading (if necessary)
    and loading the model into GPU memory.
    """
    global model, tokenizer
    
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading {model_id}...")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 2. Load Model 
    # device_map="auto" places it on the GPU.
    # attn_implementation="sdpa" uses PyTorch's native Flash Attention alternative!
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa" 
    )
    
    # Warm up the model (optional, but good for profiling consistency)
    print("Warming up the GPU...")
    dummy_input = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**dummy_input, max_new_tokens=10)
        
    print(f"Model successfully loaded on {model.device} and ready for queries!")
    
    yield # The server runs and accepts requests while yielded
    
    # Cleanup (runs when server shuts down)
    print("Shutting down and clearing VRAM...")
    model = None
    tokenizer = None
    torch.cuda.empty_cache()

# Initialize the API
app = FastAPI(title="Qwen Local Agentic Harness", lifespan=lifespan)

# Define the expected JSON payload format
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.0 # Keep at 0.0 for deterministic profiling

@app.post("/generate")
def generate_text(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    # -----------------------------------------------------------------
    # [CPU WINDOW START]: JSON Payload Parsing & Prompt Assembly
    # -----------------------------------------------------------------
    
    # Wrap the raw prompt in Qwen's specific instruction chat template
    messages = [{"role": "user", "content": req.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize and move to GPU
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # -----------------------------------------------------------------
    # RESEARCH HOOK: STOP CPU PERF COUNTERS HERE
    # (GPU takes over for the heavy compute phase)
    # -----------------------------------------------------------------

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=False if req.temperature == 0.0 else True
        )

    # -----------------------------------------------------------------
    # RESEARCH HOOK: START CPU PERF COUNTERS HERE
    # (GPU is idle, CPU handles serialization and returning to the agent flow)
    # -----------------------------------------------------------------
    
    # Slice the output to remove the input tokens (we only want the new response)
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    
    # Decode back to a string
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # -----------------------------------------------------------------
    # [CPU WINDOW END]: Return payload to the client/tool orchestrator
    # -----------------------------------------------------------------
    
    return {"response": response_text}

if __name__ == "__main__":
    # Runs the server locally on port 8000
    uvicorn.run("model_loader:app", host="0.0.0.0", port=8000, reload=False)