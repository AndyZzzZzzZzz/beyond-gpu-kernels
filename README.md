# Beyond the GPU: A Microarchitectural Analysis of LLM Tool-Calling Kernels
A microarchitectural performance study of the CPU-side bottlenecks in LLM agentic workflows. This project profiles tool-calling kernels (JSON parsing, SQLite lookups, and OS tasks) to quantify Last-Level Cache (LLC) misses, branch mispredictions, and Instructions Per Cycle (IPC) during the GPU-to-CPU orchestration transition.

**Note on Pre-computed Data**: We have intentionally retained all experimental results (.csv) and generated figures (.png) in the results/ directory for immediate review. However, the full benchmarking suite can be reproduced locally using the instructions below.

## Repository Structure
-   `src/`: Contains the Python FastAPI orchestration server (`model_loader.py`) and the connection health check (`test.py`).

-   `tools/`: Contains the isolated C++ execution kernels and their optimized variants:
    - `calculator/`: Compute-bound math expression evaluator (Baseline + AVX2)
    - `db_retrieval/`: Memory-bound SQLite full-table scanner (Baseline + Prefetch32)
    - `io_walker/`: I/O-bound filesystem walker (Baseline + OpenMP)

-   `workloads/`: Contains Python scripts to generate static stress payloads and the individual experiment execution scripts.
    - `single_step/`: Generators and test scripts for isolated single-tool experiments.
    - `multi_step/`: Test script for the sequential multi-tool agentic flow.

-   `results/`: Stores raw hardware telemetry logs (`.csv`), summarized metrics, and final visualizations (`.png`).

-   `plots.py`: Visualization suite to generate paper-ready plots from the CSV results.

-   `environment.yml`: Conda environment specification for Python dependencies.

-   `project-spec.md`: Project proposal and research methodology document.

## Prerequisites & Hardware Requirements
To accurately reproduce the microarchitectural hardware telemetry, the following environment is required:

1.  **Linux OS** with `perf` subsystem access (required for PAPI hardware counter bindings).

2.  **GPU Requirements**: The inference server requires sufficient VRAM to load the Qwen 2.5 7B model. Our testbed utilizes 2x NVIDIA Tesla V100 (32GB) GPUs.

3.  **C++ Build Tools**: `g++` compiler with C++17 support, `libsqlite3-dev`, and `libpapi-dev` must be installed on the host.

4.  **Kernel Permissions**: PAPI requires access to hardware performance counters. If you encounter permission errors, enable perf access:
    ```
    sudo sysctl -w kernel.perf_event_paranoid=-1
    ```
    To make this persistent across reboots, add `kernel.perf_event_paranoid = -1` to `/etc/sysctl.conf`.

### Environment Setup
Create and activate the Conda environment using the provided YAML file:

```
conda env create -f environment.yml
conda activate beyond-gpu
```

Then install the required Python packages via pip:

```
pip install torch transformers fastapi uvicorn requests pypapi matplotlib seaborn pandas numpy accelerate
```

> **Note**: The `environment.yml` file captures the base Conda dependencies. The `pip install` step above is required for the ML framework, web server, hardware profiling, and visualization packages that are not available through Conda channels.

## Step-by-Step Execution Guide

> **Important**: All commands in this guide assume you are working from the **project root directory** (`beyond-gpu-kernels/`) unless explicitly stated otherwise.

### Phase 1: Bootstrapping the Orchestrator
Start the FastAPI LLM orchestration server. This process will load the model weights into VRAM.

```
# Terminal 1: Spin up the model server (must run from src/)
cd src
python model_loader.py
```

The server will start on `http://0.0.0.0:2000`. Wait for the message `"Model successfully loaded on cuda:0 and ready for queries!"` before proceeding.

In a **separate terminal**, verify that the server is active and the model is inferencing correctly:

```
# Terminal 2: Run the health check (from project root)
cd src
python test.py
```
(Expected Output: A successful `200 OK` response containing a parsed JSON tool-call payload).

> **Note**: Terminal 1 must remain running for all subsequent phases. All experiment and workload commands in Phases 3–5 must be executed from **Terminal 2** (or any other separate terminal).

### Phase 2: Compiling the Native Kernels
You must compile both the baseline and hardware-optimized versions of the C++ tools. Run the following commands from the **project root directory** to link PAPI and the necessary hardware flags (e.g., AVX2, OpenMP):

1. **Compute-Bound (Calculator)**:

```
# Baseline kernel
g++ -O3 -march=native tools/calculator/evaluator.cpp -o tools/calculator/eval_baseline -lpapi -static-libstdc++ -static-libgcc

# AVX2-optimized kernel (used in Experiment 3)
g++ -O3 -march=native -mavx2 tools/calculator/evaluator.cpp -o tools/calculator/eval_avx2 -lpapi -static-libstdc++ -static-libgcc
```

2. **Memory-Bound (Database Scanner)**:

```
# Baseline kernel
g++ -O3 -march=native tools/db_retrieval/db_lookup_baseline.cpp -o tools/db_retrieval/db_lookup_baseline -lsqlite3 -lpapi -static-libstdc++ -static-libgcc

# Prefetch32-optimized kernel (used in Experiment 4)
g++ -O3 -march=native tools/db_retrieval/db_lookup_scan_prefetch32.cpp -o tools/db_retrieval/db_lookup_scan_prefetch32 -lsqlite3 -lpapi -static-libstdc++ -static-libgcc
```

3. **I/O-Bound (Filesystem Walker)**:
```
# Baseline kernel
g++ -O3 -march=native tools/io_walker/io_walker_baseline.cpp -o tools/io_walker/io_walker_baseline -lpapi -static-libstdc++ -static-libgcc

# OpenMP-optimized kernel (used in Experiment 5)
g++ -O3 -march=native -fopenmp tools/io_walker/io_walker_omp.cpp -o tools/io_walker/io_walker_omp -lpapi -static-libstdc++ -static-libgcc
```

### Phase 3: Generating Workloads
Generate the deterministic stress payloads used for testing to ensure run-to-run consistency. Run these commands from the **project root**:

```
python workloads/single_step/generate_db_stress.py
python workloads/single_step/generate_fs_stress.py
python workloads/single_step/generate_math_stress.py
```

This will populate:
- `workloads/single_step/mock_db_payload.db` (~400–500 MB SQLite database with 5M rows)
- `workloads/single_step/mock_fs_payload/` (deeply nested directory tree with ~14K files)
- `workloads/single_step/math_stress_payload.txt` (~1.4 MB of nested arithmetic expressions)

> **Note**: The database generation may take several minutes depending on disk speed.

### Phase 4: Running the Experiments
Execute the test scripts to trigger the LLM agentic loops. The orchestrator will invoke the C++ kernels, which will internally capture PAPI telemetry and output to the `results/` directory.

> **Important**: Ensure the model server from Phase 1 is still running in Terminal 1. All experiment scripts below must be run from `workloads/single_step/` (or `workloads/multi_step/` for the multi-step experiment) in a separate terminal.

#### Experiment A — Single-Step Baselines (The "Agentic Tax"):

```
cd workloads/single_step

# Control baseline: Direct C++ execution (No LLM in the loop)
python test_single_direct.py

# Full Agentic Loop: LLM orchestrates the same tool calls
python test_single_llm.py
```

#### Experiment 3 — Compute Optimization (AVX2 vs. Baseline):

```
# From workloads/single_step/
python test_exp3_llm_vanilla.py
python test_exp3_llm_optimized.py
```

#### Experiment 4 — Memory Optimization (Prefetch32 vs. Baseline):

```
# From workloads/single_step/
python test_exp4_db_vanilla.py
python test_exp4_db_optimized.py
```

#### Experiment 5 — I/O Optimization (OpenMP vs. Baseline):

```
# From workloads/single_step/
python test_exp5_io_vanilla.py
python test_exp5_io_optimized.py
```

#### Experiment B — Multi-Step Agentic Flow:

```
cd ../multi_step
python test_multistep.py
```

### Phase 5: Plotting and Analysis
To visualize the "Agentic Tax" and the microarchitectural masking effect, run the plotting suite from the **project root**:

```
cd ../../  # Return to project root (if coming from workloads/multi_step)
```

Open `plots.py` and uncomment the desired plotting functions at the bottom of the script. The available functions are:

| Function | Experiment | Description |
|----------|------------|-------------|
| `generate_expA_plot()` | Exp A | Agentic Tax: Direct vs. LLM latency comparison |
| `generate_expB_plot()` | Exp B | Multi-step instruction volume and IPC per ReAct step |
| `generate_masking_plot()` | Exp 3 | Compute masking effect: AVX2 kernel speedup vs. system speedup |
| `generate_db_masking_analysis()` | Exp 4 | Memory wall: LLC misses and IPC with prefetching |
| `generate_io_specific_plot()` | Exp 5 | I/O latency and threading efficiency with OpenMP |
| `generate_agentic_tax_viz()` | All | Final consolidated research comparison across all tool classes |

Then run:
```
python plots.py
```
All generated `.png` graphs and detailed `.csv` telemetry logs will be saved directly into the `results/` directory.