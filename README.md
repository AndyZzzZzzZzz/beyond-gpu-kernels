# Beyond the GPU: A Microarchitectural Analysis of LLM Tool-Calling Kernels
A microarchitectural performance study of the CPU-side bottlenecks in LLM agentic workflows. This project profiles tool-calling kernels (JSON parsing, SQLite lookups, and OS tasks) to quantify Last-Level Cache (LLC) misses, branch mispredictions, and Instructions Per Cycle (IPC) during the GPU-to-CPU orchestration transition.

**Note on Pre-computed Data**: We have intentionally retained all experimental results (.csv) and generated figures (.png) in the results/ directory for immediate review. However, the full benchmarking suite can be reproduced locally using the instructions below.

## Repository Structure
-   `src/`: Contains the Python FastAPI orchestration server (model_loader.py) and connection tests.

-   `tools/`: Contains the isolated C++ execution kernels and their optimized variants (Compute, Memory, and I/O bound).

-   `workloads/`: Contains Python scripts to generate static stress payloads and the individual experiment execution scripts.

-   `results/`: Stores raw hardware telemetry logs, summarized metrics, and final visualizations.

-   `plots.py`: Visualization suite to generate paper-ready plots from the CSV results.

## Prerequisites & Hardware Requirements
To accurately reproduce the microarchitectural hardware telemetry, the following environment is required:

1.  Linux OS with perf subsystem access (required for PAPI bindings).

2.  GPU Requirements: The inference server requires sufficient VRAM to load the Qwen model. Our testbed utilizes 2x NVIDIA Tesla V100 (32GB) GPUs.

4.  C++ Build Tools: g++ compiler, libsqlite3-dev, and libpapi-dev must be installed on the host.

### Environment Setup
Create and activate the Conda environment using the provided YAML file:

```
conda env create -f environment.yml
conda activate beyond-gpu
```

## Step-by-Step Execution Guide

### Phase 1: Bootstrapping the Orchestrator
Start the FastAPI LLM orchestration server. This process will load the model weights into VRAM.

```
# Terminal 1: Spin up the model server
python src/model_loader.py
```

In a separate terminal, verify that the server is active and the model is inferencing correctly:

```
# Terminal 2: Run the health check
python src/test.py
```
(Expected Output: A successful 200 OK response containing a parsed JSON tool-call payload).

### Phase 2: Compiling the Native Kernels
You must compile both the baseline and hardware-optimized versions of the C++ tools. Navigate to the root directory and run the following commands to link PAPI and the necessary hardware flags (e.g., AVX2, OpenMP):

1. Compute-Bound (Calculator):

```
g++ -O3 -march=native tools/calculator/evaluator.cpp -o tools/calculator/eval_baseline -lpapi -static-libstdc++ -static-libgcc
# Note: Ensure the AVX2 source variant is compiled with the -mavx2 flag if separated.
```

2. Memory-Bound (Database Scanner):

```
g++ -O3 -march=native tools/db_retrieval/db_lookup_baseline.cpp -o tools/db_retrieval/db_lookup_baseline -lsqlite3 -lpapi -static-libstdc++ -static-libgcc
g++ -O3 -march=native tools/db_retrieval/db_lookup_scan_prefetch32.cpp -o tools/db_retrieval/db_lookup_scan_prefetch32 -lsqlite3 -lpapi -static-libstdc++ -static-libgcc
```

3. I/O-Bound (Filesystem Walker):
```
g++ -O3 -march=native tools/io_walker/io_walker_baseline.cpp -o tools/io_walker/io_walker_baseline -lpapi -static-libstdc++ -static-libgcc
g++ -O3 -march=native -fopenmp tools/io_walker/io_walker_omp.cpp -o tools/io_walker/io_walker_omp -lpapi -static-libstdc++ -static-libgcc
```

### Phase 3: Generating Workloads
Generate the deterministic stress payloads used for testing to ensure run-to-run consistency.

```
cd workloads/single_step
python generate_db_stress.py
python generate_fs_stress.py
python generate_math_stress.py
```
(This will populate mock_db_payload.db, the mock_fs_payload/ directory, and the math text files).

### Phase 4: Running the Experiments
Execute the test scripts to trigger the LLM agentic loops. The orchestrator will invoke the C++ kernels, which will internally capture PAPI telemetry and output to the results/ directory.

#### Single-Step Experiments:

```
# Navigate to workloads/single_step
python test_single_direct.py    # Control baseline (No LLM)
python test_single_llm.py       # Full Agentic Loop

# Experiment 3: Compute (AVX2 vs Baseline)
python test_exp3_llm_vanilla.py
python test_exp3_llm_optimized.py

# Experiment 4: Memory (Prefetch32 vs Baseline)
python test_exp4_db_vanilla.py
python test_exp4_db_optimized.py

# Experiment 5: I/O (OpenMP vs Baseline)
python test_exp5_io_vanilla.py
python test_exp5_io_optimized.py
```

#### Multi-Step Experiment:

```
cd ../multi_step
python test_multistep.py
```

### Phase 5: Plotting and Analysis
To visualize the "Agentic Tax" and the microarchitectural masking effect, run the plotting suite.

```
cd ../../  # Return to project root

```
Open plots.py, uncomment the specific experiment plotting functions you wish to generate at the bottom of the script, and run:

```
python plots.py
```
All generated .png graphs and detailed .csv telemetry logs will be saved directly into the results/ directory.