# Beyond the GPU: A Microarchitectural Analysis of LLM Tool-Calling Kernels

## 1 Introduction and Motivation

With the rapid rise of autonomous AI agents, the industry is increasingly deploying these systems to automate complex software engineering workflows. However, recent optimization efforts have overwhelmingly focused on the GPU forward-pass (e.g., quantization, custom tensor cores). As LLMs transition from static chatbots to active agents, the execution bottleneck shifts abruptly to the CPU.  Agentic workflows require frequent, heavy tool invocations such as parsing deeply nested JSON, managing OS sandbox context switches, and executing dense database lookups. Unlike highly parallel machine learning workloads, these tasks are fundamentally branchy, serial, and I/O-bound. Consequently, they suffer from high Last-Level Cache (LLC) miss rates, branch mispredictions, and memory bandwidth limitations, making them behave more like traditional datacenter RPC and analytics primitives than standard ML compute. Exploring these bottlenecks in agentic workflows and identifying potential optimizations could significantly improve the adoption of AI agents in software engineering tasks.

## 2 Research Questions and Goals

To expose the microarchitectural inefficiencies of agentic workflows, this research aims to characterize the CPU-side execution pipeline by addressing the following questions:
• Which sub-kernels dominate CPU execution time and hardware resources across distinct tool classes (e.g., local database retrieval, custom string-based parsing, and local file-system walks)?
• How does this microarchitectural bottleneck distribution (e.g., LLC misses, branch mispredictions) shift between isolated, single-step tool invocations and complex, multi-step agentic flows?

## 3 Proposed Methodology

### 3.1 Workload Setup
We will construct a tightly coupled agentic framework by deploying an open-weight LLM (e.g., Qwen) locally. The LLM will be equipped with a suite of local, function-level tools, specifically including a compute-bound tool (a custom string-based arithmetic expression evaluator) and a memory-bound tool (a local SQLite database lookup). This configuration allows the LLM to simulate a representative software engineering workflow: querying a local database for system data (e.g., server logs or mock telemetry), parsing the retrieved text, and computing aggregate metrics using the calculator tool. This multi-step flow ensures we capture the overhead of both isolated tool execution and inter-tool state transitions.

### 3.2 Real-Hardware Tracing and Profiling
To evaluate the microarchitectural overhead without the noise of network latency, we will utilize dynamic hardware profiling (Linux perf) directly on the host server. We will isolate the execution windows where the GPU is idle and the CPU handles the tool-call pipeline (JSON payload parsing, tool execution, and prompt reassembly). By sampling hardware performance counters during these specific windows, we will extract precise metrics on Instructions Per Cycle (IPC), Last-Level Cache (LLC) misses, and branch mispredictions.


## 4 Evaluation Plan

We will evaluate the system based on the following metrics:
• **Latency Breakdown**: The distribution of CPU cycles consumed by LLM output serialization (JSON parsing), local tool execution, and context switching.
• **Microarchitectural Efficiency**: A comparative analysis of LLC miss rates, branch mispredictions, and IPC across the different tool classes (compute-bound parsing vs. memory-bound retrieval).

## 5 Project Milestones

The following three milestones establish the core baseline required to completely characterize the workload and answer the primary research questions.
• **Milestone 1: Infrastructure and Profiling Setup**. Configure the local environment (e.g., Conda) and deploy the Qwen model via HuggingFace. Develop the core Python/C++ execution harness to load the model onto the GPU, handle inference, and precisely trigger Linux perf counters (measuring IPC, LLC misses, and branch mispredictions) specifically when the pipeline switches to the CPU.
• **Milestone 2: Single-Step Tool Characterization**. Implement the local tool suite: a string-based arithmetic evaluator (compute-bound), a local directory traversal script (I/O-bound), and a SQLite database query executor (memory-bound). Execute single-tool workflows and profile the CPU sub-kernels to quantify the microarchitectural overhead of payload serialization and isolated tool execution.
• **Milestone 3: Multi-Step Workflow Profiling**. Design and execute complex agentic prompts requiring sequential, multi-tool invocations (e.g., retrieving data via a database lookup, followed by string parsing and calculation). Measure the compounding hardware penalties of repeated context switching and inter-tool state management.

If time permits, the following reach goals will be explored to address the identified bottlenecks:
• **Milestone 4 (Reach Goal): Targeted Optimizations**. Explore the application of specific microarchitectural optimizations, such asSIMDvectorization for the string-parsingworkloads or software prefetch hints to accelerate the database retrieval loops.
• **Milestone 5 (Reach Goal): Scaling Comparison**. Evaluate how the targeted hardware optimizations from Milestone 4 compare against traditional horizontal scaling (e.g., allocating additional CPU cores or memory bandwidth) in terms of tail latency and energy efficiency.

## 6 Expected Impacts

This research will provide a first-principles microarchitectural characterization of the "agentic bottleneck," moving beyond high-level latency to quantify hardware-level inefficiencies. By the conclusion of this 3-week study, we expect to generate a "bottleneck map" correlating specific tool classes, such as memory-bound SQLite lookups versus branchheavy expression parsing, with their respective impacts on LLC miss rates and branch prediction accuracy.

The exploratory findings will reveal whether these "boring" CPU tasks are fundamentally limited by instruction throughput or memory latency, providing a critical empirical baseline for future hardware-software co-design. Ultimately, this work identifies where the next generation of AI accelerators should focus their CPU-side optimizations to enable truly seamless, autonomous software engineering agents.

### References
[1] Ritik Raj, Hong Wang, and Tushar Krishna. “A CPU-Centric Perspective on Agentic AI.” arXiv preprint arXiv:2511.00739 (2025).
[2] Gang Liao et al. “KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta.” arXiv preprint arXiv:2512.23236 (2025).
[3] Yi Zhai, Dian Shen, Junzhou Luo, and Bin Yang. “ToolCaching: Towards Efficient Caching for LLM Tool-calling.” arXiv preprint arXiv:2601.15335 (2026).
[4] Ziji Chen, Steven W. D. Chien, Peng Qian, and Noa Zilberman. “Detecting Anomalies in Machine Learning Infrastructure via Hardware Telemetry.” arXiv preprint arXiv:2510.26008 (2025).