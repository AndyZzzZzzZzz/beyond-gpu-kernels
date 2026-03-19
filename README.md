# beyond-gpu-kernels
A microarchitectural performance study of the CPU-side bottlenecks in LLM agentic workflows. This project profiles tool-calling kernels (JSON parsing, SQLite lookups, and OS tasks) to quantify LLC misses, branch mispredictions, and IPC during the GPU-to-CPU transition.
