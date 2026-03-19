## A CPU-Centric Perspective on Agentic AI

https://arxiv.org/abs/2511.00739
https://github.com/ritikraj7/cpu-centric-agentic-ai


Tool/Agent is a simple function that takes in some input and produces some output. Examples include calculator. To interact with LLMs tools/agents are wrapped in rest api and LLMs call the rest api with json payloads.

### Infrastructure

24GB RTX 4090 GPU (can run coding models upto 14b-20b such as Qwen)
64GB of DDR4 with 16 cores.

### This paper
- In representative agentic workloads, CPU tool processing can dominate end-to-end latency (up to ~90.6%), not GPU inference.
- Throughput saturation often comes from CPU-side limits (core over-subscription, cache-coherence, synchronization) or GPU-side limits (HBM capacity/bandwidth).
- At large batch sizes, CPU dynamic energy can be a big slice (reported up to ~44%).
- Two concrete scheduling ideas help: CGAM (CPU/GPU-aware micro-batching) and MAWS (mixed workload scheduling), with notable P50/P99 gains.

Thus: “the CPU is the bottleneck, now what do we do about it—at runtime, OS, and microarchitecture?”

## Tool-call microarchitecture: treat tools as first-class kernels
**Idea**: Build an Agentic Tool Kernel Suite and optimize the “boring” CPU work: JSON parsing/serialization, HTTP fetch, decompression, embedding pre/post, tokenization, vector DB lookups, file-system walks, sandbox/VM overhead.

### Research questions

- Which sub-kernels dominate per tool class (retrieval vs web vs code exec)? Does dominance shift with agent flow type (single-step vs multi-step)?
- Can ISA/microarch features (string ops, SIMD gather, prefetch hints, compression assist) move the needle more than “more cores”?

### Method

- Trace tool invocations + CPU profiles (cycles, LLC misses, syscalls, context switches).
- Build a microbenchmark suite that replays tool-kernel traces.
- Evaluate with “cycles per tool step”, tail latency, and Joules/query. Publishable angle: “Agentic tool kernels resemble datacenter RPC + analytics primitives more than ML kernels.”

---

## Submission timelines:

**Project Progress Report [60 pts]: (Due March 23 at midnight)** This should be an updated version of the project proposal that describes the progress you’ve made so far in your project. You should focus on the tasks that you have completed and not on tasks that you’ve just started or that you have planned to start on. The report should provide at least four references for papers related to your project. Your report will be graded based on how close you came to reaching 50% of your project milestones. You should submit your progress report on canvas. Project Progress Report Format: Format for the progress report is the same as the project proposal but with a five-page limit instead of two.

**Project Presentations [60 pts]: (April 10)** Each team is required to record a presentation that will be available for the whole class. Each presentation should be at most 15 minutes long. Each member in the team is required to present a part of the presentation. The presentation should higlight important findings from the project, explain the project idea and key results. All students are invited and encouraged to watch all project presentations, and ask questions on Piazza. However, the target audience for your presentation is both instructors who will grade your presentation.

**Final Project Report and Code [160 pts]: (Due April 10 at midnight)** Your report should mimic a conference paper similar to the ones we covered in class. The report should include a title, author names, abstract, introduction, background, explanation of your project, evaluation results, a conclusion, references, and optional appendices. Your report should be no longer than 10 pages using the same format as your project proposal and progress report. Any extra material beyond 10 pages should be put in appendices. Please note that any material beyond 10 pages may not be considered for grading. As part of your final project submission, you also need to submit your project code. Grading will be based on project quality and difficulty (30 points), implementation and results (60 points), report quality (50 points), documentation and code instructions (20 points) Important: Your project report Appendix should include a description of each project member’s contribution to the project code, report and presentation.
