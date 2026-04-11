# Presentation Script — *Beyond the GPU: A Microarchitectural Analysis of LLM Tool-Calling Kernels*

**Total target length:** ~15 minutes
**Speakers:** TJ Grewal → Kunpeng Zhang → Fazal Rehman

Speaker assignments follow the appendix in the report: TJ covers motivation / background / architecture; Kunpeng covers methodology, kernels, and optimizations; Fazal covers results, limitations, and conclusion.

Rough pacing (21 slides):
- TJ (Slides 1–6) — ~4.5 min
- Kunpeng (Slides 7–12) — ~5 min
- Fazal (Slides 13–21) — ~5.5 min

---

## Part I — TJ Grewal (Motivation, Background, Architecture Overview)

### Slide 1 — Title
> Good afternoon everyone. Our project is titled *"Beyond the GPU: A Microarchitectural Analysis of LLM Tool-Calling Kernels."* I'm TJ Grewal, and I'll be presenting with my teammates Kunpeng Zhang and Fazal Rehman. The core question we set out to answer is deceptively simple: **what happens when the CPU, not the GPU, becomes the bottleneck for AI?**

### Slide 2 — Outline
> Here's how we'll walk through our work. I'll kick things off with the motivation and background on agentic systems, and set up our research questions. Kunpeng will then take you through the methodology — how we built our decoupled benchmarking harness, integrated hardware performance counters, and applied classical optimizations to our C++ kernels. Fazal will close by presenting our empirical results, our key finding — which we call the *Masking Effect* — and discuss the implications for future agentic architectures.

### Slide 3 — The Shift: From Chatbot to Autonomous Agent
> Almost all of the optimization work in generative AI over the past few years has focused on one thing: **accelerating the GPU forward-pass**. Researchers obsess over Time-To-First-Token, inter-token latency, KV-cache management — all GPU-side concerns.
>
> But LLMs are no longer just text generators. They are evolving into autonomous agents that reason, call external tools, parse results, and iterate in a loop. And the moment an agent stops generating tokens and starts *doing things* — querying databases, walking filesystems, parsing JSON — the execution bottleneck abruptly shifts **from the accelerator back to the host CPU**.
>
> And here's the problem: current systems literature largely treats the tool-execution pipeline as a black box. People report wall-clock numbers, but nobody is isolating the hardware penalties underneath. That's the gap our work fills.

### Slide 4 — ReAct Framework
> Let me quickly define the ReAct framework, which is the dominant paradigm for tool-calling agents today. ReAct stands for *Reasoning and Acting*.
>
> The loop works like this: the LLM reads a prompt, emits a structured JSON tool call, a Python orchestrator parses that JSON, executes a native subprocess, captures the standard output, and injects the result back into the model's context window. Then it repeats.
>
> The critical thing to notice is that **this loop is strictly serial**. Each step gates the next. And every single one of those arrows on the right involves CPU-bound work — parsing strings, managing context, serializing and deserializing data. This serial chain is the foundation of agentic workflows.

### Slide 5 — Architectural Divergence
> Now the deeper issue is that agentic workloads are *fundamentally mismatched* with GPU architecture. On the left you have what GPUs are great at: dense matrix math, massively parallel, predictable memory access, high arithmetic intensity.
>
> On the right you have what agents actually *do*: JSON parsing, database lookups, filesystem traversal. These are serial, branch-heavy, full of pointer chasing, and they hammer the last-level cache with misses. In systems terms, these workloads look much more like traditional datacenter RPCs than machine-learning kernels.
>
> So our hypothesis going in was: **agent tools are fundamentally hostile to GPU architectures, and the overhead of bolting an LLM onto them is probably enormous.** Our job was to measure exactly how enormous.

### Slide 6 — Research Questions & Contributions
> This led us to three research questions: How much CPU overhead does the orchestration layer actually impose? Do classical hardware optimizations — SIMD, prefetching, OpenMP — still help agentic tools? And critically, do kernel-level efficiency gains actually propagate to end-to-end latency?
>
> Our contributions, which we'll walk through in detail, are threefold. First, we quantify what we call the **Agentic Tax**: a slowdown of up to *69.5 times* on lightweight tools. Second, we characterize the microarchitectural bottlenecks specific to compute-, memory-, and I/O-bound agent tools. And third, we identify and name the **Masking Effect**: the phenomenon where hardware optimizations at the kernel level are rendered statistically invisible by the orchestration layer sitting on top.
>
> With that, I'll hand it over to Kunpeng to explain how we actually built the system to measure all of this.

---

## Part II — Kunpeng Zhang (Methodology, Kernels, Hardware Optimizations)

### Slide 7 — Decoupled System Architecture
> Thanks, TJ. So our first challenge was methodological: to accurately measure the overhead of LLM orchestration, we had to physically isolate the Python runtime from the native execution kernels. Otherwise, Python's GIL and garbage collector would contaminate every single hardware counter reading.
>
> We built a three-component decoupled architecture. The **Orchestrator**, a Python FastAPI server, handles prompt assembly, inference, and JSON routing. The **Execution Kernels** are compiled, standalone C++ binaries for our three workload classes. And the **Static Workloads** are pre-generated, deterministic payloads so we get run-to-run consistency.
>
> This decoupling is critical: because the C++ kernels run as separate subprocesses, they're completely isolated from Python's GIL and memory management. That gives us a pristine environment for profiling.
>
> For the model itself, we used Qwen 2.5 7B Instruct in FP16 with Scaled Dot-Product Attention. We turned off sampling entirely — `do_sample=False` — so that the LLM's tool-calling behavior is fully deterministic across runs.

### Slide 8 — PAPI Hardware Telemetry
> To capture low-level CPU behavior, we integrated the **Performance Application Programming Interface**, or PAPI, via its Python C-bindings. PAPI lets us tap directly into the Linux kernel's Performance Monitoring Unit, which gives us cycle-accurate telemetry of hardware events.
>
> We track four events: total cycles, total instructions, branch mispredictions, and L3 cache misses. Together these let us compute Instructions Per Cycle — the gold-standard metric for pipeline efficiency — and diagnose exactly what's stalling the CPU.
>
> Because we care about *where* the cycles go, we wrap PAPI around two specific windows: **Window 1** captures the prompt assembly and tokenization phase, and **Window 2** captures the output decoding and JSON extraction phase. We guard every EventSet in a try-finally block, because if a hallucinated JSON blows up the agentic loop, we still need to release the kernel's PMU registers cleanly. Otherwise the next run starves.
>
> All of this runs on an isolated Intel Xeon Gold 6242R at 3.1 gigahertz, which gives us predictable cycle-to-wall-clock conversions.

### Slide 9 — Three C++ Tool Kernels
> We designed three C++ kernels, each targeting a *different* microarchitectural bottleneck, so we could separate the effects.
>
> The **compute-bound** kernel is an Abstract Syntax Tree math evaluator. It parses deeply nested mathematical expressions recursively. This stresses branch prediction and tests raw instruction throughput.
>
> The **memory-bound** kernel is a SQLite database lookup. We deliberately use `LIKE` queries to force a full-table scan and bypass the indexes. This intentionally thrashes the L3 cache — it's a classic memory wall test.
>
> And the **I/O-bound** kernel is a recursive directory walker. It queries OS metadata to calculate total byte sizes, which stresses system calls and context switches.
>
> Each kernel manages its own internal PAPI profiling — counters are opened and closed *inside* the core workload loop — and each one prints a standardized JSON result to stdout so the orchestrator can consume it.

### Slide 10 — Control Baseline
> Now, to *prove* that the Agentic Tax exists as a distinct phenomenon, we needed a control. So we built a second execution path called `pipeline_mode="direct"`.
>
> In this mode, the GPU and the LLM are completely bypassed. Instead, the orchestrator uses plain regular expressions to extract target paths from the user's prompt and then invokes the C++ kernels directly. This represents the theoretical "speed of light" — the absolute minimum time to run the same tool with the same input.
>
> On the right, you have the full agentic loop where Qwen 2.5 actually parses the prompt, emits JSON, executes the tool, and injects the result. By subtracting the direct baseline from the full agentic loop, we mathematically isolate *exactly* the cycles, instructions, and cache misses consumed purely by LLM orchestration — nothing else.

### Slide 11 — Hardware Optimizations Applied
> Once we had the baseline framework, we applied three classical hardware optimizations to our kernels to see whether they would still pay off in an agentic context.
>
> First, **AVX2 SIMD vectorization** on the AST math evaluator. We compiled it with `-mavx2` so the compiler could use 256-bit wide vector registers to accelerate numerical parsing.
>
> Second, **software prefetching** on the database scan. We inserted `__builtin_prefetch` with a lookahead distance of 32 elements, explicitly telling the CPU's memory controller to pull data into L1 before the pipeline needs it — latency hiding.
>
> Third, **OpenMP concurrency** on the directory walker, so that I/O-bound system calls could overlap across multiple worker threads.
>
> All of these are textbook optimizations from a systems course. The question we're asking is: **do they still matter when you wrap them in an LLM?** That's what Fazal will show you next.

---

## Part III — Fazal Rehman (Results, Masking Effect, Conclusion)

### Slide 12 — Quantifying the Agentic Tax (Figure)
> Thanks, Kunpeng. Alright, this is the headline result, and it's the figure we spent the most time staring at.
>
> What you're looking at is end-to-end latency for all three workload classes — compute, memory, and I/O. The Y-axis is **logarithmic**. For each workload, the left bar is the direct C++ baseline and the right bar is the full agentic loop. The red annotations at the top show the multiplicative slowdown — in other words, the Agentic Tax.

### Slide 13 — Fixed-Cost Penalty
> Here's what we learned from that figure. The Agentic Tax is **not a proportional scalar** — it's a massive *fixed-cost* penalty, and it disproportionately punishes the tools that ought to be cheapest.
>
> The I/O directory walker natively runs in about 15 milliseconds. Routed through the LLM, it balloons by **69.5 times**. The compute-bound AST math evaluator suffers a 14.9 times penalty. The only workload that *doesn't* look catastrophic is the database scan — at 2.4 times — and that's not because the tax disappeared, it's pure Amdahl's Law. The DB scan natively takes about 764 milliseconds and burns 2.5 billion cycles, so the fixed orchestration cost is simply a smaller slice of a bigger pie.
>
> The takeaway: for the vast majority of real agentic tools, **execution time is entirely dominated by the framework, not by the tool itself.** That's a damning result for the current generation of agent architectures.

### Slide 14 — Multi-Step Gauntlet (Figure)
> But isolated tool calls are the *best* case for the framework. Real agents execute multi-step loops. So we built what we call the **Gauntlet prompt**: a single continuous context window where the agent has to execute the I/O walk, then the database query, then the math evaluation — sequentially.
>
> On the left you see the instruction volume on a log scale: framework versus tools. On the right you see Instructions Per Cycle — the efficiency gap between Python and native C++.

### Slide 15 — Cold Start and Compounding
> Two really interesting things pop out from the multi-step telemetry.
>
> **First, the framework has a massive cold-start penalty.** At Step 1, the Python orchestrator burns about 82 million instructions. That's a one-time cost — parsing and loading the JSON schemas. After that, the steady-state baseline drops to about 13.7 million instructions per step, but here's the key observation: it grows **linearly** up to 18.8 million because the LLM is stateless. Every step, the framework has to re-serialize a progressively larger conversation history. So agent autonomy literally compounds the tax.
>
> Interestingly, the framework's IPC actually *improves* across steps from 1.65 to 1.91 — that's because the later phases do more predictable string concatenation, which keeps the CPU pipeline fuller.
>
> **Second, the C++ kernels are completely independent of step count** — their metrics are dictated by payload alone. Instruction volume swings from 10 million for the I/O walk to 5 *billion* for the database scan. And importantly, the kernels run near the theoretical IPC ceiling of 2.0 — the DB scan actually hits 2.03 because the hardware L3 prefetcher handles sequential scans beautifully, while the AST evaluator dips slightly to 1.99 due to the hundreds of thousands of branch mispredictions from its nested `if/else` tree.
>
> So the coordination overhead is the problem — not the tools.

### Slide 16 — Vectorization Wall (Figure)
> OK, so knowing that tools are efficient and frameworks are wasteful, we tried to push the tools even further. And we hit walls. First experiment: AVX2 on the math evaluator.
>
> **Hypothesis:** 256-bit SIMD should accelerate numerical parsing.
> **Result: it failed.** The IPC stayed flat at around 1.99. Total cycles dropped from 84.1 million to 83.6 million — a measly 0.6 percent improvement.
>
> And the reason is fundamental: AST parsing is recursive, pointer-chasing, and branch-heavy. Every node lookup depends on the previous one. The compiler's auto-vectorizer simply can't find parallel lanes to fill. This tells us that recursive string-heavy agent tools hit a **fundamental vectorization wall** — they are provably resistant to classical SIMD acceleration.

### Slide 17 — Software Prefetching (Figure)
> So vectorization failed. Would explicit latency hiding succeed? We moved to the memory-bound DB scan and injected software prefetch instructions.
>
> On the right side of the figure you can see what we call the microarchitectural victory. The kernel IPC jumped **14.2 percent**, from 1.241 to 1.417. We saved about 66 million CPU cycles — roughly 50 milliseconds of wall-clock time on the kernel itself. The L3 miss count stayed constant, as expected for a cold-cache scan, but crucially those misses were being *overlapped* with useful computation instead of stalling the pipeline. Textbook success.
>
> But now look at the left side. The total end-to-end request took about **6.3 seconds**. So that 50-millisecond kernel win translated to... an end-to-end speedup of less than one percent. Almost nothing.

### Slide 18 — The Masking Effect (Figure)
> And this is the phenomenon we named the **Masking Effect**. This final figure brings it all together across every optimization domain we tried. The left panels show the system-level cycle distribution — the orchestration bar absolutely dwarfs the kernel bar. The right panels show the kernel-level IPC gains we worked hard to achieve.
>
> The contrast is the whole story. **Kernel-level victories are real, but they're rendered statistically invisible to the end user.** The framework sitting on top is so massive — measured in billions of cycles and seconds of GPU generation time — that a few dozen million CPU cycles saved inside the kernel simply vanish in the noise.
>
> This is the central finding of our paper, and it has huge implications: optimizing agent tools at the hardware level, in the *current* paradigm, is basically pointless. The bottleneck is somewhere else entirely.

### Slide 19 — Limitations
> Before concluding, a few honest limitations.
>
> First, we hit a PAPI profiling artifact on OpenMP. The high-level PAPI API only tracks the calling thread, so when our master thread spin-waited on the OpenMP barrier, it accumulated cycles but not instructions — our reported IPC collapsed to 0.14. That's a blind spot in our tooling, not a real regression. Future work would use thread-local PAPI EventSets to fix this.
>
> Second, the OpenMP I/O walker actually exhibited **textbook negative scaling** — it used 15 times *more* cycles than the serial baseline. The root cause was a global `std::atomic` counter creating severe L1 cache-line contention, combined with OS kernel serialization of the small-directory disk reads.
>
> And third, our workloads are static and sampling is off. In production, LLMs hallucinate, trigger multi-turn corrections, and generally behave worse than our deterministic setup. So our reported numbers are a **best-case lower bound**. In practice, the Agentic Tax is almost certainly larger than what we measured.

### Slide 20 — Conclusion and Future Work
> So let me bring this home. Our headline findings are that lightweight agent tools suffer up to 69.5 times latency slowdown; that a 14.2 percent kernel-level IPC improvement translates to less than 1 percent end-to-end speedup; and that this **Masking Effect** means the framework swallows every hardware victory we can manufacture inside the kernel.
>
> Based on these findings, we argue that future systems research should pivot away from optimizing tool kernels and focus aggressively on the orchestration layer itself. We propose three directions: **first**, high-throughput inference engines like vLLM with continuous batching and PagedAttention to shrink the GPU generation portion of the tax; **second**, compiled orchestrators — migrate the Python layer to Rust or C++ and swap in SIMD-accelerated JSON parsers like simdjson or orjson; and **third**, and most ambitiously, **standardized binary protocols**: fine-tuning LLMs to emit in-memory formats like Apache Arrow so that tools can execute directly on the generation buffer without any string parsing at all.
>
> The punchline is this: **until the orchestration layer is treated as a high-performance networking data plane, the promise of hardware-accelerated agentic AI will remain masked by the very framework that invokes it.**

### Slide 21 — Thank You / Q&A
> Thank you for your attention. We'd be happy to take any questions.

---

## Appendix — Anticipated Q&A

**Q: Why Qwen 2.5 7B and not a frontier model?**
> 7B lets us run deterministic, reproducible experiments on a single isolated node without API-induced variance. The Agentic Tax is a systems-level phenomenon rooted in the orchestration layer — switching to a 70B model would likely *increase* the tax, not decrease it, because the GPU generation portion grows.

**Q: Isn't a lot of this just Python being slow? Would Rust fix everything?**
> Python is a big part of it, and migrating to a compiled orchestrator is literally one of our three recommendations. But it's not the *only* cause — GPU generation time, JSON serialization, and context re-tokenization are all non-trivial components that would remain even with a Rust orchestrator.

**Q: Why did the DB scan only suffer a 2.4× penalty?**
> Pure Amdahl's Law. The DB scan natively takes ~764 ms and burns 2.5 billion cycles, so the fixed orchestration cost becomes a proportionally smaller slice. For any tool with a sub-100ms native runtime, the tax dominates completely.

**Q: Why did OpenMP make things *slower*?**
> Two compounding problems. The worker threads were fighting over a `std::atomic` global counter, which caused severe L1 cache-line contention. On top of that, parallelizing disk reads for a small directory hits immediate serialization bottlenecks inside the OS kernel itself. The fix would be lock-free thread-local accumulators plus much larger payloads.

**Q: Does your decoupled benchmark generalize beyond Qwen?**
> The methodology is model-agnostic. Any tool-calling LLM can be dropped in. What changes is the absolute magnitude of the tax, not its existence.
