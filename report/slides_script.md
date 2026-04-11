# Presentation Script — Beyond the GPU
### ~10 Minutes | 15 Slides

---

## Slide 1: Title Slide (~30 seconds)

> Good afternoon everyone. Today we're presenting "Beyond the GPU: A Microarchitectural Analysis of LLM Tool-Calling Kernels." I'm presenting alongside Kunpeng and TJ, and we'll be showing you how the push toward agentic AI is creating a hidden performance crisis — one that has nothing to do with the GPU.

---

## Slide 2: The Agentic AI Revolution (~50 seconds)

> Let's set the stage. LLMs have evolved. They're no longer just text generators — they're now autonomous agents. They invoke APIs, query databases, execute code, and chain these operations together in what we call "ReAct" loops — Reasoning and Acting.
>
> Here's the flow: the model reasons, outputs structured JSON, the orchestrator parses that JSON, dispatches a tool, captures the result, re-injects it, and repeats. It's a strictly serial pipeline.
>
> Now here's the problem: while everyone's been optimizing the GPU — TTFT, inter-token latency, batch throughput — nobody has really characterized what's happening on the CPU *between* those reasoning steps. That CPU orchestration overhead is what we're calling the "Agentic Tax," and it's largely invisible in current benchmarks.

---

## Slide 3: Research Question & Contributions (~40 seconds)

> So our core question is simple: *How much latency do agentic frameworks silently impose, and can classical hardware optimizations overcome it?*
>
> On the methodology side, we built a fully decoupled profiling harness that separates CPU orchestration from GPU inference. We used PAPI — the Performance Application Programming Interface — to capture cycle-accurate hardware telemetry including IPC, L3 cache misses, and branch mispredictions.
>
> Our key findings are twofold. First, we quantified the Agentic Tax across compute, memory, and I/O workloads. Second, we discovered what we're calling the "Masking Effect" — where real, validated kernel-level improvements simply vanish at the system level.

---

## Slide 4: System Architecture (~50 seconds)

> Let me walk you through our architecture. We have three strictly isolated layers.
>
> First, the Python orchestrator built with FastAPI. It handles prompt assembly, tokenization, and JSON routing — all the things a real agentic framework does. We used Qwen 2.5-7B in FP16 as our base model, with deterministic generation disabled for sampling to ensure reproducibility.
>
> Second, the C++ execution kernels. These are compiled native binaries that are *completely* isolated from Python's GIL and garbage collector. Each kernel has its own internal PAPI profiling, so we get pristine hardware telemetry without Python's cache pollution contaminating the results.
>
> Third, static workloads — pre-generated, deterministic payloads that ensure consistency across runs.
>
> The PAPI telemetry is captured in two specific CPU windows: one around prompt setup and tokenization, and another around JSON parsing and serialization. This windowing ensures we don't accidentally count GPU idle time as CPU overhead.

---

## Slide 5: Tool Kernels (~40 seconds)

> We designed three C++ kernels, each targeting a different fundamental bottleneck domain.
>
> The *compute-bound* kernel is an AST mathematical evaluator — it parses deeply nested expressions, stressing the branch predictor with massive control-flow divergence.
>
> The *memory-bound* kernel is a SQLite full-table scan. We explicitly bypass database indexing to force the workload to thrash the Last-Level Cache and saturate DRAM bandwidth.
>
> And the *I/O-bound* kernel is a recursive directory walker using POSIX system calls, testing OS context switching and storage latency.
>
> These three kernels give us comprehensive coverage of the kinds of tools real agentic systems use.

---

## Slide 6: The Agentic Tax — Key Result (~50 seconds)

> Now, the headline result. This chart compares direct C++ execution against the full LLM agentic loop on a logarithmic scale.
>
> Look at these numbers. The I/O kernel, which executes in about 15 milliseconds natively, suffers a *69.5 times* latency slowdown when routed through the agentic framework. The compute kernel sees a 14.9x penalty. And even the memory-bound database scan, which natively takes about 764 milliseconds, still gets hit with a 2.4x multiplier.
>
> This is not a proportional overhead. It's a massive *fixed-cost penalty*. The GPU generation time plus the Python JSON serialization overhead is so large that it completely dwarfs the actual tool execution.

---

## Slide 7: Interpreting the Agentic Tax (~40 seconds)

> Let's interpret this through Amdahl's Law. The Agentic Tax is a fixed cost — GPU token generation plus CPU-bound string processing. When the native tool is fast, like the I/O walker at 15ms, that fixed cost dominates and you see a 69.5x slowdown. When the tool itself is inherently expensive, like the 764ms database scan, the fixed overhead is proportionally smaller — hence only 2.4x.
>
> The critical takeaway: for the vast majority of agentic tasks, execution time is dictated by *framework overhead*, not the tool's computational complexity. This completely inverts the optimization priority.

---

## Slide 8: Multi-Step Analysis (~40 seconds)

> Next, we evaluated what happens in realistic multi-step workflows. We ran a "Gauntlet" prompt that forces the agent to sequentially execute all three tools within a single context window.
>
> The left panel shows instruction volume on a log scale. The right panel shows IPC — instructions per cycle — for the framework versus the tool kernels. Notice the enormous efficiency gap. The Python framework operates well below the compiled C++ kernels.

---

## Slide 9: Multi-Step Insights (~50 seconds)

> Two key phenomena emerge. On the framework side, Step 1 hits you with a "Cold Start Penalty" — 82 million instructions just for initialization: JSON schema parsing, memory allocation, async environment setup. After that, the base overhead drops, but then *grows linearly* from 13.7 million to 18.8 million instructions as the context window balloons with each step. The framework has to re-serialize the *entire* conversation history every time because the LLM is stateless.
>
> On the C++ side, the kernels are consistently operating near the theoretical maximum of the CPU at about 2.0 IPC. The DB scan hits 2.03 IPC thanks to the hardware L3 prefetcher, and the AST evaluator sits at 1.99 IPC with only a slight dip from branch mispredictions.
>
> The bottom line: orchestration overhead is a *compounding* tax that scales with agent autonomy.

---

## Slide 10: The Vectorization Wall (~40 seconds)

> Now, can we optimize our way out? We tried. For the compute-bound AST kernel, we compiled with AVX2 — that's 256-bit SIMD registers — under the hypothesis that wider data paths would accelerate numerical parsing.
>
> It didn't work. The recursive parsing logic is fundamentally branch-heavy with pointer-chasing access patterns. The compiler simply cannot auto-vectorize this kind of code. IPC stayed flat at 1.99, and total cycles dropped by just 0.6%. This is what we call the "Vectorization Wall" — string-heavy agentic tools are *intrinsically scalar*.
>
> Interestingly, this is the exact same problem that plagues JSON parsing in the orchestrator itself. Standard recursive descent parsers hit the same wall. Libraries like simdjson solve it through radical algorithmic redesign, not compiler flags.

---

## Slide 11: Software Prefetching (~40 seconds)

> For the memory-bound kernel, we had more success at the *kernel level*. We injected explicit software prefetch instructions into the scanning loop with a 32-element lookahead. This commands the CPU to asynchronously fetch data into L1 cache before the pipeline actually needs it.
>
> The results were impressive *microarchitecturally*: IPC jumped 14.2% from 1.24 to 1.42, we saved 66 million cycles — about 50 milliseconds of wall-clock time. The L3 miss count stayed the same because the data is inherently cache-unfriendly, but the *temporal penalty* of those misses was successfully hidden. A legitimate, validated hardware optimization.

---

## Slide 12: The Masking Effect (~50 seconds)

> But here's the punchline — and this is the central finding of our research. Despite that 14.2% IPC improvement and 50ms saved at the kernel level, the end-to-end system speedup was *less than 1%*.
>
> Why? Because the total request latency is about 6.3 seconds. That's dominated by GPU autoregressive generation coupled with Python JSON serialization, history concatenation, and subprocess routing. Our 50ms kernel improvement is literally invisible in the noise.
>
> We call this the "Masking Effect": the execution bloat of the orchestration framework is so massive that it swallows any microarchitectural victories achieved within the tool kernels. This is our definitive proof that optimizing isolated tools provides *diminishing returns* — the bottleneck is the orchestrator.

---

## Slide 13: Future Work (~50 seconds)

> So how do we fix this? The report outlines four major research directions.
>
> First, *speculative execution*: frameworks like PASTE can dispatch tool calls *before* the LLM confirms intent, overlapping CPU tool latency with GPU prefill time. "Act while thinking."
>
> Second, *semantic caching*: over 40% of tool calls in production are redundant. The VAAC algorithm selectively caches expensive API results while evicting cheap ones, cutting tool latency by up to 34%.
>
> Third, *JIT model routing*: instead of forcing one model through an entire workflow DAG, Aragog dynamically routes each stage to the most efficient available model — up to 217% throughput improvement.
>
> And fourth, we need to move beyond JSON entirely. SIMD-accelerated parsers like simdjson or native binary protocols like Apache Arrow could eliminate the CPU serialization bottleneck at its root.

---

## Slide 14: Conclusion (~40 seconds)

> To wrap up: the orchestration layer is the bottleneck. Not the GPU. Not the tool kernels. The orchestrator.
>
> We showed up to a 69.5x latency penalty for lightweight tools. Framework overhead compounds linearly with multi-step autonomy. Even validated kernel-level optimizations yield less than 1% system improvement due to the Masking Effect. And the path forward requires orchestrator-engine co-design — fundamentally rethinking how the framework operates — not building faster tools.
>
> As we write in the report: "Until the orchestration layer is treated as a high-performance networking data plane, the potential of hardware-accelerated agentic tools will remain hidden beneath the execution tax."

---

## Slide 15: Thank You (~10 seconds)

> Thank you for your time. We're happy to take any questions.

---

**Total estimated time: ~9.5–10.5 minutes**
