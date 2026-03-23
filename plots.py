import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def base_token_generation_benchmark(csv_path="results/cpu_profiling_log.csv", output_file="results/tokenizer_overhead_comparison.png"):
    """
    Reads the CPU profiling CSV, extracts the latest baseline vs stress test runs,
    and generates a 2x2 microarchitectural bottleneck grid.
    """
    # 1. Load the data
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 2. Filter and Extract Data
    # Drop the warmup rows (where the prompt was just the word "warmup", length ~6)
    df_clean = df[df["prompt_length_chars"] > 10].copy()

    # Grab the last two rows to represent the most recent Baseline and Stress Test runs
    latest_runs = df_clean.tail(2)

    if len(latest_runs) < 2:
        print("Not enough data to compare. Please run the test script first.")
        return

    labels = ['Baseline\n(18 chars)', 'Stress Test\n(16k chars)']

    # Extract metrics into lists
    cpu_time = latest_runs['cpu_time_ms'].values
    instructions = latest_runs['total_instructions'].values
    ipc = latest_runs['ipc'].values
    branch_misses = latest_runs['branch_mispredictions'].values
    llc_misses = latest_runs['llc_misses'].values

    # 3. Setup the Plot Canvas (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Agentic Microarchitectural Bottleneck: Tokenizer Overhead', fontsize=16, fontweight='bold')

    # --- Plot 1: Total Instructions ---
    axs[0, 0].bar(labels, instructions, color=['#4C72B0', '#C44E52'])
    axs[0, 0].set_title('Total Instructions Completed', fontsize=12)
    axs[0, 0].set_ylabel('Count (10s of Millions)')
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 2: CPU Time ---
    axs[0, 1].bar(labels, cpu_time, color=['#4C72B0', '#C44E52'])
    axs[0, 1].set_title('CPU Execution Time', fontsize=12)
    axs[0, 1].set_ylabel('Milliseconds (ms)')
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 3: IPC ---
    axs[1, 0].bar(labels, ipc, color=['#4C72B0', '#C44E52'])
    axs[1, 0].set_title('Instructions Per Cycle (IPC)', fontsize=12)
    axs[1, 0].set_ylabel('IPC')
    axs[1, 0].set_ylim(0, max(ipc) * 1.2) # Give some headroom for the label
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 4: Hardware Misses (Grouped Bar Chart) ---
    x = np.arange(len(labels))
    width = 0.35
    axs[1, 1].bar(x - width/2, branch_misses, width, label='Branch Misses', color='#55A868')
    axs[1, 1].bar(x + width/2, llc_misses, width, label='LLC Misses', color='#8172B3')
    axs[1, 1].set_title('Microarchitectural Misses', fontsize=12)
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].legend()
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Save the graphic
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit the main title
    plt.savefig(output_file, dpi=300)
    print(f"✅ Success! Plot saved locally to {output_file}")
    
    plt.close(fig)

def compute_bound_benchmark(output_file="results/math_vs_baseline.png"):
    """
    Generates a 2x2 microarchitectural bottleneck grid comparing the 
    LLM baseline token generation against the compute-bound C++ math tool.
    """
    # 1. Load the exact data points from your test run
    data = {
        "test_name": ["Base Test: Token Generation", "Test 3: Agentic Tool Call"],
        "cpu_time_ms": [22.38, 20.7],
        "total_instructions": [98212662, 244643519],
        "ipc": [2.343, 1.866],
        "branch_mispredictions": [47595, 497257],
        "llc_misses": [31054, 120212]
    }
    df = pd.DataFrame(data)

    labels = ['Token Generation\n(Baseline)', 'Agentic Math Tool\n(Compute Bound)']

    # 2. Setup the Plot Canvas (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Agentic Microarchitectural Bottleneck: Math Tool vs Baseline', fontsize=16, fontweight='bold')

    # --- Plot 1: Total Instructions ---
    axs[0, 0].bar(labels, df['total_instructions'], color=['#4C72B0', '#C44E52'])
    axs[0, 0].set_title('Total Instructions Completed', fontsize=12)
    axs[0, 0].set_ylabel('Count (100s of Millions)')
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 2: CPU Time ---
    axs[0, 1].bar(labels, df['cpu_time_ms'], color=['#4C72B0', '#C44E52'])
    axs[0, 1].set_title('CPU Execution Time', fontsize=12)
    axs[0, 1].set_ylabel('Milliseconds (ms)')
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 3: IPC ---
    axs[1, 0].bar(labels, df['ipc'], color=['#4C72B0', '#C44E52'])
    axs[1, 0].set_title('Instructions Per Cycle (IPC)', fontsize=12)
    axs[1, 0].set_ylabel('IPC')
    axs[1, 0].set_ylim(0, max(df['ipc']) * 1.2) # Give headroom for label
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 4: Hardware Misses (Grouped Bar Chart) ---
    x = np.arange(len(labels))
    width = 0.35
    axs[1, 1].bar(x - width/2, df['branch_mispredictions'], width, label='Branch Misses', color='#55A868')
    axs[1, 1].bar(x + width/2, df['llc_misses'], width, label='LLC Misses', color='#8172B3')
    axs[1, 1].set_title('Microarchitectural Misses', fontsize=12)
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].legend()
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Save the graphic
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300)
    print(f"✅ Success! Plot saved locally to {output_file}")
    
    # Close the plot to free memory
    plt.close(fig)

def tool_class_comparison_benchmark(output_file="results/four_way_tool_class_comparison.png"):
    """
    Generates a 2x2 grid comparing all four current workloads:
    Baseline Token Generation vs. Compute-Bound vs. I/O-Bound vs. Analytics (DB).
    """
    # 1. Load the exact data points from your test runs
    # (Note: Assuming ~105M total instructions for the DB query based on your IPC/Time ratio)
    data = {
        "test_name": ["Base Test: Token Generation", "Test 3: Agentic Tool Call", "Test 4: I/O Walker Tool Call", "Test 5: DB Memory-Bound Tool Call"],
        "cpu_time_ms": [22.38, 20.7, 21.61, 24.97],
        "total_instructions": [98212662, 244643519, 89019182, 105642391], 
        "ipc": [2.343, 1.866, 1.663, 2.057],
        "branch_mispredictions": [47595, 497257, 164528, 16996029],
        "llc_misses": [31054, 120212, 72276, 92226]
    }
    df = pd.DataFrame(data)

    labels = ['Baseline\n(Token Gen)', 'Math Tool\n(Compute)', 'FS Walker\n(I/O)', 'DB Query\n(Analytics)']
    
    # Colors: Blue (Baseline), Red (Compute), Green (I/O), Purple (Analytics)
    bar_colors = ['#4C72B0', '#C44E52', '#55A868', '#9B59B6']

    # 2. Setup the Plot Canvas (2x2 grid)
    # Increased width slightly to accommodate 4 labels comfortably
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Agentic Microarchitectural Bottleneck: 4-Way Workload Comparison', fontsize=18, fontweight='bold')

    # --- Plot 1: Total Instructions ---
    axs[0, 0].bar(labels, df['total_instructions'], color=bar_colors)
    axs[0, 0].set_title('Total Instructions Completed', fontsize=12)
    axs[0, 0].set_ylabel('Count (100s of Millions)')
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 2: CPU Time ---
    axs[0, 1].bar(labels, df['cpu_time_ms'], color=bar_colors)
    axs[0, 1].set_title('CPU Execution Time', fontsize=12)
    axs[0, 1].set_ylabel('Milliseconds (ms)')
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 3: IPC ---
    axs[1, 0].bar(labels, df['ipc'], color=bar_colors)
    axs[1, 0].set_title('Instructions Per Cycle (IPC)', fontsize=12)
    axs[1, 0].set_ylabel('IPC (Higher is Better)')
    axs[1, 0].set_ylim(0, max(df['ipc']) * 1.2)
    
    # Add value labels on top of the IPC bars for clarity
    for i, v in enumerate(df['ipc']):
        axs[1, 0].text(i, v + 0.05, f"{v:.3f}", ha='center', fontweight='bold')
        
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 4: Hardware Misses (Log Scale) ---
    x = np.arange(len(labels))
    width = 0.35
    
    # We MUST use a log scale here because the DB branch misses (16.9M) will completely dwarf everything else
    axs[1, 1].bar(x - width/2, df['branch_mispredictions'], width, label='Branch Misses', color='#E1812C')
    axs[1, 1].bar(x + width/2, df['llc_misses'], width, label='LLC Misses', color='#8172B3')
    
    axs[1, 1].set_yscale('log') # Crucial addition for the 4-way chart
    axs[1, 1].set_title('Microarchitectural Misses (Log Scale)', fontsize=12)
    axs[1, 1].set_ylabel('Count (Log10)')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].legend()
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Save the graphic
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300)
    print(f"✅ Success! Plot saved locally to {output_file}")
    
    plt.close(fig)
    
if __name__ == "__main__":
    # base_token_generation_benchmark()
    # compute_bound_benchmark()
    tool_class_comparison_benchmark()