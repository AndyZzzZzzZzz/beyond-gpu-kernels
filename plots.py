import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Load the data
csv_path = "results/cpu_profiling_log.csv"
if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

# 2. Filter and Extract Data
# Drop the warmup rows (where the prompt was just the word "warmup", length ~6)
df_clean = df[df["prompt_length_chars"] > 10].copy()

# Grab the last two rows to represent the most recent Baseline and Stress Test runs
latest_runs = df_clean.tail(2)

if len(latest_runs) < 2:
    print("Not enough data to compare. Please run the test_client.py script first.")
    exit(1)

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
output_file = "results/tokenizer_overhead_comparison.png"
plt.savefig(output_file, dpi=300)
print(f"✅ Success! Plot saved locally to {output_file}")