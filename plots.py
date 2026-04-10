import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_expA_plot():
    direct_csv = "results/ExpA_Direct_Baseline.csv"
    llm_csv = "results/ExpA_LLM_Baseline.csv"

    if not os.path.exists(direct_csv) or not os.path.exists(llm_csv):
        print("❌ Error: Missing CSV files. Ensure both Direct and LLM tests were run.")
        return

    # Load Data
    df_dir = pd.read_csv(direct_csv)
    df_llm = pd.read_csv(llm_csv)

    # Extract base tool names from the test_name
    df_dir['tool_type'] = df_dir['test_name'].apply(lambda x: x.split('_')[0])
    df_llm['tool_type'] = df_llm['test_name'].apply(lambda x: x.split('_')[0])

    # Aggregate by taking the latest run
    dir_agg = df_dir.groupby('tool_type').last()
    llm_agg = df_llm.groupby('tool_type').last()

    # Align data
    tools = ['Math', 'DB', 'FS']
    direct_times = [dir_agg.loc[t, 'cpu_time_ms'] if t in dir_agg.index else 0 for t in tools]
    llm_times = [llm_agg.loc[t, 'cpu_time_ms'] if t in llm_agg.index else 0 for t in tools]

    # --- Setup Canvas ---
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(tools))
    width = 0.35

    # --- Plotting ---
    bars1 = ax.bar(x - width/2, direct_times, width, label='Direct C++ Execution (Speed of Light)', color='#2CA02C', edgecolor='black')
    bars2 = ax.bar(x + width/2, llm_times, width, label='LLM Agentic Execution (Orchestration Tax)', color='#D62728', edgecolor='black')

    # Log scale
    ax.set_yscale('log')
    
    ax.set_title('End-to-End Latency: The Agentic Tax', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Workload Type', fontsize=14)
    ax.set_ylabel('Total Request Time (ms) [Log Scale]', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['Compute (AST Math)', 'Memory (DB Scan)', 'I/O (FS Walk)'], fontsize=13)
    ax.legend(fontsize=12, loc='upper left')

    # Annotate multiplier above the LLM bars
    for i in range(len(tools)):
        if direct_times[i] > 0:
            multiplier = llm_times[i] / direct_times[i]
            # Pushed the text slightly higher above the bar (1.3x)
            ax.text(x[i] + width/2, llm_times[i] * 1.3, f"{multiplier:.1f}x", ha='center', va='bottom', fontsize=12, fontweight='bold', color='#D62728')

    # =================================================================
    # [NEW] Dynamically scale the Y-axis to give the text headroom
    # =================================================================
    max_llm_time = max(llm_times)
    ax.set_ylim(top=max_llm_time * 10) # 10x gives exactly one full "log tick" of headroom

    plt.tight_layout()
    output_filename = "results/ExpA_Agentic_Tax.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"📈 Success! Publication-ready plot saved to {output_filename}")


def generate_expB_plot(csv_path="results/ExpB_Multistep_Analysis.csv"):
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty:
        print("❌ Error: CSV is empty.")
        return

    # Grab the most recent Gauntlet test automatically
    latest_test = df['test_name'].iloc[-1]
    print(f"📊 Plotting data for: {latest_test}")
    df = df[df['test_name'] == latest_test]

    # --- Setup Canvas ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data pivots
    pivot_inst = df.pivot_table(index="step_number", columns="phase", values="instructions", aggfunc="sum").fillna(0)
    pivot_ipc = df.pivot_table(index="step_number", columns="phase", values="ipc", aggfunc="mean").fillna(0)
    
    x = np.arange(len(pivot_inst.index))
    width = 0.35

    # =================================================================
    # PLOT 1: INSTRUCTION VOLUME (LOG SCALE)
    # =================================================================
    ax1 = axes[0]
    ax1.bar(x - width/2, pivot_inst.get('LLM_Framework_Overhead', [0]*len(x)), width, 
            label='Python Framework Overhead', color='#4C72B0', edgecolor='black')
    
    ax1.bar(x + width/2, pivot_inst.get('Tool_Execution', [0]*len(x)), width, 
            label='C++ Tool Execution', color='#C44E52', edgecolor='black')
    
    # Log scale is mandatory here because DB is 5 Billion, Walker is 10 Million
    ax1.set_yscale('log')
    ax1.set_title('Instruction Volume per ReAct Step', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel('Agentic Loop Step', fontsize=13)
    ax1.set_ylabel('Total Instructions Executed (Log Scale)', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Step {i}" for i in pivot_inst.index])
    ax1.legend(loc="upper right")

    # =================================================================
    # PLOT 2: MICROARCHITECTURAL EFFICIENCY (IPC)
    # =================================================================
    ax2 = axes[1]
    ax2.bar(x - width/2, pivot_ipc.get('LLM_Framework_Overhead', [0]*len(x)), width, 
            label='Python Framework Overhead', color='#4C72B0', edgecolor='black')
    
    ax2.bar(x + width/2, pivot_ipc.get('Tool_Execution', [0]*len(x)), width, 
            label='C++ Tool Execution', color='#C44E52', edgecolor='black')
    
    ax2.set_title('CPU Utilization Efficiency (IPC)', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xlabel('Agentic Loop Step', fontsize=13)
    ax2.set_ylabel('Instructions Per Cycle (Higher = Better)', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Step {i}" for i in pivot_ipc.index])
    
    # Set y-limit slightly higher to make room for tool labels
    ax2.set_ylim(0, 2.5)

    # Annotate the specific tools on top of the red bars
    for i, step in enumerate(pivot_ipc.index):
        tool_row = df[(df['step_number'] == step) & (df['phase'] == 'Tool_Execution')]
        if not tool_row.empty:
            tool_name = tool_row.iloc[0]['tool_name']
            if tool_name != "None":
                val = pivot_ipc.loc[step, 'Tool_Execution']
                # Clean up the name for the label (e.g., query_database -> query\ndatabase)
                clean_name = str(tool_name).replace('_', '\n')
                ax2.text(x[i] + width/2, val + 0.05, clean_name, ha='center', va='bottom', 
                         fontsize=10, fontweight='bold', color='#8B0000')

    # --- Save and Render ---
    plt.tight_layout()
    output_filename = f"results/{latest_test}_analysis.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"📈 Success! Publication-ready plot saved to {output_filename}")


def generate_masking_plot():
    vanilla_path = "results/Exp3_LLM_Vanilla_Detailed.csv"
    opt_path = "results/Exp3_LLM_Optimized_Detailed.csv"

    if not os.path.exists(vanilla_path) or not os.path.exists(opt_path):
        print("❌ Error: Missing Experiment 3 CSV files.")
        return

    # Load data
    df_v = pd.read_csv(vanilla_path)
    df_o = pd.read_csv(opt_path)

    def get_metrics(df):
        # Extract framework cycles and tool cycles
        fw_cycles = df[df['phase'] == 'LLM_Framework_Overhead']['cycles'].sum()
        tool_cycles = df[df['phase'] == 'Tool_Execution']['cycles'].sum()
        return fw_cycles, tool_cycles

    v_fw, v_tool = get_metrics(df_v)
    o_fw, o_tool = get_metrics(df_o)

    # Calculate Speedups
    tool_speedup = v_tool / o_tool if o_tool > 0 else 1
    total_speedup = (v_fw + v_tool) / (o_fw + o_tool)

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 7))

    labels = ['Vanilla (Baseline)', 'Hardware-Optimized (AVX2)']
    fw_data = [v_fw, o_fw]
    tool_data = [v_tool, o_tool]

    # Plot Stacked Bars
    ax.bar(labels, fw_data, label='LLM Framework Overhead', color='#4C72B0', edgecolor='black', width=0.6)
    ax.bar(labels, tool_data, bottom=fw_data, label='Tool Execution (Math Kernel)', color='#C44E52', edgecolor='black', width=0.6)

    # Formatting
    ax.set_ylabel('Total CPU Cycles', fontsize=14, fontweight='bold')
    ax.set_title('The Masking Effect: Hardware Speedup vs. System Latency', fontsize=16, pad=20, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)

    # Annotations for the Paper
    # Tool Speedup Text
    ax.text(0, v_fw + v_tool/2, f"Baseline\nTool", ha='center', color='white', fontweight='bold')
    ax.text(1, o_fw + o_tool/2, f"{tool_speedup:.1f}x Faster\nKernel", ha='center', color='black', fontweight='bold', fontsize=11)
    
    # System Summary Annotation
    summary_text = (f"Kernel Speedup: {tool_speedup:.2f}x\n"
                    f"System Speedup: {total_speedup:.2f}x\n"
                    f"Orchestration Tax: {((v_fw/(v_fw+v_tool))*100):.1f}%")
    
    plt.annotate(summary_text, xy=(0.5, 0.2), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.2),
                 fontsize=12, fontweight='bold', ha='center')

    plt.tight_layout()
    output_png = "results/Exp3_Masking_Effect.png"
    plt.savefig(output_png, dpi=300)
    print(f"📈 Success! Plot saved to {output_png}")

if __name__ == "__main__":
    generate_masking_plot()

    