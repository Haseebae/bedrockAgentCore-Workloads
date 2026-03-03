import os
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime

LOGS_DIR = "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/_aggregated_logs"
VENV_PYTHON = "/Users/haseeb/Code/iisc/bedrockAC/.venv/bin/python"

# Mapping batch numbers to paper titles for backwards compatibility/nice titles
PAPER_TITLES = {
    "1": ("Research Summary (Paper 1)", "Multi-scale competition in the Majorana-Kondo system"),
    "2": ("Research Summary (Paper 2)", "Chondrule formation by collisions of planetesimals containing volatiles triggered by Jupiter's formation"),
    "3": ("Research Summary (Paper 3)", "Resolving the flat-spectrum conundrum: clumpy aerosol distributions in sub-Neptune atmospheres")
}

def discover_papers():
    """
    Scans the LOGS_DIR and groups the JSON logs by workload and batch.
    Returns a structured dictionary similar to the old `papers` list.
    """
    discovered = defaultdict(lambda: {"empty": "", "naive": "", "m": ""})
    
    if not os.path.exists(LOGS_DIR):
        print(f"Directory not found: {LOGS_DIR}")
        return {}

    for file_name in os.listdir(LOGS_DIR):
        if not file_name.endswith(".json"):
            continue
            
        # Example format: arxiv-batch_1-memory_e.json
        name_no_ext = file_name[:-5]
        parts = name_no_ext.split("-")
        
        # We expect at least workload-batch_X-memory_Y
        if len(parts) >= 3:
            workload = parts[0]
            batch_part = parts[1]
            memory_part = parts[2]
            
            group_key = f"{workload}-{batch_part}"
            
            file_path = os.path.join(LOGS_DIR, file_name)
            
            if "memory_e" in memory_part:
                discovered[group_key]["empty"] = file_path
            elif "memory_n" in memory_part:
                discovered[group_key]["naive"] = file_path
            elif "memory_m" in memory_part:
                discovered[group_key]["m"] = file_path

    return discovered

def generate_plots():

    base_dir = Path(__file__).parent
    plotters = [
        ("Infrastructure Cost", base_dir / "infra_plotter.py", "infra_cost"),
        ("Latency", base_dir / "latency_plotter.py", "latency"),
        ("Tokens", base_dir / "tokens_plotter.py", "tokens")
    ]

    papers = discover_papers()
    if not papers:
        print("No aggregated logs found to plot.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    for group_key, runs in papers.items():
        # group_key is like arxiv-batch_1
        parts = group_key.split("-batch_")
        batch_num = parts[1] if len(parts) > 1 else "Unknown"
        
        # Get nice title if it exists, otherwise formulate one
        if batch_num in PAPER_TITLES:
            plot_title = PAPER_TITLES[batch_num][0]
        else:
            plot_title = f"Research Summary (Batch {batch_num})"
        
        # We assign an arbitrary agent_type name here, previously it was 'react_monolith'
        # The logs are aggregated and config-based now, so we just use the workload as the type
        agent_type = parts[0]
        
        cmd_args = [
            "--paper", plot_title,
            "--agent_type", agent_type
        ]

        e_log = runs.get("empty")
        n_log = runs.get("naive")
        m_log = runs.get("m")

        if e_log: cmd_args.extend(["--e", e_log])
        if n_log: cmd_args.extend(["--n", n_log])
        if m_log: cmd_args.extend(["--m", m_log])

        for name, script_path, plotter_dir_name in plotters:
            out_filename = f"{plotter_dir_name}-{agent_type}-batch_{batch_num}.pdf"
            out_path = f"/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/_plots/{timestamp}/{out_filename}"
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            current_cmd_args = cmd_args + ["--out", out_path]
            
            print(f"Running {name} plotter for '{plot_title}' ({agent_type})...")
            # Run using the python from the venv
            subprocess.run([VENV_PYTHON, str(script_path)] + current_cmd_args)

if __name__ == "__main__":
    generate_plots()
