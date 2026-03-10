import os
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json
from dotenv import load_dotenv

config_path = Path(__file__).parent.parent / "config.env"
if config_path.exists():
    load_dotenv(config_path)

BASE_LOG_DIR = os.getenv("BASE_LOG_DIR", "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs")
VENV_PYTHON = os.getenv("VENV_PATH", "/Users/haseeb/Code/iisc/bedrockAC/.venv/bin/python")
PLOTTER_LOG_EXT = os.getenv("PLOTTER_LOG_EXT", "_aggregated_logs/2026-03-10/14-44-36")
PLOTTER_PLOTS_EXT = os.getenv("PLOTTER_PLOTS_EXT", "_plots")

LOGS_DIR = os.path.join(BASE_LOG_DIR, PLOTTER_LOG_EXT)

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
    discovered = defaultdict(lambda: {"empty": "", "naive": "", "c": "", "m": "", "mc": ""})
    
    if not os.path.exists(LOGS_DIR):
        print(f"Directory not found: {LOGS_DIR}")
        return {}

    for file_name in os.listdir(LOGS_DIR):
        if not file_name.endswith(".json"):
            continue
            
        # Example format: arxiv-batch_1-memory_e.json
        name_no_ext = file_name[:-5]
        parts = name_no_ext.split("-")
        print(parts)
        
        # We expect at least workload-batch_X-memory_Y
        if len(parts) >= 3:
            workload = parts[0]
            batch_part = parts[1]
            
            group_key = f"{workload}-{batch_part}"
            
            file_path = os.path.join(LOGS_DIR, file_name)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    config_id = data.get("log_metadata", {}).get("config_id")
                    
                    if config_id == "E":
                        discovered[group_key]["empty"] = file_path
                    elif config_id == "N":
                        discovered[group_key]["naive"] = file_path
                    elif config_id == "C":
                        discovered[group_key]["c"] = file_path
                    elif config_id == "M":
                        discovered[group_key]["m"] = file_path
                    elif config_id == "MC":
                        discovered[group_key]["mc"] = file_path
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

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
    plots_dir = os.path.join(BASE_LOG_DIR, PLOTTER_PLOTS_EXT, timestamp)
    all_exceeds = []
    all_output_lines = []
    
    for group_key, runs in papers.items():
        # group_key is like arxiv-batch_1
        parts = group_key.split("-batch_")
        batch_num = parts[1] if len(parts) > 1 else "Unknown"
        
        # We assign an arbitrary agent_type name here, previously it was 'react_monolith'
        # The logs are aggregated and config-based now, so we just use the workload as the type
        agent_type = parts[0]
        
        # Get nice title if it exists, otherwise formulate one
        if agent_type == "log":
            plot_title = f"Log Analytics (Log File {batch_num})"
        elif batch_num in PAPER_TITLES:
            plot_title = PAPER_TITLES[batch_num][0]
        else:
            plot_title = f"Research Summary (Batch {batch_num})"
        
        cmd_args = [
            "--paper", plot_title,
            "--agent_type", agent_type
        ]

        e_log = runs.get("empty")
        n_log = runs.get("naive")
        c_log = runs.get("c")
        m_log = runs.get("m")
        mc_log = runs.get("mc")

        if e_log: cmd_args.extend(["--e", e_log])
        if n_log: cmd_args.extend(["--n", n_log])
        if c_log: cmd_args.extend(["--c", c_log])
        if m_log: cmd_args.extend(["--m", m_log])
        if mc_log: cmd_args.extend(["--mc", mc_log])

        for name, script_path, plotter_dir_name in plotters:
            out_filename = f"{plotter_dir_name}-{agent_type}-batch_{batch_num}.pdf"
            out_path = os.path.join(plots_dir, out_filename)
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            current_cmd_args = cmd_args + ["--out", out_path]
            
            print(f"Running {name} plotter for '{plot_title}' ({agent_type})...")
            # Run using the python from the venv, capture output
            result = subprocess.run(
                [VENV_PYTHON, str(script_path)] + current_cmd_args,
                capture_output=True, text=True
            )
            
            # Print and collect output
            if result.stdout:
                print(result.stdout, end='')
                all_output_lines.append(result.stdout)
                # Collect [EXCEEDS] lines
                for line in result.stdout.splitlines():
                    if line.strip().startswith("[EXCEEDS]"):
                        all_exceeds.append(f"[{name}] {line.strip()}")
            if result.stderr:
                print(result.stderr, end='')

    # --- Consolidated Exceeds Summary ---
    summary_lines = []
    summary_lines.append(f"\n{'!'*80}")
    if all_exceeds:
        summary_lines.append(f"EXCEEDS SUMMARY - {len(all_exceeds)} value(s) exceeded graph scale")
        summary_lines.append(f"{'!'*80}")
        for entry in all_exceeds:
            summary_lines.append(entry)
    else:
        summary_lines.append("EXCEEDS SUMMARY - No values exceeded the graph scale")
        summary_lines.append(f"{'!'*80}")
    summary_lines.append(f"{'!'*80}")
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Save report to the plots directory
    report_path = os.path.join(plots_dir, "_report.txt")
    os.makedirs(plots_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("".join(all_output_lines))
        f.write("\n")
        f.write(summary_text)
        f.write("\n")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    generate_plots()
