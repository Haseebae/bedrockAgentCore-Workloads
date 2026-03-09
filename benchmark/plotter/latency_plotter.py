import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pathlib import Path

# --- Configuration ---
CONFIG_ORDER = ['E', 'N', 'C', 'M', 'MC']
QUERIES = ['Query1', 'Query2', 'Query3']

PAPER_COLORS = {
    'Paper1': '#4472C4',  
    'Paper2': '#8B4513',  
    'Paper3': '#385723',  
}

def get_agent_color(base_hex, agent_name):
    """Generates light, medium, and dark shades of the base color for the stack."""
    base_color = base_hex.lstrip('#')
    rgb = tuple(int(base_color[i:i+2], 16) for i in (0, 2, 4))
    
    if agent_name == "plan":
        new_rgb = tuple(int(c * 0.4 + 255 * 0.6) for c in rgb)
    elif agent_name == "act":
        new_rgb = tuple(int(c * 0.8) for c in rgb)
    else:  # evaluate
        new_rgb = tuple(int(c * 0.4) for c in rgb)
        
    new_rgb = tuple(min(255, max(0, int(c))) for c in new_rgb)
    return '#%02x%02x%02x' % new_rgb

def extract_metrics_from_trace(filepath):
    """Extracts and sums LLM and MCP latencies from the specific trace JSON format."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    results = {}
    traces = data.get("traces", {})
    
    for iter_key, iter_data in traces.items():
        success_status = iter_data.get("success", True)
        query_name = f"Query{iter_key}"
        
        # Storing sums natively instead of raw values, as plot logic averages values
        plan_lat, act_lat, eval_lat = 0, 0, 0
        mcp_calls = 0 # Now accurately getting mcp_calls from the tools node in aggregated logs
        
        graphs = iter_data.get("graphs", [])
        for node in graphs:
            n_name = node.get("node_name", "").lower()
            
            # Using aggregated fields directly
            lat = node.get("llm_network_latency_ms", 0)
            
            if n_name == "planner":
                plan_lat += lat
            elif n_name == "actor":
                act_lat += lat
            elif n_name == "evaluator":
                eval_lat += lat
            elif n_name == "tools":
                act_lat += node.get("tool_execution_time_ms", 0)
                mcp_calls += node.get("tool_call_count", 0)

        results[query_name] = {
            "plan": plan_lat / 1000.0,
            "act": act_lat / 1000.0,
            "evaluate": eval_lat / 1000.0,
            "mcp_calls": mcp_calls,
            "success": success_status
        }
    
    return results

def load_data(files_by_config):
    """Loads and aggregates runs per config based on provided file paths."""
    aggregated = {config: {q: {} for q in QUERIES} for config in CONFIG_ORDER}

    for config in CONFIG_ORDER:
        run_files = files_by_config.get(config, [])
        if not run_files:
            continue
            
        raw_data = {q: {'plan': [], 'act': [], 'evaluate': [], 'mcp': [], 'success': []} for q in QUERIES}
        
        for file in run_files:
            run_metrics = extract_metrics_from_trace(file)
            if not run_metrics:
                continue
            
            for q in QUERIES:
                if q in run_metrics:
                    raw_data[q]['plan'].append(run_metrics[q]['plan'])
                    raw_data[q]['act'].append(run_metrics[q]['act'])
                    raw_data[q]['evaluate'].append(run_metrics[q]['evaluate'])
                    raw_data[q]['mcp'].append(run_metrics[q]['mcp_calls'])
                    raw_data[q]['success'].append(run_metrics[q]['success'])
        
        for q in QUERIES:
            if raw_data[q]['plan']:
                plan_mean = np.mean(raw_data[q]['plan'])
                act_mean = np.mean(raw_data[q]['act'])
                evaluate_mean = np.mean(raw_data[q]['evaluate'])
                mcp_mean = np.mean(raw_data[q]['mcp'])
                
                is_success = all(raw_data[q]['success'])
                dnf = not is_success
                
                aggregated[config][q] = {
                    'plan': plan_mean,
                    'act': act_mean,
                    'evaluate': evaluate_mean,
                    'mcp': mcp_mean,
                    'dnf': dnf
                }
            else:
                aggregated[config][q] = {'dnf': True}
                
    return aggregated

def plot_single_paper(paper_name, paper_data, base_color, output_path):
    """Generates the plot."""
    fig, ax = plt.subplots(figsize=(10, 6)) # Narrowed figsize to match target image aspect ratio
    ax2 = ax.twinx() 
    
    is_log = "log" in paper_name.lower()
    
    # --- Scale limits ---
    MAX_LATENCY = 400
    MAX_MCP = 15 if is_log else 8
    
    # --- Refactored Visual Layout Configuration ---
    bar_width = 0.4
    inter_config_spacing = 0.15
    inter_query_spacing = 0.8
    
    x_positions = []
    x_labels = []
    current_x = 0
    
    mcp_x = []
    mcp_y = []
    exceeds_list = []
    
    # --- Value logging ---
    print(f"\n{'='*80}")
    print(f"LATENCY PLOTTER - {paper_name}")
    print(f"{'='*80}")

    for q_idx, query in enumerate(QUERIES):
        query_start_x = current_x
        query_num = query.replace('Query', '')
            
        for config in CONFIG_ORDER:
            query_metrics = paper_data[config].get(query, {})
            
            x_center = current_x + bar_width / 2
            x_positions.append(x_center)
            x_labels.append(config)
            
            has_data = query_metrics and 'plan' in query_metrics
            dnf = query_metrics.get('dnf', False)
            
            # Draw bars only if we actually have duration metrics
            if has_data:
                p = query_metrics.get('plan', 0)
                a = query_metrics.get('act', 0)
                e = query_metrics.get('evaluate', 0)
                total_lat = p + a + e
                mcp_val = query_metrics.get('mcp', 0)
                
                # Log values
                lat_exceeds = total_lat > MAX_LATENCY
                mcp_exceeds = mcp_val > MAX_MCP
                lat_flag = "[EXCEEDS] " if lat_exceeds else "          "
                mcp_flag = "[EXCEEDS] " if mcp_exceeds else "          "
                print(f"{lat_flag}{paper_name}, Query {query_num}, {config} - max: {MAX_LATENCY}s, value: {total_lat:.2f}s (plan: {p:.2f}, act: {a:.2f}, eval: {e:.2f})")
                print(f"{mcp_flag}{paper_name}, Query {query_num}, {config} - max: {MAX_MCP}, value: {mcp_val:.2f} (mcp_calls)")
                if lat_exceeds:
                    exceeds_list.append(f"  {paper_name}, Query {query_num}, {config} - latency: {total_lat:.2f}s > {MAX_LATENCY}s")
                if mcp_exceeds:
                    exceeds_list.append(f"  {paper_name}, Query {query_num}, {config} - mcp_calls: {mcp_val:.2f} > {MAX_MCP}")
                
                bottom = 0
                if p > 0:
                    ax.bar(x_center, p, width=bar_width, bottom=bottom, color=get_agent_color(base_color, 'plan'), edgecolor='black')
                    bottom += p
                if a > 0:
                    ax.bar(x_center, a, width=bar_width, bottom=bottom, color=get_agent_color(base_color, 'act'), edgecolor='black')
                    bottom += a
                if e > 0:
                    ax.bar(x_center, e, width=bar_width, bottom=bottom, color=get_agent_color(base_color, 'evaluate'), edgecolor='black')
                    bottom += e
                
                if dnf:
                    y_pos = bottom + 5 if bottom > 0 else 10
                    ax.text(x_center, y_pos, 'DNF', color='#E24A33', rotation=90, 
                            ha='center', va='bottom', fontweight='bold', fontsize=12)

                mcp_x.append(x_center)
                mcp_y.append(mcp_val)
            
            # Print DNF if it failed completely without duration data
            elif dnf:
                ax.text(x_center, 10, 'DNF', color='#E24A33', rotation=90, 
                        ha='center', va='bottom', fontweight='bold', fontsize=12)

            # Unconditionally advance the X position to keep structural gaps
            current_x += (bar_width + inter_config_spacing)
        
        query_end_x = current_x - inter_config_spacing
        group_center_x = (query_start_x + query_end_x) / 2
        
        ax.text(group_center_x, 370, query, ha='center', va='center', fontweight='bold', fontsize=14)
        
        if q_idx < len(QUERIES) - 1:
            separator_x = query_end_x + inter_query_spacing / 2
            ax.axvline(x=separator_x, color='lightgray', linestyle='-', linewidth=1)

        current_x += inter_query_spacing
        
    if mcp_x:
        ax2.scatter(mcp_x, mcp_y, marker='D', s=20, color='#A078C4', edgecolor='black', linewidth=0.5, zorder=10)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontweight='bold', fontsize=12)
    ax.set_ylabel('Avg. Time (seconds)', fontweight='bold', fontsize=14)
    ax.set_ylim(0, MAX_LATENCY)
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(axis='y', which='major', linestyle='-', alpha=0.6, color='lightgray', linewidth=1.0)
    ax.grid(axis='y', which='minor', linestyle='--', alpha=0.3, color='lightgray', linewidth=0.5)
    ax.set_axisbelow(True)

    ax2.set_ylabel('Avg. MCP Tool Calls', fontweight='bold', fontsize=14)
    ax2.set_ylim(0, MAX_MCP)
    if is_log:
        ax2.yaxis.set_major_locator(MultipleLocator(5))
        ax2.yaxis.set_minor_locator(MultipleLocator(1))
    else:
        ax2.yaxis.set_major_locator(MultipleLocator(2))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
    
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=get_agent_color(base_color, 'plan'), edgecolor='black', label='Plan (Light)'),
        Rectangle((0, 0), 1, 1, facecolor=get_agent_color(base_color, 'act'), edgecolor='black', label='Act (Medium)'),
        Rectangle((0, 0), 1, 1, facecolor=get_agent_color(base_color, 'evaluate'), edgecolor='black', label='Evaluate (Dark)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#A078C4', markeredgecolor='black', markersize=6, label='MCP Tool Calls')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1.15), ncol=4, framealpha=1, edgecolor='black', fontsize=10)
    
    plt.figtext(0.5, -0.05, f"{paper_name}", ha="center", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate agent performance plots from JSON trace logs.")
    parser.add_argument('--paper', type=str, default="Paper 1", help="Name of the paper (e.g., 'Paper 1')")
    parser.add_argument('--agent_type', type=str, default="react_monolith", help="Agent type (e.g., 'react_monolith')")
    parser.add_argument('--out', type=str, default="", help="Output file path")
    parser.add_argument('--e', nargs='*', default=[], help="Paths to JSON logs for 'E' config")
    parser.add_argument('--n', nargs='*', default=[], help="Paths to JSON logs for 'N' config")
    parser.add_argument('--c', nargs='*', default=[], help="Paths to JSON logs for 'C' config")
    parser.add_argument('--m', nargs='*', default=[], help="Paths to JSON logs for 'M' config")
    parser.add_argument('--mc', nargs='*', default=[], help="Paths to JSON logs for 'MC' config")
    
    args = parser.parse_args()

    files_by_config = {
        'E': args.e,
        'N': args.n,
        'C': args.c,
        'M': args.m,
        'MC': args.mc
    }

    base_color = PAPER_COLORS.get(args.paper.replace(' ', ''), '#4472C4') 
    
    if not args.out:
        safe_title = args.paper.replace(' ', '_')
        args.out = str(Path(__file__).parent / "plots" / f"{safe_title}_{args.agent_type}.pdf")


    paper_data = load_data(files_by_config)
    
    plot_single_paper(args.paper, paper_data, base_color, args.out)

if __name__ == "__main__":
    main()