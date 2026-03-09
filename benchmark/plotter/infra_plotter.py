import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from pathlib import Path

# --- Configuration ---
CONFIG_ORDER = ['E', 'N', 'C', 'M', 'MC']
QUERIES = ['Query1', 'Query2', 'Query3']

# Base colors
C_LLM = '#5A8DCA'        # Blue
C_AGENT = '#F5B757'      # Orange / Yellow
C_MCP = '#89C765'        # Green (Matched to provided image's MCP color)

def calculate_llm_cost(input_tokens, output_tokens, cached_tokens):
    """Calculate LLM cost in cents based on gpt-4o-mini pricing."""
    input_cost = (input_tokens / 1_000_000) * 0.15
    cached_cost = (cached_tokens / 1_000_000) * 0.075
    output_cost = (output_tokens / 1_000_000) * 0.60
    total_cost = input_cost + cached_cost + output_cost
    return total_cost * 100

def calculate_agent_faas_cost(num_invocations, time_s):
    """Calculate Amazon Bedrock AgentCore Runtime execution cost in cents."""
    time_h = time_s / 3600.0
    
    # Active Consumption Based calculation
    cpu_cost = 1.0 * time_h * 0.0895
    mem_cost = 0.5 * time_h * 0.00945
    
    return (cpu_cost + mem_cost) * 100

def calculate_mcp_cost(num_mcp_calls, time_s):
    """Calculate MCP Lambda execution cost in cents."""
    cost_part1 = num_mcp_calls * 0.000025
    cost_part2 = 0.5 * time_s * 0.0000166
    return (cost_part1 + cost_part2) * 100

def extract_costs_from_trace(filepath):
    """Extracts tokens, latencies, and success status from trace format to compute costs."""
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
        
        in_tok = out_tok = cached_tok = 0
        agent_time_ms = 0
        agent_calls = 1 # Placeholder
        
        mcp_time_ms = 0
        mcp_calls = 1 # Placeholder
        
        for node in iter_data.get("graphs", []):
            node_name = node.get("node_name", "").lower()
            
            if node_name in ["planner", "actor", "evaluator"]:
                in_tok += node.get("llm_input_tokens", 0)
                out_tok += node.get("llm_output_tokens", 0)
                cached_tok += node.get("llm_cached_tokens", 0)
                agent_time_ms += node.get("llm_network_latency_ms", 0)
                
            elif node_name == "tools":
                mcp_time_ms += node.get("tool_execution_time_ms", 0)
        
        # Calculate resulting costs inside the trace iteration
        llm_cost = calculate_llm_cost(in_tok, out_tok, cached_tok)
        
        # Use provided pricing details if available, otherwise fallback
        pricing_details = iter_data.get("pricing_details", {})
        if pricing_details and "total_cents" in pricing_details:
            agent_cost = pricing_details["total_cents"]
        else:
            agent_cost = calculate_agent_faas_cost(agent_calls, agent_time_ms / 1000.0)
            
        mcp_cost = calculate_mcp_cost(mcp_calls, mcp_time_ms / 1000.0)
        
        results[query_name] = {
            "llm": llm_cost,
            "agent": agent_cost,
            "mcp": mcp_cost,
            "total": llm_cost + agent_cost + mcp_cost,
            "success": success_status
        }
        
    return results

def load_data(files_by_config):
    """Loads and aggregates run data matching configurations."""
    aggregated = {config: {q: {} for q in QUERIES} for config in CONFIG_ORDER}

    for config in CONFIG_ORDER:
        run_files = files_by_config.get(config, [])
        if not run_files:
            continue
            
        raw_data = {q: {'llm': [], 'agent': [], 'mcp': [], 'success': []} for q in QUERIES}
        
        for file in run_files:
            run_metrics = extract_costs_from_trace(file)
            if not run_metrics:
                continue
            
            for q in QUERIES:
                if q in run_metrics:
                    raw_data[q]['llm'].append(run_metrics[q]['llm'])
                    raw_data[q]['agent'].append(run_metrics[q]['agent'])
                    raw_data[q]['mcp'].append(run_metrics[q]['mcp'])
                    raw_data[q]['success'].append(run_metrics[q]['success'])
        
        for q in QUERIES:
            if raw_data[q]['llm']:
                llm_mean = np.mean(raw_data[q]['llm'])
                agent_mean = np.mean(raw_data[q]['agent'])
                mcp_mean = np.mean(raw_data[q]['mcp'])
                
                # Determine DNF if ANY of the aggregated runs failed
                is_success = all(raw_data[q]['success'])
                dnf = not is_success
                
                aggregated[config][q] = {
                    'llm': llm_mean,
                    'agent': agent_mean,
                    'mcp': mcp_mean,
                    'dnf': dnf
                }
            else:
                aggregated[config][q] = {} # Empty dict implies no data
                
    return aggregated

def plot_single_paper(paper_name, paper_data, output_path):
    """Generates the 3-panel plot replicating the image layout."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    plt.subplots_adjust(wspace=0) 
    
    bar_width = 0.6
    x_positions = np.arange(len(CONFIG_ORDER))
    
    is_log = "log" in paper_name.lower()
    max_y = 5.0
    
    # --- Value logging ---
    exceeds_list = []
    print(f"\n{'='*80}")
    print(f"INFRA COST PLOTTER - {paper_name}")
    print(f"{'='*80}")
    
    for i, query in enumerate(QUERIES):
        ax = axes[i]
        query_num = query.replace('Query', '')
        
        ax.set_ylim(0, max_y)
        ax.set_xlim(-0.5, len(CONFIG_ORDER) - 0.5)
        
        # Adding Grids and Ticks
        if is_log:
            ax.yaxis.set_major_locator(MultipleLocator(1.25))
            ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(1.0))
            ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax.grid(axis='y', which='major', linestyle='-', alpha=0.5, color='gray')
        ax.grid(axis='y', which='minor', linestyle=':', alpha=0.3, color='gray')
        ax.set_axisbelow(True)
        
        # Query Titles - centered over the bar group
        group_center = (x_positions[0] + x_positions[-1]) / 2
        query_y = 4.8
        ax.text(group_center, query_y, query, ha='center', va='top', fontweight='bold', fontsize=18)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(CONFIG_ORDER, fontweight='bold', fontsize=16)
        
        # Add labels & Legend on the primary axis (First Panel)
        if i == 0:
            ax.set_ylabel('Avg. Cost (cents)', fontweight='bold', fontsize=18)
            legend_elements = [
                Line2D([0], [0], color=C_LLM, lw=8, label='LLM Cost'),
                Line2D([0], [0], color=C_AGENT, lw=8, label='Bedrock AgentCore Pricing'),
                Line2D([0], [0], color=C_MCP, lw=8, label='MCP Lambda Cost')
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.85), 
                      fontsize=13, framealpha=1, edgecolor='gray')

        # Stack the Bars 
        for j, config in enumerate(CONFIG_ORDER):
            query_metrics = paper_data[config].get(query, {})
            
            # Skip empty configs
            if not query_metrics:
                continue
            
            llm = query_metrics.get('llm', 0)
            agent = query_metrics.get('agent', 0)
            mcp = query_metrics.get('mcp', 0)
            total = llm + agent + mcp
            dnf = query_metrics.get('dnf', False)
            
            # Log values
            cost_exceeds = total > max_y
            flag = "[EXCEEDS] " if cost_exceeds else "          "
            print(f"{flag}{paper_name}, Query {query_num}, {config} - max: {max_y}¢, value: {total:.4f}¢ (llm: {llm:.4f}, agent: {agent:.4f}, mcp: {mcp:.4f})")
            if cost_exceeds:
                exceeds_list.append(f"  {paper_name}, Query {query_num}, {config} - cost: {total:.4f}¢ > {max_y}¢")
            
            # Bottom: LLM
            ax.bar(j, llm, width=bar_width, color=C_LLM, edgecolor='black', linewidth=1)
            # Middle: AgentCore
            ax.bar(j, agent, width=bar_width, bottom=llm, color=C_AGENT, edgecolor='black', linewidth=1)
            # Top: MCP Lambda
            ax.bar(j, mcp, width=bar_width, bottom=(llm+agent), color=C_MCP, edgecolor='black', linewidth=1)
            
            # Put DNF tag slightly above the stacked bar based on actual cost incurred
            if dnf:
                total_h = llm + agent + mcp
                ax.text(j, total_h + 0.05, 'DNF', color='#E13E28', rotation=90, 
                        ha='center', va='bottom', fontweight='bold', fontsize=18)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)

    plt.figtext(0.5, -0.05, f"{paper_name}", 
                ha="center", fontsize=20, fontweight='bold', fontfamily='serif')
    
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cost plot saved successfully to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate cost-stacked plots from JSON trace logs.")
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
        'E': args.e, 'N': args.n, 'C': args.c, 'M': args.m, 'MC': args.mc
    }

    if not args.out:
        safe_title = args.paper.replace(' ', '_')
        args.out = str(Path(__file__).parent / "plots" / f"{safe_title}_{args.agent_type}.pdf")


    paper_data = load_data(files_by_config)
    
    plot_single_paper(args.paper, paper_data, args.out)

if __name__ == "__main__":
    main()