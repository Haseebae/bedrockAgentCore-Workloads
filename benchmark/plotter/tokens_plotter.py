import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pathlib import Path

# --- Configuration ---
CONFIG_ORDER = ['E', 'N', 'C', 'M', 'MC']
QUERIES = ['Query1', 'Query2', 'Query3']

# Color mapping matched to the original image's look
COLORS = {
    'input': {
        'plan': '#8DA0CB',      # Light Blue
        'act': '#4472C4',       # Medium Blue
        'evaluate': '#1F497D'   # Dark Blue
    },
    'output': {
        'plan': '#B3CDE3',      # Light Navy/Grey
        'act': '#315B7E',       # Navy
        'evaluate': '#151515'   # Very Dark Navy/Black
    }
}

def calculate_cost_cents(input_tokens, output_tokens, cached_tokens):
    """Calculate cost in cents based on gpt-4o-mini pricing."""
    input_cost = (input_tokens / 1_000_000) * 0.15
    cached_cost = (cached_tokens / 1_000_000) * 0.075
    output_cost = (output_tokens / 1_000_000) * 0.60
    total_dollars = input_cost + cached_cost + output_cost
    return total_dollars * 100  # Convert to cents

def extract_tokens_from_trace(filepath):
    """Extracts input and output tokens from the session trace JSON."""
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
        tokens = {
            'plan': {'in': 0, 'out': 0, 'cache': 0},
            'act': {'in': 0, 'out': 0, 'cache': 0},
            'evaluate': {'in': 0, 'out': 0, 'cache': 0}
        }
        
        for node in iter_data.get("graphs", []):
            n_name = node.get("node_name", "").lower()
            
            i_tok = node.get("llm_input_tokens", 0)
            o_tok = node.get("llm_output_tokens", 0)
            c_tok = node.get("llm_cached_tokens", 0)
            
            if n_name == "planner":
                tokens['plan']['in'] += i_tok
                tokens['plan']['out'] += o_tok
                tokens['plan']['cache'] += c_tok
            elif n_name == "actor":
                tokens['act']['in'] += i_tok
                tokens['act']['out'] += o_tok
                tokens['act']['cache'] += c_tok
            elif n_name == "evaluator":
                tokens['evaluate']['in'] += i_tok
                tokens['evaluate']['out'] += o_tok
                tokens['evaluate']['cache'] += c_tok
        
        tokens['success'] = success_status
        results[query_name] = tokens
    return results

def load_data(files_by_config):
    """Loads and aggregates run token metrics per config."""
    aggregated = {config: {q: {} for q in QUERIES} for config in CONFIG_ORDER}

    for config in CONFIG_ORDER:
        run_files = files_by_config.get(config, [])
        if not run_files:
            continue
            
        raw_data = {q: {'plan': [], 'act': [], 'evaluate': [], 'success': []} for q in QUERIES}
        
        for file in run_files:
            run_metrics = extract_tokens_from_trace(file)
            if not run_metrics:
                continue
            
            for q in QUERIES:
                if q in run_metrics:
                    raw_data[q]['plan'].append(run_metrics[q]['plan'])
                    raw_data[q]['act'].append(run_metrics[q]['act'])
                    raw_data[q]['evaluate'].append(run_metrics[q]['evaluate'])
                    raw_data[q]['success'].append(run_metrics[q]['success'])
        
        for q in QUERIES:
            if raw_data[q]['plan']:
                mean_tokens = {}
                total_in = 0
                total_out = 0
                total_cache = 0
                
                for agent in ['plan', 'act', 'evaluate']:
                    agent_in = np.mean([x['in'] for x in raw_data[q][agent]])
                    agent_out = np.mean([x['out'] for x in raw_data[q][agent]])
                    agent_cache = np.mean([x['cache'] for x in raw_data[q][agent]])
                    
                    mean_tokens[agent] = {
                        'in': agent_in,
                        'out': agent_out,
                        'cache': agent_cache
                    }
                    total_in += agent_in
                    total_out += agent_out
                    total_cache += agent_cache
                
                cost_cents = calculate_cost_cents(total_in, total_out, total_cache)
                
                is_success = all(raw_data[q]['success'])
                dnf = not is_success
                
                aggregated[config][q] = mean_tokens
                aggregated[config][q]['total_in'] = total_in
                aggregated[config][q]['total_out'] = total_out
                aggregated[config][q]['cost_cents'] = cost_cents
                aggregated[config][q]['dnf'] = dnf
            else:
                aggregated[config][q] = {'dnf': True}
                
    return aggregated

def format_k(x, pos):
    if x >= 1000:
        return f'{int(x/1000)}k'
    return f'{int(x)}'

def format_cents(x, pos):
    return f'{x:.2f}'

def plot_token_data(paper_name, paper_data, output_path):
    """Generates the dual-axis stacked token bar chart with a cost axis."""
    is_log = "log" in paper_name.lower()

    fig, main_ax = plt.subplots(figsize=(14, 7))
    ax2 = main_ax.twinx()
    ax3 = main_ax.twinx()
    ax3.spines['right'].set_position(('outward', 75))
    
    axes_input = [main_ax]
    
    if is_log:
        MAX_INPUT_TOKENS = 210000
        MAX_OUTPUT_TOKENS = 16000
        MAX_COST = 5.00
    else:
        MAX_INPUT_TOKENS = 150000
        MAX_OUTPUT_TOKENS = 16000
        MAX_COST = 5.00

    # --- Refactored Visual Layout Configuration ---
    bar_width = 0.25
    intra_config_spacing = 0.02   # Gap between Input and Output bars of the same config
    inter_config_spacing = 0.15   # Gap between different configs (e.g., E to N)
    inter_query_spacing = 0.8     # Gap between Query sections
    
    x_labels = []
    x_positions_center = []
    current_x = 0
    
    # --- Value logging ---
    exceeds_list = []
    print(f"\n{'='*80}")
    print(f"TOKENS PLOTTER - {paper_name}")
    print(f"{'='*80}")
    
    for q_idx, query in enumerate(QUERIES):
        query_start_x = current_x
        query_num = query.replace('Query', '')
        
        for config in CONFIG_ORDER:
            query_metrics = paper_data[config].get(query, {})
            
            # Calculate coordinates unconditionally to preserve spacing
            x_input = current_x + bar_width / 2
            x_output = current_x + bar_width + intra_config_spacing + bar_width / 2
            x_center = (x_input + x_output) / 2
            
            x_labels.append(config)
            x_positions_center.append(x_center)
            
            # Only plot if metrics actually exist
            if query_metrics and not (len(query_metrics) == 1 and query_metrics.get('dnf') == True and paper_data[config].get(query, {}).get('total_in') is None):
                dnf = query_metrics.get('dnf', False)
                
                # Calculate totals for logging
                total_in_tokens = sum(query_metrics.get(agent, {}).get('in', 0) for agent in ['plan', 'act', 'evaluate'])
                total_out_tokens = sum(query_metrics.get(agent, {}).get('out', 0) for agent in ['plan', 'act', 'evaluate'])
                cost_cents = query_metrics.get('cost_cents', 0)
                
                # Log values
                in_exceeds = total_in_tokens > MAX_INPUT_TOKENS
                out_exceeds = total_out_tokens > MAX_OUTPUT_TOKENS
                cost_exceeds = cost_cents > MAX_COST
                in_flag = "[EXCEEDS] " if in_exceeds else "          "
                out_flag = "[EXCEEDS] " if out_exceeds else "          "
                cost_flag = "[EXCEEDS] " if cost_exceeds else "          "
                print(f"{in_flag}{paper_name}, Query {query_num}, {config} - max: {MAX_INPUT_TOKENS}, value: {total_in_tokens:.0f} (input_tokens)")
                print(f"{out_flag}{paper_name}, Query {query_num}, {config} - max: {MAX_OUTPUT_TOKENS}, value: {total_out_tokens:.0f} (output_tokens)")
                print(f"{cost_flag}{paper_name}, Query {query_num}, {config} - max: {MAX_COST}¢, value: {cost_cents:.4f}¢ (llm_cost)")
                if in_exceeds:
                    exceeds_list.append(f"  {paper_name}, Query {query_num}, {config} - input_tokens: {total_in_tokens:.0f} > {MAX_INPUT_TOKENS}")
                if out_exceeds:
                    exceeds_list.append(f"  {paper_name}, Query {query_num}, {config} - output_tokens: {total_out_tokens:.0f} > {MAX_OUTPUT_TOKENS}")
                if cost_exceeds:
                    exceeds_list.append(f"  {paper_name}, Query {query_num}, {config} - llm_cost: {cost_cents:.4f}¢ > {MAX_COST}¢")
                    
                # Plot Stacked Input Tokens
                inp_bottom = 0
                for agent in ['plan', 'act', 'evaluate']:
                    val = query_metrics.get(agent, {}).get('in', 0)
                    if val > 0:
                        for ax_i in axes_input:
                            ax_i.bar(x_input, val, width=bar_width, bottom=inp_bottom, 
                                     color=COLORS['input'][agent], edgecolor='black')
                        inp_bottom += val
                
                # Plot Stacked Output Tokens
                out_bottom = 0
                for agent in ['plan', 'act', 'evaluate']:
                    val = query_metrics.get(agent, {}).get('out', 0)
                    if val > 0:
                        ax2.bar(x_output, val, width=bar_width, bottom=out_bottom, 
                                color=COLORS['output'][agent], edgecolor='black')
                        out_bottom += val
                        
                # Plot Cost marker
                cost_cents = query_metrics.get('cost_cents', 0)
                if cost_cents > 0:
                    ax3.scatter([x_center], [cost_cents], marker='D', s=60, color='#D4B483', 
                                edgecolor='black', linewidth=1, zorder=10)

                if dnf:
                    y_pos = inp_bottom + 4000 if inp_bottom > 0 else 10000
                    target_ax = main_ax
                    target_ax.text(x_center, y_pos, 'DNF', color='#E24A33', rotation=90, 
                            ha='center', va='bottom', fontweight='bold', fontsize=18)

            # Advance x position regardless of missing data
            current_x += (bar_width * 2 + intra_config_spacing + inter_config_spacing)
        
        # Calculate the true center of the query block
        query_end_x = current_x - inter_config_spacing
        group_center_x = (query_start_x + query_end_x) / 2
        
        # Place Query label correctly centered
        query_y = 205000 if is_log else 75000
        main_ax.text(group_center_x, query_y, query, ha='center', va='center', fontweight='bold', fontsize=18)
        
        # Add vertical divider midway through the inter_query_spacing
        if q_idx < len(QUERIES) - 1:
            separator_x = query_end_x + inter_query_spacing / 2
            for ax_i in axes_input:
                ax_i.axvline(x=separator_x, color='lightgray', linestyle='-', linewidth=1)

        current_x += inter_query_spacing

    # Axes Setup
    main_ax.set_ylim(0, MAX_INPUT_TOKENS)
    if is_log:
        main_ax.yaxis.set_major_locator(MultipleLocator(50000))
        main_ax.yaxis.set_minor_locator(MultipleLocator(10000))
    else:
        main_ax.yaxis.set_major_locator(MultipleLocator(30000))
        main_ax.yaxis.set_minor_locator(MultipleLocator(10000))
    main_ax.set_ylabel('Avg. Input Tokens', fontweight='bold', fontsize=16)

    ax2.set_ylim(0, MAX_OUTPUT_TOKENS)
    if is_log:
        ax2.yaxis.set_major_locator(MultipleLocator(4000))
        ax2.yaxis.set_minor_locator(MultipleLocator(1000))
    else:
        ax2.yaxis.set_major_locator(MultipleLocator(4000))
        ax2.yaxis.set_minor_locator(MultipleLocator(1000))
    ax2.set_ylabel('Avg. Output Tokens', fontweight='bold', fontsize=16)

    ax3.set_ylim(0, MAX_COST)
    ax3.yaxis.set_major_locator(MultipleLocator(0.50))
    ax3.set_ylabel('LLM Cost (cents)', fontweight='bold', fontsize=16)

    # Formatters
    k_formatter = FuncFormatter(format_k)
    for ax_i in axes_input:
        ax_i.yaxis.set_major_formatter(k_formatter)
    ax2.yaxis.set_major_formatter(k_formatter)
    ax3.yaxis.set_major_formatter(FuncFormatter(format_cents))

    # Parameters & Grid
    for ax_i in axes_input:
        ax_i.tick_params(axis='both', which='major', labelsize=14)
        ax_i.grid(axis='y', which='major', linestyle='-', alpha=0.5, color='gray')
        ax_i.grid(axis='y', which='minor', linestyle='--', alpha=0.2, color='gray')
        ax_i.set_axisbelow(True)
    
    ax2.tick_params(axis='y', which='major', labelsize=14)
    ax3.tick_params(axis='y', which='major', labelsize=14)
    ax2.set_axisbelow(True)

    # X-axis Labels
    main_ax.set_xticks(x_positions_center)
    main_ax.set_xticklabels(x_labels, fontweight='bold', fontsize=14)

    # Custom Legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS['input']['plan'], edgecolor='black', label='Input: Planner'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['output']['plan'], edgecolor='black', label='Output: Planner'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['input']['act'], edgecolor='black', label='Input: Actor'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['output']['act'], edgecolor='black', label='Output: Actor'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['input']['evaluate'], edgecolor='black', label='Input: Evaluator'),
        Rectangle((0, 0), 1, 1, facecolor=COLORS['output']['evaluate'], edgecolor='black', label='Output: Evaluator'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#D4B483', markeredgecolor='black', markersize=8, label='Cost'),
    ]
    axes_input[0].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
              ncol=4, framealpha=1, edgecolor='black', fontsize=12)
    
    plt.figtext(0.5, -0.05, f"{paper_name}", ha="center", fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate token performance plots from JSON trace logs.")
    parser.add_argument('--paper', type=str, default="a", help="Sub-label for paper (e.g. 'a')")
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

    if not args.out:
        safe_title = args.paper.replace(' ', '_')
        args.out = str(Path(__file__).parent / "plots" / f"{safe_title}_{args.agent_type}.pdf")


    paper_data = load_data(files_by_config)
    
    plot_token_data(args.paper, paper_data, args.out)

if __name__ == "__main__":
    main()