# utils.py - Utility Functions

import pandas as pd
import json
from datetime import datetime
from typing import Dict, List
from pyvis.network import Network


def save_results(results: Dict, filepath: str, format: str = 'json'):
    """
    Save results to file.
    
    Parameters:
    -----------
    results : Dict
        Results dictionary
    filepath : str
        Path to save file
    format : str
        'json' or 'csv'
    """
    if format == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    elif format == 'csv':
        # Convert to DataFrame if possible
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
    
    print(f"Results saved to {filepath}")


def create_results_summary(results: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame from results.
    
    Parameters:
    -----------
    results : Dict
        Results dictionary from factuality inference
        
    Returns:
    --------
    pd.DataFrame
        Summary dataframe
    """
    summary_data = []
    
    for triple_result in results['triple_results']:
        summary_data.append({
            'hypothesis': triple_result['hypothesis'],
            'verdict': triple_result['final_verdict'],
            'confidence': triple_result['confidence'],
            'explanation': triple_result['explanation'],
            'num_evidence_paths': len(triple_result['evidence_results'])
        })
    
    return pd.DataFrame(summary_data)


def visualize_kg(kg_df: pd.DataFrame, 
                output_path: str = "kg_visualization.html",
                height: str = "750px",
                width: str = "100%"):
    """
    Create an interactive visualization of a knowledge graph.
    
    Parameters:
    -----------
    kg_df : pd.DataFrame
        KG dataframe with columns: subject, predicate, object
    output_path : str
        Path to save HTML visualization
    height, width : str
        Dimensions of visualization
    """
    net = Network(
        notebook=True,
        directed=True,
        height=height,
        width=width,
        cdn_resources="in_line",
    )
    
    # Track added nodes
    added_nodes = set()
    
    def add_node_safe(node_id: str):
        if node_id not in added_nodes:
            net.add_node(
                node_id,
                label=str(node_id),
                title=f"Entity: {node_id}",
                color="lightblue"
            )
            added_nodes.add(node_id)
    
    # Add edges from triples
    for _, row in kg_df.iterrows():
        s = str(row['subject']).strip()
        p = str(row['predicate']).strip()
        o = str(row['object']).strip()
        
        if not s or not p or not o:
            continue
        
        add_node_safe(s)
        add_node_safe(o)
        
        edge_title = f"{s} —[{p}]→ {o}"
        net.add_edge(s, o, label=p, title=edge_title)
    
    net.set_options("""
    var options = {
      "edges": { "arrows": { "to": { "enabled": true } } },
      "physics": { "enabled": true, "barnesHut": { "gravitationalConstant": -30000 } }
    }
    """)
    
    net.save_graph(output_path)
    print(f"KG visualization saved to {output_path}")
    
    return net


def format_evidence_path(path: List[tuple]) -> str:
    """
    Format an evidence path nicely for display.
    
    Parameters:
    -----------
    path : List[tuple]
        List of (subject, predicate, object) tuples
        
    Returns:
    --------
    str
        Formatted path string
    """
    if not path:
        return "(empty path)"
    
    formatted = []
    for i, (s, p, o) in enumerate(path, 1):
        formatted.append(f"{i}. ({s}) —[{p}]→ ({o})")
    
    return "\n".join(formatted)


def calculate_metrics(results_df: pd.DataFrame, 
                     ground_truth_col: str = 'ground_truth',
                     prediction_col: str = 'model_prediction') -> Dict:
    """
    Calculate evaluation metrics if ground truth is available.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with ground truth and predictions
    ground_truth_col : str
        Name of ground truth column
    prediction_col : str
        Name of prediction column
        
    Returns:
    --------
    Dict
        Dictionary of metrics
    """
    # Filter for rows with ground truth
    df = results_df[results_df[ground_truth_col].notna()].copy()
    
    if len(df) == 0:
        return {'error': 'No ground truth labels available'}
    
    # Map labels
    label_map = {
        'T': 'TRUE NEWS',
        'F': 'FAKE NEWS',
        'TRUE NEWS': 'TRUE NEWS',
        'FAKE NEWS': 'FAKE NEWS'
    }
    
    df['gt_mapped'] = df[ground_truth_col].map(label_map)
    df['pred_mapped'] = df[prediction_col]
    
    # Calculate metrics
    correct = (df['gt_mapped'] == df['pred_mapped']).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0
    
    # Confusion matrix
    true_pos = ((df['gt_mapped'] == 'TRUE NEWS') & (df['pred_mapped'] == 'TRUE NEWS')).sum()
    true_neg = ((df['gt_mapped'] == 'FAKE NEWS') & (df['pred_mapped'] == 'FAKE NEWS')).sum()
    false_pos = ((df['gt_mapped'] == 'FAKE NEWS') & (df['pred_mapped'] == 'TRUE NEWS')).sum()
    false_neg = ((df['gt_mapped'] == 'TRUE NEWS') & (df['pred_mapped'] == 'FAKE NEWS')).sum()
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(true_pos),
        'true_negatives': int(true_neg),
        'false_positives': int(false_pos),
        'false_negatives': int(false_neg),
        'total': int(total)
    }


def export_detailed_report(results: Dict, 
                          filepath: str = "detailed_report.html"):
    """
    Export detailed HTML report of results.
    
    Parameters:
    -----------
    results : Dict
        Results from factuality inference
    filepath : str
        Output path for HTML report
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fake News Detection Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            h1 {{ color: #1E3A8A; border-bottom: 3px solid #3B82F6; padding-bottom: 10px; }}
            h2 {{ color: #3B82F6; margin-top: 30px; }}
            .verdict {{ padding: 20px; border-radius: 8px; margin: 20px 0; font-size: 1.2em; }}
            .verdict-true {{ background: #D1FAE5; color: #065F46; border-left: 5px solid #10B981; }}
            .verdict-fake {{ background: #FEE2E2; color: #991B1B; border-left: 5px solid #EF4444; }}
            .verdict-unverified {{ background: #FEF3C7; color: #92400E; border-left: 5px solid #F59E0B; }}
            .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 10px 20px; background: #EFF6FF; border-radius: 5px; }}
            .metric-label {{ font-size: 0.9em; color: #6B7280; }}
            .metric-value {{ font-size: 1.5em; font-weight: bold; color: #1E3A8A; }}
            .triple {{ background: #F9FAFB; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 3px solid #3B82F6; }}
            .evidence {{ background: #FFFBEB; padding: 10px; margin: 5px 0; border-radius: 3px; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Fake News Detection Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="verdict verdict-{results['overall_verdict'].lower().replace(' ', '-')}">
                <strong>Overall Verdict:</strong> {results['overall_verdict']}<br>
                <strong>Explanation:</strong> {results['overall_explanation']}
            </div>
            
            <h2>Summary Metrics</h2>
            <div class="metric">
                <div class="metric-label">Total Claims</div>
                <div class="metric-value">{results['summary']['total_triples']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Supported</div>
                <div class="metric-value">{results['summary']['supported']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Refuted</div>
                <div class="metric-value">{results['summary']['refuted']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Confidence</div>
                <div class="metric-value">{results['summary']['avg_confidence']:.2%}</div>
            </div>
            
            <h2>Detailed Triple Analysis</h2>
    """
    
    for i, triple_result in enumerate(results['triple_results'], 1):
        verdict_emoji = {
            'SUPPORTED': '✅',
            'REFUTED': '❌',
            'NOT_ENOUGH_INFO': '❓'
        }.get(triple_result['final_verdict'], '❓')
        
        html += f"""
            <div class="triple">
                <h3>Claim {i}: {triple_result['hypothesis']}</h3>
                <p><strong>{verdict_emoji} Verdict:</strong> {triple_result['final_verdict']} 
                   (Confidence: {triple_result['confidence']:.2%})</p>
                <p><strong>Explanation:</strong> {triple_result['explanation']}</p>
        """
        
        if triple_result['evidence_results']:
            html += "<h4>Evidence:</h4>"
            for j, evidence in enumerate(triple_result['evidence_results'], 1):
                html += f"""
                <div class="evidence">
                    <strong>Path {j}:</strong> {evidence['premise'][:200]}...<br>
                    <strong>Verdict:</strong> {evidence['verdict']} 
                    (Confidence: {evidence['confidence']:.2%})
                </div>
                """
        
        html += "</div>"
    
    html += """
        </div>
    </body>
    </html>
    """
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Detailed report saved to {filepath}")
