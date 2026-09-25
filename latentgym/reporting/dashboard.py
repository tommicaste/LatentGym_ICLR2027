"""
Interactive HTML dashboard — single self-contained file with all tables,
charts, and leaderboard from a DataStore output directory.

Tabs: Leaderboard, Results Table, Per-Environment, Complexity, Learning Curves,
      Double-Agent (if present). Filter dropdowns for model/env/metric.
No server required — pure HTML + JS.

Usage:
    from latentgym.reporting.dashboard import render_dashboard, render_dashboard_from_dir

    # From pre-loaded data
    data = DataStore.load("results/run_001")
    html = render_dashboard(data, title="Run 001 Results")

    # Directly from directory
    html = render_dashboard_from_dir("results/run_001")

    with open("dashboard.html", "w") as f:
        f.write(html)
"""
from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, List, Optional

from .data_store import DataStore


def render_dashboard_from_dir(
    data_dir: str,
    title: str = "Benchmark Dashboard",
) -> str:
    """Load DataStore and render interactive dashboard HTML."""
    data = DataStore.load(data_dir)
    return render_dashboard(data, title=title)


def render_dashboard(
    data: Dict[str, Any],
    title: str = "Benchmark Dashboard",
) -> str:
    """Render an interactive HTML dashboard from DataStore.load() output.

    Generates a single standalone HTML file containing:
        - Leaderboard table (sortable)
        - Results table (model × config, filterable by env)
        - Per-environment summary table
        - Per-complexity breakdown table
        - Learning curve charts (one per config, as embedded PNGs)
        - Double-agent tables + charts (if data present)

    All filtering is client-side via JavaScript.

    Args:
        data: Output of DataStore.load(data_dir)
        title: Page title

    Returns:
        Standalone HTML string
    """
    metrics = data.get("metrics", {})
    metadata = data.get("metadata", {})
    per_model = metrics.get("per_model", {})
    per_env = metrics.get("per_env", {})
    per_complexity = metrics.get("per_latent_complexity", {})
    leaderboard = metrics.get("leaderboard", [])
    detailed = metrics.get("detailed", {}).get("exploration_exploitation", {})
    da_schedule = metrics.get("double_agent", {}).get("per_schedule", {})
    da_agent = metrics.get("double_agent", {}).get("per_agent", {})
    da_comparisons = metrics.get("double_agent", {}).get("comparisons", {})

    has_double = bool(da_schedule)

    # Collect filter values
    models = sorted(per_model.keys())
    envs = sorted(per_env.keys())
    complexities = sorted(per_complexity.keys())
    all_bids = sorted({bid for bm in per_model.values() for bid in bm})

    # Generate chart images as base64
    charts_b64 = _generate_charts_b64(per_model, per_env, da_schedule, all_bids, envs)

    # Build tab content
    leaderboard_html = _build_leaderboard_tab(leaderboard)
    results_html = _build_results_tab(per_model, models, all_bids, envs)
    per_env_html = _build_per_env_tab(per_env, models, envs)
    complexity_html = _build_complexity_tab(per_complexity, models, complexities)
    curves_html = _build_learning_curves_tab(charts_b64, all_bids, per_model)
    ep_compare_html, var_compare_html = _build_compare_tabs(per_model, models, all_bids, envs)
    double_html = _build_double_agent_tab(da_schedule, da_agent, da_comparisons, charts_b64) if has_double else ""

    # Tab list
    tabs = [
        ("leaderboard", "Leaderboard"),
        ("results", "Results Table"),
        ("per-env", "Per-Environment"),
        ("complexity", "By Complexity"),
        ("curves", "Learning Curves"),
        ("ep-compare", "Episode Compare"),
        ("var-compare", "Variable Compare"),
    ]
    if has_double:
        tabs.append(("double", "Double-Agent"))

    tab_buttons = "\n".join(
        f'<button class="tab-btn{" active" if i==0 else ""}" onclick="switchTab(\'{tid}\')">{tlabel}</button>'
        for i, (tid, tlabel) in enumerate(tabs)
    )

    tab_panels = f"""
    <div id="tab-leaderboard" class="tab-panel active">{leaderboard_html}</div>
    <div id="tab-results" class="tab-panel">{results_html}</div>
    <div id="tab-per-env" class="tab-panel">{per_env_html}</div>
    <div id="tab-complexity" class="tab-panel">{complexity_html}</div>
    <div id="tab-curves" class="tab-panel">{curves_html}</div>
    <div id="tab-ep-compare" class="tab-panel">{ep_compare_html}</div>
    <div id="tab-var-compare" class="tab-panel">{var_compare_html}</div>
    """
    if has_double:
        tab_panels += f'<div id="tab-double" class="tab-panel">{double_html}</div>'

    run_name = metadata.get("run_name", "")
    created = metadata.get("created_at", "")[:19]
    n_models = len(models)
    n_configs = len(all_bids)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{_esc(title)}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 0; padding: 0; background: #f5f5f5; }}
  .top-bar {{ background: #263238; color: white; padding: 14px 24px; }}
  .top-bar h1 {{ margin: 0; font-size: 1.3em; }}
  .top-bar .sub {{ font-size: 0.85em; opacity: 0.7; margin-top: 4px; }}
  .tabs {{ background: #37474F; display: flex; padding: 0 24px; gap: 0; }}
  .tab-btn {{ background: none; border: none; color: rgba(255,255,255,0.7); padding: 10px 18px;
              font-size: 0.9em; cursor: pointer; border-bottom: 3px solid transparent; }}
  .tab-btn:hover {{ color: white; background: rgba(255,255,255,0.05); }}
  .tab-btn.active {{ color: white; border-bottom-color: #4FC3F7; }}
  .content {{ max-width: 1200px; margin: 0 auto; padding: 20px 24px; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.88em; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  th {{ background: #f0f0f0; cursor: pointer; white-space: nowrap; }}
  th:hover {{ background: #e0e0e0; }}
  tr:hover {{ background: #f9f9f9; }}
  .card {{ background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .card h3 {{ margin: 0 0 10px 0; font-size: 1.05em; }}
  .metric-row {{ display: flex; gap: 16px; margin-bottom: 16px; }}
  .metric-card {{ background: white; border-radius: 8px; padding: 12px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; flex: 1; }}
  .metric-card .val {{ font-size: 1.6em; font-weight: bold; color: #263238; }}
  .metric-card .lbl {{ font-size: 0.8em; color: #888; }}
  .chart-img {{ max-width: 100%; border-radius: 6px; margin: 8px 0; }}
  .filter-bar {{ margin-bottom: 12px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .filter-bar label {{ font-size: 0.85em; color: #555; }}
  .filter-bar select {{ padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 0.88em; }}
  .success {{ background-color: #e8f5e9; }}
  .failure {{ background-color: #ffebee; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 0.88em; }}
  .rank-1 {{ font-weight: bold; color: #F9A825; }}
  .rank-2 {{ font-weight: bold; color: #90A4AE; }}
  .rank-3 {{ font-weight: bold; color: #8D6E63; }}
  details {{ margin: 4px 0; }}
  summary {{ cursor: pointer; font-weight: 500; }}
</style>
</head>
<body>

<div class="top-bar">
  <h1>{_esc(title)}</h1>
  <div class="sub">{_esc(run_name)} &nbsp;|&nbsp; {_esc(created)} &nbsp;|&nbsp; {n_models} models &times; {n_configs} configs</div>
</div>

<div class="tabs">{tab_buttons}</div>

<div class="content">
{tab_panels}
</div>

<script>
function switchTab(id) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  event.target.classList.add('active');
}}

function sortTable(tableId, col) {{
  const table = document.getElementById(tableId);
  const tbody = table.querySelector('tbody') || table;
  const rows = Array.from(tbody.querySelectorAll('tr')).filter(r => r.querySelector('td'));
  const dir = table.dataset.sortDir === 'asc' ? 'desc' : 'asc';
  table.dataset.sortDir = dir;
  rows.sort((a, b) => {{
    let va = a.children[col]?.textContent?.trim() || '';
    let vb = b.children[col]?.textContent?.trim() || '';
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return dir === 'asc' ? na - nb : nb - na;
    return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

function filterTable(tableId, col, value) {{
  const table = document.getElementById(tableId);
  const rows = table.querySelectorAll('tbody tr, tr');
  rows.forEach(r => {{
    if (!r.querySelector('td')) return;
    const cell = r.children[col]?.textContent?.trim() || '';
    r.style.display = (value === 'all' || cell.startsWith(value)) ? '' : 'none';
  }});
}}

// Shared chart utility functions (used by Learning Curves, Episode Compare, Variable Compare)
var C=['#1976D2','#D32F2F','#388E3C','#F57C00','#7B1FA2','#00796B','#C2185B','#455A64'];
function sm(a,w){{if(w<=0)return a;var o=[];for(var i=0;i<a.length;i++){{var s=0,c=0;for(var j=Math.max(0,i-w);j<=Math.min(a.length-1,i+w);j++){{s+=a[j];c++;}}o.push(s/c);}}return o;}}
function rng(g,ymin_id,ymax_id){{var mn=1e9,mx=.01;for(var k in g)for(var x in g[k]){{var arr=g[k][x],v=0;for(var j=0;j<arr.length;j++)v+=arr[j];v/=arr.length;if(v>mx)mx=v;if(v<mn)mn=v;}}if(mn>1e8)mn=0;var yminEl=document.getElementById(ymin_id),ymaxEl=document.getElementById(ymax_id);var yminV=yminEl?yminEl.value:'',ymaxV=ymaxEl?ymaxEl.value:'';var lo=(yminV!=='')?parseFloat(yminV):Math.max(0,mn-(mx-mn)*.1);var hi=(ymaxV!=='')?parseFloat(ymaxV):mx+(mx-mn)*.1;if(hi<=lo)hi=lo+.01;return[lo,hi];}}
function leg(gk){{if(gk.length<=1&&gk[0]==='_a')return'';var h='<div style="margin-top:12px;padding:8px 14px;background:#f8f9fa;border-radius:6px;border:1px solid #eee;display:inline-flex;gap:16px;flex-wrap:wrap;">';for(var i=0;i<gk.length;i++)h+='<div style="display:flex;align-items:center;gap:5px;font-size:.88em;"><div style="width:14px;height:14px;background:'+C[i%C.length]+';border-radius:3px;"></div>'+gk[i]+'</div>';return h+'</div>';}}
function lineChart(g,xv,gk,lo,hi,sw,yl){{var r=hi-lo,W=750,H=340,L=65,R=20,T=20,B=50,pw=W-L-R,ph=H-T-B;var h='<svg viewBox="0 0 '+W+' '+H+'" style="width:100%;height:auto;background:white;border-radius:8px;border:1px solid #e0e0e0;">';h+='<rect x="'+L+'" y="'+T+'" width="'+pw+'" height="'+ph+'" fill="#fafafa"/>';for(var i=0;i<=5;i++){{var y=T+ph-(i/5)*ph,v=lo+(i/5)*r;h+='<line x1="'+L+'" y1="'+y+'" x2="'+(W-R)+'" y2="'+y+'" stroke="#e8e8e8"/><text x="'+(L-6)+'" y="'+(y+4)+'" text-anchor="end" font-size="11" fill="#666">'+v.toFixed(2)+'</text>';}}h+='<line x1="'+L+'" y1="'+(H-B)+'" x2="'+(W-R)+'" y2="'+(H-B)+'" stroke="#999"/><line x1="'+L+'" y1="'+T+'" x2="'+L+'" y2="'+(H-B)+'" stroke="#999"/>';h+='<text x="'+((L+W-R)/2)+'" y="'+(H-8)+'" text-anchor="middle" font-size="12" fill="#555">Episode</text>';h+='<text x="14" y="'+((T+H-B)/2)+'" text-anchor="middle" font-size="12" fill="#555" transform="rotate(-90,14,'+((T+H-B)/2)+')">'+yl+'</text>';var mx=Math.max.apply(null,xv);for(var i=0;i<xv.length;i++){{var x=parseInt(xv[i]),px=L+(x/(mx||1))*pw;if(xv.length<=15||x%Math.max(1,Math.floor(mx/10))===0||x===mx)h+='<text x="'+px+'" y="'+(H-B+15)+'" text-anchor="middle" font-size="10" fill="#666">'+x+'</text>';}}for(var gi=0;gi<gk.length;gi++){{var k=gk[gi],c=C[gi%C.length],ra=[];for(var i=0;i<xv.length;i++){{var a=(g[k]||{{}})[parseInt(xv[i])]||[],v=0;for(var j=0;j<a.length;j++)v+=a[j];if(a.length)v/=a.length;ra.push(v);}}var s=sm(ra,sw),pts=[];for(var i=0;i<xv.length;i++){{var px=L+(parseInt(xv[i])/(mx||1))*pw,py=T+ph-((s[i]-lo)/r)*ph;py=Math.max(T,Math.min(H-B,py));pts.push(px+','+py);}}h+='<polyline points="'+pts.join(' ')+'" fill="none" stroke="'+c+'" stroke-width="2.5" stroke-linejoin="round"/>';for(var i=0;i<pts.length;i++){{var p=pts[i].split(',');h+='<circle cx="'+p[0]+'" cy="'+p[1]+'" r="4" fill="white" stroke="'+c+'" stroke-width="2"/>';}}}}return h+'</svg>';}}
function barChart(g,xv,gk,lo,hi){{var r=hi-lo,bh=260,bw=Math.max(30,Math.min(70,650/(xv.length*Math.max(1,gk.length))));var h='<div style="display:flex;align-items:flex-end;gap:4px;height:'+bh+'px;border-bottom:2px solid #ccc;border-left:2px solid #ccc;padding:0 12px;margin-left:55px;position:relative;">';for(var i=0;i<=4;i++){{var y=bh-(i/4)*bh,v=lo+(i/4)*r;h+='<div style="position:absolute;left:-55px;top:'+(y-7)+'px;font-size:10px;color:#666;text-align:right;width:48px;">'+v.toFixed(2)+'</div>';}}for(var i=0;i<xv.length;i++){{for(var gi=0;gi<gk.length;gi++){{var a=(g[gk[gi]]||{{}})[xv[i]]||[],v=0;for(var j=0;j<a.length;j++)v+=a[j];if(a.length)v/=a.length;var p=Math.max(2,((v-lo)/r)*(bh-10));h+='<div style="width:'+bw+'px;height:'+p+'px;background:'+C[gi%C.length]+';border-radius:4px 4px 0 0;" title="'+v.toFixed(4)+'"></div>';}}if(i<xv.length-1)h+='<div style="width:8px;"></div>';}}h+='</div><div style="display:flex;gap:4px;padding:0 12px;margin-left:55px;">';for(var i=0;i<xv.length;i++){{var w=bw*gk.length+(i<xv.length-1?8:0),lb=String(xv[i]);if(lb.length>20)lb='...'+lb.substring(lb.length-17);h+='<div style="width:'+w+'px;text-align:center;font-size:.78em;color:#555;padding-top:4px;" title="'+xv[i]+'">'+lb+'</div>';}}return h+'</div>';}}
function tbl(xv,gk,g,xl){{var h='<table><tr><th>'+xl+'</th>';for(var i=0;i<gk.length;i++)h+='<th>'+gk[i]+'</th>';h+='</tr>';for(var i=0;i<xv.length;i++){{h+='<tr><td>'+xv[i]+'</td>';for(var gi=0;gi<gk.length;gi++){{var a=(g[gk[gi]]||{{}})[xv[i]]||[],v=0;for(var j=0;j<a.length;j++)v+=a[j];if(a.length)v/=a.length;h+='<td>'+v.toFixed(4)+'</td>';}}h+='</tr>';}}return h+'</table>';}}
</script>
</body>
</html>"""


# =============================================================================
# Tab builders
# =============================================================================

def _build_leaderboard_tab(leaderboard: List[Dict]) -> str:
    if not leaderboard:
        return '<p>No leaderboard data.</p>'

    rows = ""
    for entry in leaderboard:
        rank = entry.get("rank", 0)
        rank_cls = f"rank-{rank}" if rank <= 3 else ""
        rows += f'<tr><td class="{rank_cls}">{rank}</td><td>{_esc(entry.get("model", ""))}</td>'
        rows += f'<td>{entry.get("avg_reward", 0):.4f}</td>'
        rows += f'<td>{entry.get("n_configs", "—")}</td></tr>\n'

    return f"""
    <div class="card">
    <h3>Leaderboard (ranked by avg reward across all configs)</h3>
    <table id="tbl-leaderboard">
      <tr><th onclick="sortTable('tbl-leaderboard',0)">Rank</th>
          <th onclick="sortTable('tbl-leaderboard',1)">Model</th>
          <th onclick="sortTable('tbl-leaderboard',2)">Avg Reward</th>
          <th onclick="sortTable('tbl-leaderboard',3)"># Configs</th></tr>
      {rows}
    </table>
    </div>"""


def _build_results_tab(per_model: Dict, models: List[str], all_bids: List[str], envs: List[str]) -> str:
    env_opts = '<option value="all">All Environments</option>'
    for e in envs:
        env_opts += f'<option value="{_esc(e)}">{_esc(e)}</option>'

    rows = ""
    for model in models:
        for bid in all_bids:
            m = per_model.get(model, {}).get(bid, {})
            if not m:
                continue
            env = bid.split("/")[0]
            short_bid = "/".join(bid.split("/")[1:]) if "/" in bid else bid
            rows += f'<tr><td>{_esc(env)}</td><td>{_esc(model)}</td><td><code>{_esc(short_bid)}</code></td>'
            rows += f'<td>{m.get("avg_mean_reward", 0):.4f}</td>'
            rows += f'<td>{m.get("avg_improvement", 0):+.4f}</td>'
            rows += f'<td>{m.get("avg_success_rate", 0):.4f}</td>'
            rows += f'<td>{m.get("avg_initial_reward", 0):.4f}</td>'
            rows += f'<td>{m.get("avg_final_reward", 0):.4f}</td>'
            rows += f'<td>{m.get("learning_slope", 0):+.4f}</td>'
            rows += f'<td>{m.get("avg_total_turns", 0):.1f}</td>'
            rows += f'<td>{m.get("n_trajectories", 0)}</td></tr>\n'

    return f"""
    <div class="filter-bar">
      <label>Filter:</label>
      <select onchange="filterTable('tbl-results', 0, this.value)">{env_opts}</select>
    </div>
    <div class="card">
    <h3>Full Results (model &times; config)</h3>
    <table id="tbl-results">
      <tr><th onclick="sortTable('tbl-results',0)">Env</th>
          <th onclick="sortTable('tbl-results',1)">Model</th>
          <th onclick="sortTable('tbl-results',2)">Config</th>
          <th onclick="sortTable('tbl-results',3)">Mean Reward</th>
          <th onclick="sortTable('tbl-results',4)">Improvement</th>
          <th onclick="sortTable('tbl-results',5)">Success</th>
          <th onclick="sortTable('tbl-results',6)">Initial</th>
          <th onclick="sortTable('tbl-results',7)">Final</th>
          <th onclick="sortTable('tbl-results',8)">Slope</th>
          <th onclick="sortTable('tbl-results',9)">Turns</th>
          <th onclick="sortTable('tbl-results',10)"># Traj</th></tr>
      {rows}
    </table>
    </div>"""


def _build_per_env_tab(per_env: Dict, models: List[str], envs: List[str]) -> str:
    if not per_env:
        return '<p>No per-environment data.</p>'

    metric_keys = ["avg_mean_reward", "avg_improvement", "avg_success_rate",
                    "avg_initial_reward", "avg_final_reward"]

    sections = ""
    for metric in metric_keys:
        label = metric.replace("avg_", "").replace("_", " ").title()
        rows = ""
        for env in envs:
            rows += f'<tr><td><strong>{_esc(env)}</strong></td>'
            for model in models:
                val = per_env.get(env, {}).get(model, {}).get(metric, None)
                rows += f'<td>{val:.4f}</td>' if val is not None else '<td>—</td>'
            rows += '</tr>\n'

        model_headers = "".join(f"<th>{_esc(m)}</th>" for m in models)
        sections += f"""
        <div class="card">
        <h3>{label}</h3>
        <table><tr><th>Environment</th>{model_headers}</tr>{rows}</table>
        </div>"""

    return sections


def _build_complexity_tab(per_complexity: Dict, models: List[str], complexities: List[str]) -> str:
    if not per_complexity:
        return '<p>No complexity data.</p>'

    rows = ""
    for c in complexities:
        rows += f'<tr><td><strong>{_esc(c)}</strong></td>'
        for model in models:
            val = per_complexity.get(c, {}).get(model, {}).get("avg_mean_reward", None)
            rows += f'<td>{val:.4f}</td>' if val is not None else '<td>—</td>'
        rows += '</tr>\n'

    model_headers = "".join(f"<th>{_esc(m)}</th>" for m in models)
    return f"""
    <div class="card">
    <h3>Performance by Latent Complexity</h3>
    <table><tr><th>Complexity</th>{model_headers}</tr>{rows}</table>
    </div>"""


def _build_compare_tabs(per_model, models, all_bids, envs):
    """Build Episode Compare and Variable Compare tabs. Returns (ep_html, var_html)."""
    import json

    dims = {"model": set(), "env": set(), "latent": set(), "prompt": set(), "feedback": set()}
    flat, ep_data = [], []
    for mn, bm in per_model.items():
        dims["model"].add(mn)
        for bid, m in bm.items():
            p = bid.split("/")
            e, l, pr, fb = (p[0] if len(p)>0 else ""), (p[1] if len(p)>1 else ""), (p[2] if len(p)>2 else ""), (p[3] if len(p)>3 else "")
            dims["env"].add(e); dims["latent"].add(l); dims["prompt"].add(pr); dims["feedback"].add(fb)
            flat.append({"model":mn,"env":e,"latent":l,"prompt":pr,"feedback":fb,
                "avg_trajectory_reward":m.get("avg_trajectory_reward",0),"avg_mean_reward":m.get("avg_mean_reward",0),
                "avg_improvement":m.get("avg_improvement",0),"avg_total_turns":m.get("avg_total_turns",0),
                "avg_mean_turns_per_episode":m.get("avg_mean_turns_per_episode",0),
                "avg_initial_reward":m.get("avg_initial_reward",0),"avg_final_reward":m.get("avg_final_reward",0)})
            er, et = m.get("per_episode_avg_rewards",[]), m.get("per_episode_avg_turns",[])
            for ei in range(max(len(er),len(et))):
                ep_data.append({"model":mn,"env":e,"latent":l,"prompt":pr,"feedback":fb,"episode":ei,
                    "reward":er[ei] if ei<len(er) else 0,"turns":et[ei] if ei<len(et) else 0})

    dj = json.dumps({k:sorted(v) for k,v in dims.items()}, default=str)
    fj = json.dumps(flat, default=str)
    ej = json.dumps(ep_data, default=str)

    # Build filter HTML for both tabs
    fh = {"ep":"", "var":""}
    for d in ["model","env","latent","prompt","feedback"]:
        vs = sorted(dims[d])
        opts = '<option value="all">All (avg)</option>' + "".join(f'<option value="{_esc(v)}">{_esc(v)}</option>' for v in vs)
        for px in ["ep","var"]:
            onch = f' onchange="cascadeEnvLatent(\'{px}\', \'{d}\')"' if d in ("env", "latent") else ""
            fh[px] += f'''<div id="{px}-dim-{d}"><label style="font-size:0.85em;color:#555;">{d.title()}</label><br><select id="{px}-f-{d}"{onch} style="padding:4px 8px;min-width:100px;">{opts}</select></div>'''

    ep_html = f'''
    <div class="card">
    <h3>Episode Compare</h3>
    <p style="color:#666;font-size:0.9em;">X = Episode Number. Compare one variable as colored lines.</p>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;">
      <div><label style="font-size:0.85em;font-weight:bold;">Compare</label><br>
        <select id="ep-compare" onchange="epHideFilter()" style="padding:5px 8px;">
          <option value="model">Model</option><option value="latent">Latent</option>
          <option value="prompt">Prompt</option><option value="feedback">Feedback</option><option value="env">Environment</option>
        </select></div>
      <div><label style="font-size:0.85em;font-weight:bold;">Y-Axis</label><br>
        <select id="ep-metric" style="padding:5px 8px;">
          <option value="reward">Reward per Episode</option><option value="turns">Turns per Episode</option>
        </select></div>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px;padding:10px;background:#f9f9f9;border-radius:6px;border:1px solid #eee;">
      <span style="font-size:0.85em;font-weight:bold;align-self:center;">Filter:</span>{fh["ep"]}
    </div>
    <div style="display:flex;gap:12px;align-items:end;margin-bottom:12px;">
      <div><label style="font-size:0.85em;color:#555;">Y-Min</label><br><input id="ep-ymin" type="number" step="any" style="padding:4px;width:70px;border:1px solid #ccc;border-radius:4px;" placeholder="auto"></div>
      <div><label style="font-size:0.85em;color:#555;">Y-Max</label><br><input id="ep-ymax" type="number" step="any" style="padding:4px;width:70px;border:1px solid #ccc;border-radius:4px;" placeholder="auto"></div>
      <div><label style="font-size:0.85em;color:#555;">Smoothing: <span id="ep-sv">0</span></label><br><input id="ep-sm" type="range" min="0" max="5" value="0" style="width:100px;" oninput="document.getElementById('ep-sv').textContent=this.value;epRender()"></div>
      <button onclick="epRender()" style="padding:8px 20px;background:#388E3C;color:white;border:none;border-radius:6px;cursor:pointer;font-weight:500;">Render</button>
    </div>
    <div id="ep-chart" style="min-height:40px;max-width:700px;"></div>
    <div id="ep-table" style="margin-top:8px;"></div>
    <button onclick="epSave()" style="margin-top:8px;padding:5px 14px;background:#1976D2;color:white;border:none;border-radius:4px;cursor:pointer;">+ Save Below</button>
    </div><div id="ep-saved"></div>'''

    var_html = f'''
    <div class="card">
    <h3>Variable Compare</h3>
    <p style="color:#666;font-size:0.9em;">X = a variable. Other variables averaged or fixed.</p>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;">
      <div><label style="font-size:0.85em;font-weight:bold;">X-Axis</label><br>
        <select id="var-xaxis" onchange="varHideFilter()" style="padding:5px 8px;">
          <option value="model">Model</option><option value="latent">Latent</option>
          <option value="env">Environment</option><option value="prompt">Prompt</option><option value="feedback">Feedback</option>
        </select></div>
      <div><label style="font-size:0.85em;font-weight:bold;">Y-Axis</label><br>
        <select id="var-metric" style="padding:5px 8px;">
          <option value="avg_trajectory_reward">Total Reward</option><option value="avg_mean_reward">Mean Reward</option>
          <option value="avg_initial_reward">Initial Reward</option><option value="avg_final_reward">Final Reward</option>
          <option value="avg_improvement">Improvement</option><option value="avg_total_turns">Total Turns</option>
          <option value="avg_mean_turns_per_episode">Avg Turns/Episode</option>
        </select></div>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px;padding:10px;background:#f9f9f9;border-radius:6px;border:1px solid #eee;">
      <span style="font-size:0.85em;font-weight:bold;align-self:center;">Filter:</span>{fh["var"]}
    </div>
    <div style="display:flex;gap:12px;align-items:end;margin-bottom:12px;">
      <div><label style="font-size:0.85em;color:#555;">Y-Min</label><br><input id="var-ymin" type="number" step="any" style="padding:4px;width:70px;border:1px solid #ccc;border-radius:4px;" placeholder="auto"></div>
      <div><label style="font-size:0.85em;color:#555;">Y-Max</label><br><input id="var-ymax" type="number" step="any" style="padding:4px;width:70px;border:1px solid #ccc;border-radius:4px;" placeholder="auto"></div>
      <button onclick="varRender()" style="padding:8px 20px;background:#388E3C;color:white;border:none;border-radius:6px;cursor:pointer;font-weight:500;">Render</button>
    </div>
    <div id="var-chart" style="min-height:40px;"></div>
    <div id="var-table" style="margin-top:8px;"></div>
    <button onclick="varSave()" style="margin-top:8px;padding:5px 14px;background:#1976D2;color:white;border:none;border-radius:4px;cursor:pointer;">+ Save Below</button>
    </div><div id="var-saved"></div>'''

    js = f'''<script>
    var D={dj};
    var F={fj};
    var E={ej};
    console.log('Compare JS loaded: D keys='+Object.keys(D)+', F rows='+F.length+', E rows='+E.length);
    // Build env↔latent cascade mapping
    var ENV_TO_LATENTS={{}},LATENT_TO_ENV={{}};
    for(var i=0;i<F.length;i++){{if(!ENV_TO_LATENTS[F[i].env])ENV_TO_LATENTS[F[i].env]={{}};ENV_TO_LATENTS[F[i].env][F[i].latent]=1;LATENT_TO_ENV[F[i].latent]=F[i].env;}}
    function cascadeEnvLatent(px,changed){{var envSel=document.getElementById(px+'-f-env'),latSel=document.getElementById(px+'-f-latent');if(!envSel||!latSel)return;if(changed==='latent'&&latSel.value!=='all'){{var pe=LATENT_TO_ENV[latSel.value];if(pe&&envSel.value!==pe)envSel.value=pe;}}var cur=latSel.value,env=envSel.value,allowed;if(env==='all')allowed=Object.keys(LATENT_TO_ENV);else allowed=Object.keys(ENV_TO_LATENTS[env]||{{}});allowed.sort();var h='<option value="all">All (avg)</option>';for(var i=0;i<allowed.length;i++)h+='<option value="'+allowed[i]+'">'+allowed[i]+'</option>';latSel.innerHTML=h;latSel.value=(cur!=='all'&&allowed.indexOf(cur)>=0)?cur:'all';}}
    function gf(px,skip){{var f={{}};var ds=['model','env','latent','prompt','feedback'];for(var i=0;i<ds.length;i++){{var d=ds[i];if(skip.indexOf(d)>=0)continue;var s=document.getElementById(px+'-f-'+d);if(s&&s.value!=='all')f[d]=s.value;}}return f;}}
    function fd(src,f){{var o=[];for(var i=0;i<src.length;i++){{var ok=1;for(var k in f)if(src[i][k]!==f[k]){{ok=0;break;}}if(ok)o.push(src[i]);}}return o;}}
    function sv(px,ci,ti,si){{var ch=document.getElementById(ci).innerHTML,tb=document.getElementById(ti).innerHTML;if(!window['_n'+px])window['_n'+px]=0;window['_n'+px]++;var ds=['model','env','latent','prompt','feedback'],parts=[];for(var i=0;i<ds.length;i++){{var s=document.getElementById(px+'-f-'+ds[i]);if(s){{var el=document.getElementById(px+'-dim-'+ds[i]);if(el&&el.style.display==='none')continue;parts.push(ds[i]+'='+(s.value==='all'?'All':s.value));}}}}var p=document.createElement('div');p.className='card';p.innerHTML='<div style="display:flex;justify-content:space-between;margin-bottom:6px;"><div><b>Chart '+window['_n'+px]+'</b> <span style="font-size:.82em;color:#666;">'+parts.join(' | ')+'</span></div><button onclick="this.parentElement.parentElement.remove()" style="background:#D32F2F;color:white;border:none;border-radius:4px;padding:3px 10px;cursor:pointer;">X</button></div>'+ch+tb;document.getElementById(si).appendChild(p);}}
    function epHideFilter(){{var cv=document.getElementById('ep-compare').value;var ds=['model','env','latent','prompt','feedback'];for(var i=0;i<ds.length;i++){{var el=document.getElementById('ep-dim-'+ds[i]);if(el)el.style.display=(ds[i]===cv)?'none':'';}}}}
    function varHideFilter(){{var x=document.getElementById('var-xaxis').value;var ds=['model','env','latent','prompt','feedback'];for(var i=0;i<ds.length;i++){{var el=document.getElementById('var-dim-'+ds[i]);if(el)el.style.display=(ds[i]===x)?'none':'';}}}}
    function epRender(){{try{{var cv=document.getElementById('ep-compare').value,m=document.getElementById('ep-metric').value;var f=gf('ep',[cv]),fl=fd(E,f);if(!fl.length){{document.getElementById('ep-chart').innerHTML='<p style="color:#888;">No data.</p>';document.getElementById('ep-table').innerHTML='';return;}}var g={{}};for(var i=0;i<fl.length;i++){{var gk=fl[i][cv],xk=fl[i].episode;if(!g[gk])g[gk]={{}};if(!g[gk][xk])g[gk][xk]=[];var v=parseFloat(fl[i][m]);if(!isNaN(v))g[gk][xk].push(v);}}var xs={{}};for(var i=0;i<fl.length;i++)xs[fl[i].episode]=1;var xv=Object.keys(xs).sort(function(a,b){{return a-b;}});var gks=Object.keys(g).sort();var r=rng(g,'ep-ymin','ep-ymax');var sw=parseInt(document.getElementById('ep-sm').value)||0;document.getElementById('ep-sv').textContent=sw;var yl=document.getElementById('ep-metric').options[document.getElementById('ep-metric').selectedIndex].text;document.getElementById('ep-chart').innerHTML=lineChart(g,xv,gks,r[0],r[1],sw,yl)+leg(gks);document.getElementById('ep-table').innerHTML=tbl(xv,gks,g,'Episode');}}catch(e){{document.getElementById('ep-chart').innerHTML='<p style="color:red;">'+e.message+'</p>';}}}}
    function epSave(){{sv('ep','ep-chart','ep-table','ep-saved');}}
    function varRender(){{try{{var x=document.getElementById('var-xaxis').value,m=document.getElementById('var-metric').value;var f=gf('var',[x]),fl=fd(F,f);if(!fl.length){{document.getElementById('var-chart').innerHTML='<p style="color:#888;">No data.</p>';document.getElementById('var-table').innerHTML='';return;}}var g={{'_a':{{}}}};for(var i=0;i<fl.length;i++){{var xk=fl[i][x];if(!g['_a'][xk])g['_a'][xk]=[];var v=parseFloat(fl[i][m]);if(!isNaN(v))g['_a'][xk].push(v);}}var xv=Object.keys(g['_a']).sort();var gks=['_a'];var r=rng(g,'var-ymin','var-ymax');document.getElementById('var-chart').innerHTML=barChart(g,xv,gks,r[0],r[1]);document.getElementById('var-table').innerHTML=tbl(xv,gks,g,x.charAt(0).toUpperCase()+x.slice(1));}}catch(e){{document.getElementById('var-chart').innerHTML='<p style="color:red;">'+e.message+'</p>';}}}}
    function varSave(){{sv('var','var-chart','var-table','var-saved');}}
    setTimeout(function(){{try{{epHideFilter();varHideFilter();console.log('Compare init OK');}}catch(e){{console.error('Compare init error:',e);}}}},100);
    </script>'''

    return ep_html, var_html + js


def _build_learning_curves_tab(charts_b64: Dict[str, str], all_bids: List[str],
                               per_model: Dict[str, Dict[str, Any]] = None) -> str:
    import json as _json

    # Build per-bid data: { bid: { model: { rewards: [...], turns: [...] } } }
    lc_data: Dict[str, Dict[str, Any]] = {}
    if per_model:
        for mn, bm in per_model.items():
            for bid, m in bm.items():
                if bid not in lc_data:
                    lc_data[bid] = {}
                rewards = m.get("per_episode_avg_rewards", [])
                turns = m.get("per_episode_avg_turns", [])
                if rewards or turns:
                    lc_data[bid][mn] = {"r": rewards, "t": turns}

    if not lc_data:
        return '<p>No learning curve data available.</p>'

    lc_json = _json.dumps(lc_data, default=str)

    # Extract unique envs for filter
    lc_envs = sorted({bid.split("/")[0] for bid in lc_data})
    env_opts = '<option value="all">All</option>' + "".join(
        f'<option value="{_esc(e)}">{_esc(e)}</option>' for e in lc_envs)

    # Placeholder divs for each bid
    bid_cards = ""
    for bid in sorted(lc_data.keys()):
        short = "/".join(bid.split("/")[1:]) if "/" in bid else bid
        env = bid.split("/")[0]
        safe_id = bid.replace("/", "__").replace(" ", "_")
        bid_cards += f"""
        <details open class="lc-bid-card" data-env="{_esc(env)}">
        <summary style="font-weight:500;margin-bottom:6px;">{_esc(env)} / {_esc(short)}</summary>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
          <div><div style="font-size:0.85em;font-weight:500;color:#555;margin-bottom:4px;">Reward per Episode</div>
               <div id="lc-r-{_esc(safe_id)}"></div></div>
          <div><div style="font-size:0.85em;font-weight:500;color:#555;margin-bottom:4px;">Turns per Episode</div>
               <div id="lc-t-{_esc(safe_id)}"></div></div>
        </div>
        </details>"""

    return f"""
    <div class="card">
    <h3>Learning Curves</h3>
    <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:end;margin-bottom:16px;padding:10px;background:#f9f9f9;border-radius:6px;border:1px solid #eee;">
      <div><label style="font-size:0.85em;font-weight:bold;">Environment</label><br>
        <select id="lc-env-filter" onchange="lcFilter()" style="padding:5px 8px;">{env_opts}</select></div>
      <div><label style="font-size:0.85em;color:#555;">Smoothing: <span id="lc-sv">0</span></label><br>
        <input id="lc-sm" type="range" min="0" max="5" value="0" style="width:120px;"
               oninput="document.getElementById('lc-sv').textContent=this.value;lcRenderAll()"></div>
    </div>
    {bid_cards}
    </div>
    <script>
    var LC={lc_json};
    function lcFilter(){{var ev=document.getElementById('lc-env-filter').value;var cards=document.querySelectorAll('.lc-bid-card');for(var i=0;i<cards.length;i++){{var ce=cards[i].getAttribute('data-env');cards[i].style.display=(ev==='all'||ce===ev)?'':'none';}}}}
    function lcRenderAll(){{try{{var sw=parseInt(document.getElementById('lc-sm').value)||0;document.getElementById('lc-sv').textContent=sw;for(var bid in LC){{var safe=bid.replace(/\\//g,'__').replace(/ /g,'_');var md=LC[bid],models=Object.keys(md).sort();var maxEp=0;for(var mi=0;mi<models.length;mi++){{var rl=md[models[mi]].r.length,tl=md[models[mi]].t.length;if(rl>maxEp)maxEp=rl;if(tl>maxEp)maxEp=tl;}}if(maxEp===0)continue;var xv=[];for(var ei=0;ei<maxEp;ei++)xv.push(ei);var gR={{}},gT={{}};for(var mi=0;mi<models.length;mi++){{var mn=models[mi];gR[mn]={{}};gT[mn]={{}};for(var ei=0;ei<maxEp;ei++){{var rv=md[mn].r[ei];if(rv!==undefined)gR[mn][ei]=[rv];else gR[mn][ei]=[0];var tv=md[mn].t[ei];if(tv!==undefined)gT[mn][ei]=[tv];else gT[mn][ei]=[0];}}}}var rEl=document.getElementById('lc-r-'+safe),tEl=document.getElementById('lc-t-'+safe);if(rEl){{var lo=1e9,hi=-1e9;for(var mn in gR)for(var x in gR[mn]){{var v=gR[mn][x][0];if(v<lo)lo=v;if(v>hi)hi=v;}}var pad=(hi-lo)*0.1;lo=Math.max(0,lo-pad);hi=hi+pad;if(hi<=lo)hi=lo+0.01;rEl.innerHTML=lineChart(gR,xv,models,lo,hi,sw,'Avg Reward')+leg(models);}}if(tEl){{var lo=1e9,hi=-1e9;for(var mn in gT)for(var x in gT[mn]){{var v=gT[mn][x][0];if(v<lo)lo=v;if(v>hi)hi=v;}}var pad=(hi-lo)*0.1;lo=Math.max(0,lo-pad);hi=hi+pad;if(hi<=lo)hi=lo+0.01;tEl.innerHTML=lineChart(gT,xv,models,lo,hi,sw,'Avg Turns')+leg(models);}}}}}}catch(e){{console.error('LC render error:',e);}}}}
    setTimeout(function(){{lcRenderAll();lcFilter();}},200);
    </script>"""


def _build_double_agent_tab(
    da_schedule: Dict, da_agent: Dict, da_comparisons: Dict, charts_b64: Dict[str, str]
) -> str:
    sections = ""

    # Schedule summary table
    if da_schedule:
        rows = ""
        for key, m in sorted(da_schedule.items()):
            rows += f'<tr><td><code>{_esc(key)}</code></td>'
            rows += f'<td>{m.get("switch_episode", 0)}</td>'
            rows += f'<td>{m.get("avg_pre_switch_reward", 0):.4f}</td>'
            rows += f'<td>{m.get("avg_post_switch_reward", 0):.4f}</td>'
            rows += f'<td>{m.get("avg_transfer_effect", 0):+.4f}</td>'
            rows += f'<td>{m.get("avg_adaptation_speed", 0):+.4f}</td>'
            rows += f'<td>{m.get("n_trajectories", 0)}</td></tr>\n'

        sections += f"""
        <div class="card">
        <h3>Double-Agent Schedule Summary</h3>
        <table id="tbl-da-schedule">
          <tr><th>Schedule</th><th>Switch Ep</th><th>Pre-Switch</th><th>Post-Switch</th>
              <th>Transfer</th><th>Adaptation</th><th># Traj</th></tr>
          {rows}
        </table>
        </div>"""

    # Per-agent breakdown
    if da_agent:
        rows = ""
        for key, agents in sorted(da_agent.items()):
            for agent, m in sorted(agents.items()):
                rows += f'<tr><td><code>{_esc(key)}</code></td><td>{_esc(agent)}</td>'
                rows += f'<td>{m.get("avg_reward", 0):.4f}</td>'
                rows += f'<td>{m.get("std_reward", 0):.4f}</td>'
                rows += f'<td>{m.get("success_rate", 0):.4f}</td>'
                rows += f'<td>{m.get("avg_turns", 0):.1f}</td>'
                rows += f'<td>{m.get("n_episodes", 0)}</td></tr>\n'

        sections += f"""
        <div class="card">
        <h3>Per-Agent Breakdown</h3>
        <table>
          <tr><th>Schedule</th><th>Agent</th><th>Avg Reward</th><th>Std</th>
              <th>Success</th><th>Turns</th><th># Episodes</th></tr>
          {rows}
        </table>
        </div>"""

    # Comparison
    if da_comparisons and da_comparisons.get("per_benchmark_id"):
        comp_rows = ""
        for bid, m in sorted(da_comparisons["per_benchmark_id"].items()):
            comp_rows += f'<tr><td><code>{_esc(bid)}</code></td>'
            comp_rows += f'<td>{m.get("overall_improvement", 0):+.4f}</td>'
            comp_rows += f'<td>{m.get("initial_prior_improvement", 0):+.4f}</td>'
            comp_rows += f'<td>{m.get("icl_difference", 0):+.4f}</td>'
            tf = m.get("transfer_f_to_b")
            tb = m.get("transfer_b_to_f")
            comp_rows += f'<td>{tf:+.4f}</td>' if tf is not None else '<td>—</td>'
            comp_rows += f'<td>{tb:+.4f}</td>' if tb is not None else '<td>—</td>'
            comp_rows += '</tr>\n'

        sections += f"""
        <div class="card">
        <h3>Model Comparison (Finetuned vs Base)</h3>
        <table>
          <tr><th>Config</th><th>Overall Impr.</th><th>Initial Prior</th><th>ICL Diff</th>
              <th>Transfer F&rarr;B</th><th>Transfer B&rarr;F</th></tr>
          {comp_rows}
        </table>
        </div>"""

    # Charts
    for chart_key, chart_label in [
        ("bar_pre_post", "Pre vs Post Switch Reward"),
        ("bar_transfer", "Transfer Effects"),
    ]:
        if chart_key in charts_b64:
            sections += f"""
            <div class="card">
            <h3>{chart_label}</h3>
            <img class="chart-img" src="data:image/png;base64,{charts_b64[chart_key]}" alt="{chart_label}">
            </div>"""

    return sections or '<p>No double-agent data.</p>'


# =============================================================================
# Chart generation (matplotlib → base64 PNG)
# =============================================================================

def _generate_charts_b64(
    per_model: Dict, per_env: Dict, da_schedule: Dict,
    all_bids: List[str], envs: List[str],
) -> Dict[str, str]:
    """Generate all charts as base64-encoded PNGs. Returns empty dict if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plots import (
            plot_learning_curves, plot_env_bar_chart,
            plot_pre_post_switch_comparison, plot_transfer_effects,
        )
    except ImportError:
        return {}

    charts: Dict[str, str] = {}

    # Env bar chart
    if per_env:
        try:
            fig = plot_env_bar_chart(per_env)
            charts["bar_env"] = _fig_to_b64(fig)
            plt.close(fig)
        except Exception:
            pass

    # Learning curves are now rendered interactively in JS (no matplotlib PNGs needed)

    # Double-agent charts
    if da_schedule:
        try:
            fig = plot_pre_post_switch_comparison(da_schedule)
            charts["bar_pre_post"] = _fig_to_b64(fig)
            plt.close(fig)
        except Exception:
            pass

        try:
            fig = plot_transfer_effects(da_schedule)
            charts["bar_transfer"] = _fig_to_b64(fig)
            plt.close(fig)
        except Exception:
            pass

    return charts


def _fig_to_b64(fig) -> str:
    """Convert matplotlib Figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
