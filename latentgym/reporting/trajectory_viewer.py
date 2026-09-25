"""
Trajectory viewer — render trajectories as human-readable text or HTML.

Supports:
    - Single trajectory rendering (text + HTML) with full metadata, ground truth,
      agent assignments, turn efficiency, and env params
    - Batch browsing from a DataStore directory with filters
    - Multi-trajectory comparison (side-by-side episode tables)

Usage:
    viewer = TrajectoryViewer()

    # Single trajectory
    traj = DataStore.load_trajectory("results/trajectories/.../traj_0000.json")
    print(viewer.render_text(traj))

    # Browse from DataStore
    trajs = viewer.browse("results/", model="gpt-4o", env="bandits",
                          latent="loyal_favorite_0", prompt="full_info")
    for traj in trajs:
        print(viewer.render_text(traj))

    # Compare trajectories across models
    print(viewer.render_comparison([traj_gpt4, traj_claude], label="bandits"))
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrajectoryViewer:
    """Renders trajectory data as human-readable text or HTML.

    All render methods accept a trajectory dict (from TrajectoryResult.to_dict()
    or DataStore.load_trajectory()).
    """

    # ── Browse / filter ──────────────────────────────────────────────────

    def browse(
        self,
        data_dir: str,
        model: Optional[str] = None,
        env: Optional[str] = None,
        latent: Optional[str] = None,
        prompt: Optional[str] = None,
        feedback: Optional[str] = None,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """Load and filter trajectories from a DataStore directory.

        Args:
            data_dir: DataStore output directory
            model: Filter by model name (substring match)
            env: Filter by env_name (exact match)
            latent: Filter by latent_id (exact match)
            prompt: Filter by prompt_id (exact match)
            feedback: Filter by feedback_id (exact match)
            max_results: Maximum trajectories to return

        Returns:
            List of trajectory dicts matching all filters.
        """
        traj_dir = Path(data_dir) / "trajectories"
        if not traj_dir.exists():
            return []

        # Find all traj_*.json files recursively (handles any nesting depth —
        # model names with "/" like "openrouter/openai:gpt-4o" create extra dirs)
        results = []
        for traj_path in sorted(traj_dir.rglob("traj_*.json")):
            if len(results) >= max_results:
                break

            traj = self._load_json(traj_path)
            if not traj:
                continue

            # Apply filters
            model_name = traj.get("model_name", "")
            if model and model.lower() not in model_name.lower():
                continue
            if env and traj.get("env_name", "") != env:
                continue
            if latent and traj.get("latent_id", "") != latent:
                continue
            if prompt and traj.get("prompt_id", "") != prompt:
                continue
            if feedback and traj.get("feedback_id", "") != feedback:
                continue

            results.append(traj)

        return results

    def list_filters(self, data_dir: str) -> Dict[str, List[str]]:
        """Scan a DataStore directory and return available filter values.

        Returns:
            Dict with keys: models, envs, latents, prompts, feedbacks
        """
        traj_dir = Path(data_dir) / "trajectories"
        models, envs, latents, prompts, feedbacks = set(), set(), set(), set(), set()

        if not traj_dir.exists():
            return {"models": [], "envs": [], "latents": [], "prompts": [], "feedbacks": []}

        # Sample one traj per directory to extract filter values
        seen_dirs = set()
        for traj_path in traj_dir.rglob("traj_*.json"):
            parent = str(traj_path.parent)
            if parent in seen_dirs:
                continue
            seen_dirs.add(parent)
            traj = self._load_json(traj_path)
            if traj:
                models.add(traj.get("model_name", ""))
                envs.add(traj.get("env_name", ""))
                latents.add(traj.get("latent_id", ""))
                prompts.add(traj.get("prompt_id", ""))
                feedbacks.add(traj.get("feedback_id", ""))

        return {
            "models": sorted(models),
            "envs": sorted(envs - {""}),
            "latents": sorted(latents - {""}),
            "prompts": sorted(prompts - {""}),
            "feedbacks": sorted(feedbacks - {""}),
        }

    # ── Single trajectory rendering ──────────────────────────────────────

    def render_text(self, traj: Dict[str, Any]) -> str:
        """Render a trajectory as plain text with full metadata and ground truth."""
        lines = []

        # ── Header ──
        lines.append("=" * 74)
        lines.append(f"TRAJECTORY: {traj.get('benchmark_id', 'unknown')}")
        lines.append("=" * 74)
        lines.append(f"Model:      {traj.get('model_name', '?')}")
        lines.append(f"Env:        {traj.get('env_name', '?')} / {traj.get('latent_id', '?')}")
        lines.append(f"Prompt:     {traj.get('prompt_id', '?')}")
        lines.append(f"Feedback:   {traj.get('feedback_id', '?')}")
        lines.append(f"Reward:     {traj.get('reward_type', '?')}")
        lines.append(f"Seed:       {traj.get('seed', 0)}")
        lines.append(f"Episodes:   {traj.get('num_episodes', len(traj.get('episode_outcomes', [])))}")

        # ── Env params ──
        env_params = traj.get("env_params", {})
        if env_params:
            lines.append(f"\nENV PARAMS")
            for k, v in sorted(env_params.items()):
                lines.append(f"  {k}: {v}")

        # ── Agent assignments ──
        assignments = traj.get("agent_assignments", [])
        if assignments and len(set(assignments)) > 1:
            lines.append(f"\nAGENT ASSIGNMENTS")
            for i, agent in enumerate(assignments):
                lines.append(f"  Episode {i}: {agent}")

        # ── Episode table ──
        outcomes = traj.get("episode_outcomes", [])
        ep_rewards = traj.get("episode_rewards", [])
        ep_turns = traj.get("episode_turns", [])

        lines.append(f"\nEPISODE SUMMARY")
        has_agents = assignments and len(set(assignments)) > 1
        header = f"{'Ep':>3}  {'Reward':>8}  {'Turns':>6}  {'Eff':>5}  {'Outcome':<8}"
        if has_agents:
            header += f"  {'Agent':<16}"
        lines.append(header)
        lines.append("-" * len(header))

        for i, outcome in enumerate(outcomes):
            r = outcome.get("reward", ep_rewards[i] if i < len(ep_rewards) else 0)
            t = outcome.get("turns", ep_turns[i] if i < len(ep_turns) else 0)
            eff = outcome.get("turn_efficiency", 0)
            ot = outcome.get("outcome_type", "partial")
            row = f"{i:>3}  {r:>8.4f}  {t:>6}  {eff:>5.2f}  {ot:<8}"
            if has_agents:
                agent = assignments[i] if i < len(assignments) else "?"
                row += f"  {agent:<16}"
            lines.append(row)

        if ep_rewards:
            lines.append("-" * len(header))
            lines.append(f"{'Tot':>3}  {sum(ep_rewards):>8.4f}  {sum(ep_turns):>6}")
            if len(ep_rewards) >= 2:
                improvement = ep_rewards[-1] - ep_rewards[0]
                lines.append(f"Improvement: {improvement:+.4f} (last - first)")
                lines.append(f"Success rate: {traj.get('success_rate', 0):.1%}")

        # ── Ground truth per episode ──
        episode_configs = traj.get("episode_configs", [])
        if not episode_configs:
            # Fall back to ground_truth in episode_outcomes
            episode_configs = [o.get("ground_truth", {}) for o in outcomes]

        has_gt = any(bool(gt) for gt in episode_configs)
        if has_gt:
            lines.append(f"\nGROUND TRUTH PER EPISODE")
            lines.append("-" * 74)
            for i, gt in enumerate(episode_configs):
                if not gt:
                    continue
                gt_str = self._format_ground_truth(gt)
                lines.append(f"  Ep {i}: {gt_str}")

        # ── Conversation ──
        conversation = traj.get("conversation", [])
        reasoning_trace = traj.get("reasoning_trace", [])
        if conversation:
            lines.append(f"\n{'=' * 74}")
            lines.append("CONVERSATION")
            lines.append("=" * 74)
            assistant_idx = 0
            for msg in conversation:
                role = msg.get("role", "?").upper()
                content = msg.get("content", "")
                if role == "ASSISTANT":
                    # Show reasoning before the action if available
                    if assistant_idx < len(reasoning_trace) and reasoning_trace[assistant_idx]:
                        lines.append(f"\n[REASONING (internal — not shown to env)]")
                        lines.append(reasoning_trace[assistant_idx])
                    assistant_idx += 1
                lines.append(f"\n[{role}]")
                lines.append(content)

        # Reasoning summary
        has_reasoning = any(r for r in reasoning_trace if r)
        if has_reasoning:
            lines.append(f"\n{'─' * 74}")
            n_with = sum(1 for r in reasoning_trace if r)
            lines.append(f"REASONING: {n_with}/{len(reasoning_trace)} turns had reasoning tokens")

        lines.append(f"\n{'=' * 74}")
        return "\n".join(lines)

    def render_html(self, traj: Dict[str, Any]) -> str:
        """Render a trajectory as a standalone HTML page with full metadata."""
        outcomes = traj.get("episode_outcomes", [])
        ep_rewards = traj.get("episode_rewards", [])
        ep_turns = traj.get("episode_turns", [])
        conversation = traj.get("conversation", [])
        env_params = traj.get("env_params", {})
        assignments = traj.get("agent_assignments", [])
        episode_configs = traj.get("episode_configs", [])
        if not episode_configs:
            episode_configs = [o.get("ground_truth", {}) for o in outcomes]

        improvement = (ep_rewards[-1] - ep_rewards[0]) if len(ep_rewards) >= 2 else 0
        total_reward = sum(ep_rewards) if ep_rewards else 0
        has_agents = assignments and len(set(assignments)) > 1
        has_gt = any(bool(gt) for gt in episode_configs)

        # Reasoning badge (computed early for header)
        reasoning_trace = traj.get("reasoning_trace", [])
        has_reasoning = any(r for r in reasoning_trace if r)
        reasoning_badge = ""
        if has_reasoning:
            n_with = sum(1 for r in reasoning_trace if r)
            reasoning_badge = f"""
            <div class="metric-box"><div class="val">{n_with}/{len(reasoning_trace)}</div><div class="label">Reasoning</div></div>"""

        # ── Env params HTML ──
        params_html = ""
        if env_params:
            params_rows = "".join(
                f"<tr><td><code>{k}</code></td><td>{v}</td></tr>"
                for k, v in sorted(env_params.items())
            )
            params_html = f"""
            <div class="section-title">Environment Parameters</div>
            <table><tr><th>Param</th><th>Value</th></tr>{params_rows}</table>"""

        # ── Episode table HTML ──
        ep_headers = "<th>Ep</th><th>Reward</th><th>Turns</th><th>Efficiency</th><th>Outcome</th>"
        if has_agents:
            ep_headers += "<th>Agent</th>"
        if has_gt:
            ep_headers += "<th>Ground Truth</th>"

        ep_rows = ""
        for i, outcome in enumerate(outcomes):
            r = outcome.get("reward", ep_rewards[i] if i < len(ep_rewards) else 0)
            t = outcome.get("turns", ep_turns[i] if i < len(ep_turns) else 0)
            eff = outcome.get("turn_efficiency", 0)
            ot = outcome.get("outcome_type", "partial")
            color = "#d4edda" if ot == "win" else "#f8d7da"
            agent_cell = f"<td>{assignments[i] if i < len(assignments) else '?'}</td>" if has_agents else ""
            gt = episode_configs[i] if i < len(episode_configs) else {}
            gt_cell = f"<td><code>{self._format_ground_truth_short(gt)}</code></td>" if has_gt else ""
            ep_rows += f"""
            <tr style="background-color: {color}">
                <td>{i}</td><td>{r:.4f}</td><td>{t}</td><td>{eff:.2f}</td>
                <td>{ot}</td>
                {agent_cell}{gt_cell}
            </tr>"""

        # ── Ground truth detail section ──
        gt_html = ""
        if has_gt:
            gt_rows = ""
            for i, gt in enumerate(episode_configs):
                if not gt:
                    continue
                gt_escaped = _html_escape(json.dumps(gt, indent=2, default=str))
                gt_rows += f"""
                <details><summary>Episode {i}</summary>
                <pre style="font-size:0.85em; background:#f5f5f5; padding:8px; border-radius:4px;">{gt_escaped}</pre>
                </details>"""
            gt_html = f"""
            <div class="section-title">Ground Truth Details</div>
            {gt_rows}"""

        # ── Conversation HTML ──
        conv_html = ""
        assistant_idx = 0
        for msg in conversation:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if role == "assistant":
                # Show reasoning as a collapsible block before the action
                if assistant_idx < len(reasoning_trace) and reasoning_trace[assistant_idx]:
                    reasoning_escaped = _html_escape(reasoning_trace[assistant_idx])
                    conv_html += f"""
            <details style="margin:8px 0;">
                <summary style="background:#f3e5f5; border-left:4px solid #9C27B0;
                                padding:8px 10px; border-radius:4px; cursor:pointer;">
                    <strong style="font-size:0.85em; color:#6A1B9A;">🧠 REASONING</strong>
                    <span style="font-size:0.8em; color:#888;"> (internal thinking — not shown to env)</span>
                </summary>
                <div style="background:#fce4ec22; border-left:4px solid #CE93D8;
                            padding:10px; margin:0; border-radius:0 0 4px 4px;">
                    <pre style="margin:0; white-space:pre-wrap; font-size:0.85em;
                                color:#4A148C; font-family:'SF Mono','Consolas',monospace;">{reasoning_escaped}</pre>
                </div>
            </details>"""
                assistant_idx += 1
                bg, border = "#e3f2fd", "#1976D2"
            elif role == "system":
                bg, border = "#fff3e0", "#F57C00"
            else:
                bg, border = "#f5f5f5", "#9e9e9e"
            content_escaped = _html_escape(content)
            conv_html += f"""
            <div style="background:{bg}; border-left:4px solid {border};
                        margin:8px 0; padding:10px; border-radius:4px;">
                <strong style="font-size:0.85em; color:#555;">{role.upper()}</strong>
                <pre style="margin:4px 0; white-space:pre-wrap; font-size:0.9em;">{content_escaped}</pre>
            </div>"""


        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Trajectory: {traj.get('benchmark_id', '')}</title>
    <style>
        body {{ font-family: -apple-system, 'Segoe UI', monospace; max-width: 960px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
        th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9em; }}
        th {{ background: #f0f0f0; }}
        code {{ background: #f5f5f5; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
        .header {{ background: #263238; color: white; padding: 16px; border-radius: 8px; margin-bottom: 20px; }}
        .header h2 {{ margin: 0 0 10px 0; }}
        .meta {{ font-size: 0.9em; opacity: 0.85; line-height: 1.6; }}
        .metrics {{ display: flex; gap: 20px; margin-top: 10px; }}
        .metric-box {{ background: rgba(255,255,255,0.15); padding: 6px 14px; border-radius: 6px; }}
        .metric-box .val {{ font-size: 1.3em; font-weight: bold; }}
        .metric-box .label {{ font-size: 0.8em; opacity: 0.7; }}
        .section-title {{ font-size: 1.1em; font-weight: bold; margin: 24px 0 8px 0;
                          border-bottom: 2px solid #263238; padding-bottom: 4px; }}
        details {{ margin: 4px 0; }}
        summary {{ cursor: pointer; font-weight: 500; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>{traj.get('benchmark_id', 'unknown')}</h2>
        <div class="meta">
            Model: <strong>{traj.get('model_name', '?')}</strong> &nbsp;|&nbsp;
            Env: {traj.get('env_name', '?')} / {traj.get('latent_id', '?')} &nbsp;|&nbsp;
            Prompt: {traj.get('prompt_id', '?')} &nbsp;|&nbsp;
            Feedback: {traj.get('feedback_id', '?')} &nbsp;|&nbsp;
            Seed: {traj.get('seed', 0)}
        </div>
        <div class="metrics">
            <div class="metric-box"><div class="val">{total_reward:.3f}</div><div class="label">Total Reward</div></div>
            <div class="metric-box"><div class="val">{improvement:+.3f}</div><div class="label">Improvement</div></div>
            <div class="metric-box"><div class="val">{len(outcomes)}</div><div class="label">Episodes</div></div>
            {reasoning_badge}
        </div>
    </div>

    {params_html}

    <div class="section-title">Episode Summary</div>
    <table>
        <tr>{ep_headers}</tr>
        {ep_rows}
    </table>

    {gt_html}

    <div class="section-title">Conversation</div>
    {conv_html}
</body>
</html>"""

    # ── Multi-trajectory comparison ──────────────────────────────────────

    def render_comparison(
        self,
        trajectories: List[Dict[str, Any]],
        label: str = "",
    ) -> str:
        """Render multiple trajectories as a side-by-side comparison.

        Shows per-episode reward table across all trajectories, plus
        summary stats for each.
        """
        if not trajectories:
            return "No trajectories to compare."

        lines = []
        lines.append(f"{'=' * 74}")
        lines.append(f"COMPARISON: {label}")
        lines.append(f"{'=' * 74}")

        # ── Summary table ──
        lines.append(f"\n{'Model':<20}  {'Total':>8}  {'Mean':>8}  {'Impr':>8}  "
                     f"{'Success':>8}  {'Turns':>6}  {'Eps':>4}")
        lines.append("-" * 76)
        for traj in trajectories:
            model = traj.get("model_name", "?")
            ep_rewards = traj.get("episode_rewards", [])
            total = sum(ep_rewards)
            mean = total / len(ep_rewards) if ep_rewards else 0
            improvement = (ep_rewards[-1] - ep_rewards[0]) if len(ep_rewards) >= 2 else 0
            success = traj.get("success_rate", 0)
            turns = traj.get("total_turns", 0)
            eps = len(ep_rewards)
            lines.append(f"{model:<20}  {total:>8.4f}  {mean:>8.4f}  {improvement:>+8.4f}  "
                        f"{success:>8.1%}  {turns:>6}  {eps:>4}")

        # ── Per-episode comparison ──
        max_eps = max((len(t.get("episode_rewards", [])) for t in trajectories), default=0)
        if max_eps > 0:
            models = [t.get("model_name", "?")[:16] for t in trajectories]
            lines.append(f"\nPER-EPISODE REWARDS")
            header = f"{'Ep':>3}  " + "  ".join(f"{m:>16}" for m in models)
            lines.append(header)
            lines.append("-" * len(header))
            for ep in range(max_eps):
                row = f"{ep:>3}  "
                for traj in trajectories:
                    rewards = traj.get("episode_rewards", [])
                    if ep < len(rewards):
                        row += f"{rewards[ep]:>16.4f}"
                    else:
                        row += f"{'—':>16}"
                    row += "  "
                lines.append(row)

        # ── Ground truth comparison (if available) ──
        has_any_gt = False
        for traj in trajectories:
            configs = traj.get("episode_configs", [])
            outcomes = traj.get("episode_outcomes", [])
            if not configs:
                configs = [o.get("ground_truth", {}) for o in outcomes]
            if any(bool(gt) for gt in configs):
                has_any_gt = True
                break

        if has_any_gt:
            lines.append(f"\nGROUND TRUTH (first trajectory)")
            first = trajectories[0]
            configs = first.get("episode_configs", [])
            if not configs:
                configs = [o.get("ground_truth", {}) for o in first.get("episode_outcomes", [])]
            for i, gt in enumerate(configs):
                if gt:
                    lines.append(f"  Ep {i}: {self._format_ground_truth(gt)}")

        lines.append(f"\n{'=' * 74}")
        return "\n".join(lines)

    # ── Interactive multi-trajectory HTML ───────────────────────────────

    def render_interactive_html(
        self,
        trajectories: List[Dict[str, Any]],
        title: str = "Trajectory Explorer",
    ) -> str:
        """Render an interactive HTML page with filter dropdowns and trajectory navigation.

        All trajectories are embedded as JSON in the page. Dropdowns for
        model, env, latent, prompt, feedback filter client-side. Prev/Next
        buttons navigate within the filtered set. No server required.

        Args:
            trajectories: List of trajectory dicts (from to_dict() or load_trajectory())
            title: Page title

        Returns:
            Standalone HTML string with embedded JS.
        """
        # Derive env_name from benchmark_id if env_name is generic/missing
        def _derive_env(t):
            en = t.get("env_name", "")
            if en and en != "MultiEpisodeEnv":
                return en
            bid = t.get("benchmark_id", "")
            if bid and "/" in bid:
                return bid.split("/")[0]
            return en or "?"

        for t in trajectories:
            t["env_name"] = _derive_env(t)

        # Extract unique filter values
        models = sorted({t.get("model_name", "?") for t in trajectories})
        envs = sorted({t.get("env_name", "?") for t in trajectories})
        latents = sorted({t.get("latent_id", "?") for t in trajectories})
        prompts = sorted({t.get("prompt_id", "?") for t in trajectories})
        feedbacks = sorted({t.get("feedback_id", "?") for t in trajectories})

        def _options(values):
            opts = '<option value="all">All</option>\n'
            for v in values:
                opts += f'            <option value="{_html_escape(v)}">{_html_escape(v)}</option>\n'
            return opts

        # Serialize trajectories as JSON (strip conversation to reduce size,
        # keep a truncated version for display)
        embedded_trajs = []
        for t in trajectories:
            slim = {
                "model_name": t.get("model_name", "?"),
                "env_name": t.get("env_name", "?"),
                "latent_id": t.get("latent_id", "?"),
                "prompt_id": t.get("prompt_id", "?"),
                "feedback_id": t.get("feedback_id", "?"),
                "reward_type": t.get("reward_type", "?"),
                "benchmark_id": t.get("benchmark_id", ""),
                "seed": t.get("seed", 0),
                "env_params": t.get("env_params", {}),
                "agent_assignments": t.get("agent_assignments", []),
                "episode_rewards": t.get("episode_rewards", []),
                "episode_turns": t.get("episode_turns", []),
                "cumulative_reward": t.get("cumulative_reward", 0),
                "improvement": t.get("improvement", 0),
                "total_turns": t.get("total_turns", 0),
                "episode_outcomes": t.get("episode_outcomes", []),
                "episode_configs": t.get("episode_configs", []),
                "conversation": t.get("conversation", []),
                "reasoning_trace": t.get("reasoning_trace", []),
            }
            embedded_trajs.append(slim)

        trajs_json = json.dumps(embedded_trajs, default=str)

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{_html_escape(title)}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; max-width: 1000px; margin: 0 auto; padding: 16px; background: #fafafa; }}
  .filters {{ background: #263238; color: white; padding: 16px; border-radius: 8px; margin-bottom: 16px; display: flex; flex-wrap: wrap; gap: 12px; align-items: end; }}
  .filters label {{ font-size: 0.8em; opacity: 0.7; display: block; margin-bottom: 2px; }}
  .filters select {{ padding: 4px 8px; border-radius: 4px; border: none; font-size: 0.9em; min-width: 120px; }}
  .nav {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }}
  .nav button {{ padding: 6px 16px; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer; font-size: 0.9em; }}
  .nav button:hover {{ background: #e0e0e0; }}
  .nav button:disabled {{ opacity: 0.4; cursor: default; }}
  .nav .counter {{ font-size: 0.9em; color: #555; }}
  .header {{ background: #263238; color: white; padding: 14px; border-radius: 8px; margin-bottom: 12px; }}
  .header h2 {{ margin: 0 0 6px 0; font-size: 1.1em; }}
  .meta {{ font-size: 0.85em; opacity: 0.85; line-height: 1.5; }}
  .metric-row {{ display: flex; gap: 16px; margin-top: 8px; }}
  .metric-box {{ background: rgba(255,255,255,0.12); padding: 4px 12px; border-radius: 5px; text-align: center; }}
  .metric-box .val {{ font-size: 1.2em; font-weight: bold; }}
  .metric-box .lbl {{ font-size: 0.75em; opacity: 0.7; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 12px; font-size: 0.88em; }}
  th, td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .section {{ font-size: 1em; font-weight: bold; margin: 18px 0 6px 0; border-bottom: 2px solid #263238; padding-bottom: 3px; }}
  .msg {{ margin: 6px 0; padding: 8px 10px; border-radius: 4px; border-left: 4px solid #9e9e9e; background: #f5f5f5; }}
  .msg.system {{ background: #fff3e0; border-color: #F57C00; }}
  .msg.assistant {{ background: #e3f2fd; border-color: #1976D2; }}
  .msg .role {{ font-size: 0.8em; font-weight: bold; color: #555; }}
  .msg pre {{ margin: 3px 0 0 0; white-space: pre-wrap; font-size: 0.88em; font-family: 'SF Mono', 'Consolas', monospace; }}
  details {{ margin: 3px 0; }}
  summary {{ cursor: pointer; font-weight: 500; font-size: 0.9em; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 0.88em; }}
  .success {{ background-color: #d4edda; }}
  .failure {{ background-color: #f8d7da; }}
  #no-results {{ padding: 40px; text-align: center; color: #888; font-size: 1.1em; }}
  .review-box {{ background: #fff9e6; border: 1px solid #f0d070; border-radius: 6px; padding: 12px; margin: 12px 0; }}
  .review-box h4 {{ margin: 0 0 8px 0; font-size: 0.95em; }}
  .review-row {{ display: flex; gap: 14px; flex-wrap: wrap; align-items: center; margin-bottom: 8px; }}
  .review-row label {{ font-size: 0.85em; font-weight: 500; display: flex; align-items: center; gap: 4px; }}
  .review-row select, .review-row input[type="checkbox"] {{ font-size: 0.88em; }}
  .review-row select:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .review-box textarea {{ width: 100%; min-height: 50px; padding: 6px; border: 1px solid #ddd; border-radius: 4px; font-family: inherit; font-size: 0.88em; resize: vertical; }}
  .review-saved {{ color: #388E3C; font-size: 0.8em; margin-left: 8px; }}
  .other-review {{ background: #eef; border: 1px solid #ccd; border-radius: 6px; padding: 8px 10px; margin-top: 8px; font-size: 0.88em; }}
  .other-review .rname {{ font-weight: bold; color: #1565C0; }}
  .other-review .badges span {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.78em; margin: 0 4px 0 0; }}
  .badge-checked {{ background: #c8e6c9; color: #1b5e20; }}
  .badge-pri-fine {{ background: #e3f2fd; color: #0d47a1; }}
  .badge-pri-important {{ background: #fff3e0; color: #e65100; }}
  .badge-pri-critical {{ background: #ffcdd2; color: #b71c1c; }}
  .badge-bh-good {{ background: #dcedc8; color: #33691e; }}
  .badge-bh-bad {{ background: #f8bbd0; color: #880e4f; }}
</style>
</head>
<body>

<div class="filters">
  <div><label>Model</label><select id="f-model">{_options(models)}</select></div>
  <div><label>Environment</label><select id="f-env">{_options(envs)}</select></div>
  <div><label>Latent</label><select id="f-latent">{_options(latents)}</select></div>
  <div><label>Prompt</label><select id="f-prompt">{_options(prompts)}</select></div>
  <div><label>Feedback</label><select id="f-feedback">{_options(feedbacks)}</select></div>
  <div><label>Reward Op</label>
    <select id="f-reward-op"><option value="none">--</option><option value="lt">&lt;</option><option value="le">&le;</option><option value="gt">&gt;</option><option value="ge">&ge;</option><option value="eq">=</option></select>
  </div>
  <div><label>Reward Value</label>
    <input id="f-reward-val" type="number" step="any" placeholder="e.g. 0.5" style="padding:4px 8px;border-radius:4px;border:none;font-size:0.9em;width:100px;">
  </div>
  <div><label>Checked</label><select id="f-checked"><option value="all">All</option><option value="checked">Checked only</option><option value="unchecked">Unchecked only</option></select></div>
  <div><label>Priority</label><select id="f-priority"><option value="all">All</option><option value="fine">Fine</option><option value="important">Important</option><option value="critical">Critical</option></select></div>
  <div><label>Behavior</label><select id="f-behavior"><option value="all">All</option><option value="good">Good</option><option value="bad">Bad</option></select></div>
</div>

<div class="nav">
  <button id="btn-prev" onclick="navigate(-1)">&larr; Prev</button>
  <span class="counter" id="counter">0 / 0</span>
  <button id="btn-next" onclick="navigate(1)">Next &rarr;</button>
  <span style="margin-left:auto;display:flex;gap:8px;align-items:center;">
    <span id="reviewer-label" style="font-size:0.85em;color:#555;">Reviewer: <b id="reviewer-name">?</b> <a href="#" onclick="changeReviewer();return false;" style="font-size:0.85em;">(change)</a></span>
    <button onclick="exportReviews()" title="Download your reviews as reviews_&lt;name&gt;.json">Export My Reviews</button>
    <button onclick="document.getElementById('import-file').click()" title="Load someone else's reviews JSON">Import Reviews</button>
    <input id="import-file" type="file" accept=".json" style="display:none;" onchange="importReviews(event)">
    <button onclick="reloadOthers()" title="Re-fetch all reviews_*.json from the server">Reload Others</button>
  </span>
</div>

<div id="viewer"></div>
<div id="no-results" style="display:none;">No trajectories match the selected filters.</div>

<script>
const ALL_TRAJS = {trajs_json};
let filtered = [];
let idx = 0;

function esc(s) {{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

// Build env↔latent mapping for cascading filters
const ENV_TO_LATENTS = {{}};
const LATENT_TO_ENV = {{}};
ALL_TRAJS.forEach(t => {{
  if (!ENV_TO_LATENTS[t.env_name]) ENV_TO_LATENTS[t.env_name] = new Set();
  ENV_TO_LATENTS[t.env_name].add(t.latent_id);
  LATENT_TO_ENV[t.latent_id] = t.env_name;
}});

function cascadeFilters(changedId) {{
  const envSel = document.getElementById('f-env');
  const latSel = document.getElementById('f-latent');

  // If latent selected → auto-select its env
  if (changedId === 'f-latent' && latSel.value !== 'all') {{
    const parentEnv = LATENT_TO_ENV[latSel.value];
    if (parentEnv && envSel.value !== parentEnv) envSel.value = parentEnv;
  }}

  // Rebuild latent options based on env selection
  const currentLatent = latSel.value;
  const env = envSel.value;
  const allowedLatents = env === 'all'
    ? Object.keys(LATENT_TO_ENV)
    : Array.from(ENV_TO_LATENTS[env] || []);
  allowedLatents.sort();

  let html = '<option value="all">All</option>';
  for (const l of allowedLatents) {{
    html += '<option value="' + l + '">' + l + '</option>';
  }}
  latSel.innerHTML = html;
  // Preserve latent selection if still valid, else reset
  if (currentLatent !== 'all' && allowedLatents.includes(currentLatent)) {{
    latSel.value = currentLatent;
  }} else {{
    latSel.value = 'all';
  }}
}}

function trajKey(t) {{
  return (t.model_name||'') + '||' + (t.benchmark_id||'') + '||' + (t.seed!=null?t.seed:'');
}}
// Current reviewer's name (stored in localStorage)
let REVIEWER = localStorage.getItem('reviewer_name') || '';
// Other reviewers' reviews: {{reviewer_name: {{trajKey: review, ...}}}}
let OTHER_REVIEWS = {{}};

function getMyReviews() {{
  try {{
    const r = localStorage.getItem('my_reviews_' + REVIEWER);
    return r ? JSON.parse(r) : {{}};
  }} catch(e) {{ return {{}}; }}
}}
function setMyReviews(obj) {{
  try {{ localStorage.setItem('my_reviews_' + REVIEWER, JSON.stringify(obj)); return true; }}
  catch(e) {{ return false; }}
}}
function loadReview(t) {{
  const all = getMyReviews();
  return all[trajKey(t)] || {{checked: false, priority: 'fine', behavior: 'good', remarks: ''}};
}}
function saveReview(t, r) {{
  const all = getMyReviews();
  all[trajKey(t)] = r;
  return setMyReviews(all);
}}
// All reviews (mine + others) keyed by reviewer name -> {{trajKey: review}}
function getAllReviewsByReviewer() {{
  const out = Object.assign({{}}, OTHER_REVIEWS);
  if (REVIEWER) out[REVIEWER] = getMyReviews();
  return out;
}}
// Get all reviews for a given trajectory: [{{reviewer, ...review}}, ...]
function getAllReviewsForTraj(t) {{
  const k = trajKey(t);
  const out = [];
  const all = getAllReviewsByReviewer();
  for (const name in all) {{
    const r = all[name][k];
    if (r) out.push(Object.assign({{reviewer: name}}, r));
  }}
  return out;
}}
function changeReviewer() {{
  const newName = prompt('Enter your reviewer name (letters, numbers, _ only):', REVIEWER || '');
  if (!newName) return;
  const clean = newName.replace(/[^a-zA-Z0-9_]/g, '');
  if (!clean) {{ alert('Invalid name.'); return; }}
  REVIEWER = clean;
  localStorage.setItem('reviewer_name', REVIEWER);
  document.getElementById('reviewer-name').textContent = REVIEWER;
  applyFilters();
}}
function exportReviews() {{
  if (!REVIEWER) {{ alert('Set your reviewer name first.'); changeReviewer(); return; }}
  const data = getMyReviews();
  const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'reviews_' + REVIEWER + '.json';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}
function importReviews(ev) {{
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {{
    try {{
      const data = JSON.parse(e.target.result);
      // Infer reviewer name from filename: reviews_<name>.json
      const m = file.name.match(/^reviews_([^.]+)\\.json$/);
      const name = m ? m[1] : file.name.replace(/\\.json$/, '');
      OTHER_REVIEWS[name] = data;
      alert('Imported ' + Object.keys(data).length + ' reviews from ' + name);
      applyFilters();
    }} catch(err) {{
      alert('Failed to parse JSON: ' + err.message);
    }}
  }};
  reader.readAsText(file);
  ev.target.value = '';
}}
// Fetch all reviews_*.json from current directory (parses http.server's HTML index)
async function reloadOthers() {{
  OTHER_REVIEWS = {{}};
  try {{
    const resp = await fetch('.');
    const html = await resp.text();
    // Parse links from directory listing
    const matches = [...html.matchAll(/href="(reviews_[^"]+\\.json)"/g)];
    for (const m of matches) {{
      const fname = m[1];
      const rname = fname.replace(/^reviews_/, '').replace(/\\.json$/, '');
      try {{
        const r = await fetch(fname);
        if (r.ok) OTHER_REVIEWS[rname] = await r.json();
      }} catch(e) {{ /* skip */ }}
    }}
  }} catch(e) {{ /* not served via http.server */ }}
  // Don't include our own file as "other"
  if (REVIEWER && OTHER_REVIEWS[REVIEWER]) delete OTHER_REVIEWS[REVIEWER];
  applyFilters();
}}
function rewardMatches(total, op, val) {{
  if (op === 'none' || val === '' || val === null || isNaN(val)) return true;
  const v = parseFloat(val);
  if (op === 'lt') return total < v;
  if (op === 'le') return total <= v;
  if (op === 'gt') return total > v;
  if (op === 'ge') return total >= v;
  if (op === 'eq') return Math.abs(total - v) < 1e-9;
  return true;
}}

function applyFilters(e) {{
  if (e && e.target) cascadeFilters(e.target.id);
  const fm = document.getElementById('f-model').value;
  const fe = document.getElementById('f-env').value;
  const fl = document.getElementById('f-latent').value;
  const fp = document.getElementById('f-prompt').value;
  const ff = document.getElementById('f-feedback').value;
  const rop = document.getElementById('f-reward-op').value;
  const rval = document.getElementById('f-reward-val').value;
  const fc = document.getElementById('f-checked').value;
  const fpri = document.getElementById('f-priority').value;
  const fbh = document.getElementById('f-behavior').value;

  filtered = ALL_TRAJS.filter(t => {{
    if (fm !== 'all' && t.model_name !== fm) return false;
    if (fe !== 'all' && t.env_name !== fe) return false;
    if (fl !== 'all' && t.latent_id !== fl) return false;
    if (fp !== 'all' && t.prompt_id !== fp) return false;
    if (ff !== 'all' && t.feedback_id !== ff) return false;
    const total = (t.episode_rewards||[]).reduce((a,b)=>a+b, 0);
    if (!rewardMatches(total, rop, rval)) return false;
    // Collect reviews from all reviewers (including me)
    const reviews = getAllReviewsForTraj(t);
    if (fc === 'checked') {{ if (!reviews.some(r => r.checked)) return false; }}
    if (fc === 'unchecked') {{ if (reviews.some(r => r.checked)) return false; }}
    if (fpri !== 'all') {{ if (!reviews.some(r => r.checked && r.priority === fpri)) return false; }}
    if (fbh !== 'all') {{ if (!reviews.some(r => r.checked && r.behavior === fbh)) return false; }}
    return true;
  }});
  idx = 0;
  render();
}}

function navigate(dir) {{
  idx = Math.max(0, Math.min(filtered.length - 1, idx + dir));
  render();
}}

function render() {{
  const el = document.getElementById('viewer');
  const noRes = document.getElementById('no-results');
  const counter = document.getElementById('counter');
  document.getElementById('btn-prev').disabled = idx <= 0;
  document.getElementById('btn-next').disabled = idx >= filtered.length - 1;

  if (filtered.length === 0) {{
    el.innerHTML = '';
    noRes.style.display = 'block';
    counter.textContent = '0 / 0';
    return;
  }}
  noRes.style.display = 'none';
  counter.textContent = (idx + 1) + ' / ' + filtered.length;

  const t = filtered[idx];
  const er = t.episode_rewards || [];
  const et = t.episode_turns || [];
  const total = er.reduce((a,b) => a+b, 0);
  const impr = er.length >= 2 ? er[er.length-1] - er[0] : 0;
  const agents = t.agent_assignments || [];
  const multiAgent = new Set(agents).size > 1;
  const outcomes = t.episode_outcomes || [];
  const configs = t.episode_configs || [];
  const hasGT = configs.some(c => c && Object.keys(c).length > 0) ||
                outcomes.some(o => o.ground_truth && Object.keys(o.ground_truth).length > 0);

  // Header
  let html = '<div class="header">';
  html += '<h2>' + esc(t.benchmark_id || 'unknown') + '</h2>';
  html += '<div class="meta">Model: <strong>' + esc(t.model_name) + '</strong> | ';
  html += 'Env: ' + esc(t.env_name) + ' / ' + esc(t.latent_id) + ' | ';
  html += 'Prompt: ' + esc(t.prompt_id) + ' | Feedback: ' + esc(t.feedback_id) + ' | Seed: ' + t.seed + '</div>';
  html += '<div class="metric-row">';
  html += '<div class="metric-box"><div class="val">' + total.toFixed(3) + '</div><div class="lbl">Total Reward</div></div>';
  html += '<div class="metric-box"><div class="val">' + (impr >= 0 ? '+' : '') + impr.toFixed(3) + '</div><div class="lbl">Improvement</div></div>';
  html += '<div class="metric-box"><div class="val">' + er.length + '</div><div class="lbl">Episodes</div></div>';
  html += '</div></div>';

  // Review box (my review, editable)
  const rev = loadReview(t);
  const disAttr = rev.checked ? '' : 'disabled';
  const meTitle = REVIEWER ? ('Your Review (as ' + esc(REVIEWER) + ')') : 'Your Review (set name first)';
  html += '<div class="review-box"><h4>' + meTitle + ' <span id="review-saved" class="review-saved"></span></h4>';
  html += '<div class="review-row">';
  html += '<label><input type="checkbox" id="rev-checked" ' + (rev.checked?'checked':'') + ' onchange="onReviewChange()"> Checked</label>';
  html += '<label>Priority: <select id="rev-priority" ' + disAttr + ' onchange="onReviewChange()">';
  ['fine','important','critical'].forEach(v => {{
    html += '<option value="' + v + '"' + (rev.priority===v?' selected':'') + '>' + v.charAt(0).toUpperCase()+v.slice(1) + '</option>';
  }});
  html += '</select></label>';
  html += '<label>Behavior: <select id="rev-behavior" ' + disAttr + ' onchange="onReviewChange()">';
  [['good','Good'],['bad','Bad']].forEach(([v,lbl]) => {{
    html += '<option value="' + v + '"' + (rev.behavior===v?' selected':'') + '>' + lbl + '</option>';
  }});
  html += '</select></label>';
  html += '</div>';
  html += '<textarea id="rev-remarks" placeholder="Remarks..." oninput="onReviewChange()">' + esc(rev.remarks||'') + '</textarea>';
  html += '</div>';

  // Other reviewers' reviews (read-only)
  const otherReviews = getAllReviewsForTraj(t).filter(r => r.reviewer !== REVIEWER);
  if (otherReviews.length) {{
    html += '<div style="font-size:0.88em;color:#555;margin:8px 0 4px 0;">Other Reviews (' + otherReviews.length + '):</div>';
    for (const r of otherReviews) {{
      html += '<div class="other-review"><div><span class="rname">' + esc(r.reviewer) + '</span> ';
      if (r.checked) {{
        html += '<span class="badges">';
        html += '<span class="badge-checked">✓ Checked</span>';
        html += '<span class="badge-pri-' + esc(r.priority) + '">' + esc(r.priority.charAt(0).toUpperCase()+r.priority.slice(1)) + '</span>';
        html += '<span class="badge-bh-' + esc(r.behavior) + '">' + esc(r.behavior.charAt(0).toUpperCase()+r.behavior.slice(1)) + '</span>';
        html += '</span>';
      }} else {{
        html += '<span style="color:#888;font-size:0.85em;">(unchecked)</span>';
      }}
      html += '</div>';
      if (r.remarks) html += '<div style="margin-top:4px;font-style:italic;color:#444;">"' + esc(r.remarks) + '"</div>';
      html += '</div>';
    }}
  }}

  // Env params
  const ep = t.env_params || {{}};
  if (Object.keys(ep).length) {{
    html += '<div class="section">Environment Parameters</div><table><tr><th>Param</th><th>Value</th></tr>';
    for (const [k,v] of Object.entries(ep).sort()) html += '<tr><td><code>'+esc(k)+'</code></td><td>'+esc(JSON.stringify(v))+'</td></tr>';
    html += '</table>';
  }}

  // Episode table
  html += '<div class="section">Episode Summary</div><table><tr><th>Ep</th><th>Reward</th><th>Turns</th><th>Eff</th><th>Outcome</th>';
  if (multiAgent) html += '<th>Agent</th>';
  if (hasGT) html += '<th>Ground Truth</th>';
  html += '</tr>';
  for (let i = 0; i < outcomes.length; i++) {{
    const o = outcomes[i];
    const cls = (o.outcome_type === 'win') ? 'success' : 'failure';
    html += '<tr class="'+cls+'"><td>'+i+'</td><td>'+(o.reward||0).toFixed(4)+'</td><td>'+(o.turns||0)+'</td>';
    html += '<td>'+(o.turn_efficiency||0).toFixed(2)+'</td>';
    html += '<td>'+(o.outcome_type||'')+'</td>';
    if (multiAgent) html += '<td>'+(agents[i]||'?')+'</td>';
    if (hasGT) {{
      const gt = (configs[i] && Object.keys(configs[i]).length) ? configs[i] : (o.ground_truth || {{}});
      html += '<td><code>' + esc(fmtGT(gt)) + '</code></td>';
    }}
    html += '</tr>';
  }}
  html += '</table>';

  // Ground truth details
  if (hasGT) {{
    html += '<div class="section">Ground Truth Details</div>';
    for (let i = 0; i < Math.max(configs.length, outcomes.length); i++) {{
      const gt = (i < configs.length && configs[i] && Object.keys(configs[i]).length) ? configs[i] : ((i < outcomes.length) ? (outcomes[i].ground_truth || {{}}) : {{}});
      if (gt && Object.keys(gt).length) {{
        html += '<details><summary>Episode '+i+'</summary><pre style="font-size:0.85em;background:#f5f5f5;padding:8px;border-radius:4px;">'+esc(JSON.stringify(gt,null,2))+'</pre></details>';
      }}
    }}
  }}

  // Conversation (with reasoning trace)
  const conv = t.conversation || [];
  const reasoning = t.reasoning_trace || [];
  if (conv.length) {{
    const hasReasoning = reasoning.some(r => r);
    if (hasReasoning) {{
      const nWith = reasoning.filter(r => r).length;
      html += '<div class="section">Conversation <span style="font-size:0.8em;font-weight:normal;color:#9C27B0;">\U0001f9e0 ' + nWith + '/' + reasoning.length + ' turns with reasoning</span></div>';
    }} else {{
      html += '<div class="section">Conversation</div>';
    }}
    // Build episode boundary map from episode_turns data
    // assistantTurnIdx → which episode it belongs to
    // The user message AFTER the last assistant turn of an episode is the transition
    const epTurns = et; // episode_turns array
    const epBoundaries = []; // cumulative assistant turn counts where episodes end
    let cumTurns = 0;
    for (let e = 0; e < epTurns.length; e++) {{
      cumTurns += epTurns[e];
      epBoundaries.push(cumTurns); // episode e ends after this many assistant turns
    }}

    let assistantIdx = 0;
    let currentEp = 0;
    for (const msg of conv) {{
      const role = (msg.role||'?').toLowerCase();
      const content = msg.content || '';

      if (role === 'assistant') {{
        // Show reasoning before the action if available
        if (assistantIdx < reasoning.length && reasoning[assistantIdx]) {{
          html += '<details style="margin:6px 0;"><summary style="background:#f3e5f5;border-left:4px solid #9C27B0;padding:6px 10px;border-radius:4px;cursor:pointer;font-size:0.88em;">';
          html += '<strong style="color:#6A1B9A;">\U0001f9e0 REASONING</strong> <span style="color:#888;font-size:0.85em;">(internal thinking \\u2014 not shown to env)</span></summary>';
          html += '<div style="background:#f3e5f522;border-left:4px solid #CE93D8;padding:8px;"><pre style="margin:0;white-space:pre-wrap;font-size:0.83em;color:#4A148C;">' + esc(reasoning[assistantIdx]) + '</pre></div></details>';
        }}
        assistantIdx++;

        // Check if this assistant turn ends an episode
        if (currentEp < epBoundaries.length && assistantIdx === epBoundaries[currentEp]) {{
          // Render the assistant message first
          html += '<div class="msg assistant"><div class="role">ASSISTANT <span style="font-size:0.75em;color:#888;">(Ep ' + (currentEp+1) + ')</span></div><pre>'+esc(content)+'</pre></div>';
          // Episode boundary banner
          if (currentEp < epBoundaries.length - 1) {{
            html += '<div style="background:linear-gradient(135deg,#1565C0,#1976D2);color:white;text-align:center;padding:10px 12px;border-radius:6px;margin:14px 0;font-weight:bold;font-size:0.95em;letter-spacing:0.5px;">';
            html += 'Episode ' + (currentEp+1) + ' ended \\u2192 Episode ' + (currentEp+2) + '</div>';
          }} else {{
            html += '<div style="background:linear-gradient(135deg,#2E7D32,#388E3C);color:white;text-align:center;padding:10px 12px;border-radius:6px;margin:14px 0;font-weight:bold;font-size:0.95em;letter-spacing:0.5px;">';
            html += 'Episode ' + (currentEp+1) + ' ended \\u2014 Trajectory Complete</div>';
          }}
          currentEp++;
          continue; // already rendered the assistant message
        }}
      }}

      // Render message with episode tag
      const epLabel = (currentEp < epBoundaries.length) ? ' <span style="font-size:0.75em;color:#888;">(Ep ' + (currentEp+1) + ')</span>' : '';
      if (role === 'assistant') {{
        html += '<div class="msg assistant"><div class="role">ASSISTANT' + epLabel + '</div><pre>'+esc(content)+'</pre></div>';
      }} else if (role === 'system') {{
        html += '<div class="msg system"><div class="role">SYSTEM</div><pre>'+esc(content)+'</pre></div>';
      }} else {{
        // User messages right after episode boundary are transitions — style differently
        const isTransition = (currentEp > 0 && assistantIdx === epBoundaries[currentEp-1]);
        if (isTransition) {{
          html += '<div style="background:#FFF8E1;border-left:4px solid #F9A825;margin:6px 0;padding:8px 10px;border-radius:4px;">';
          html += '<div style="font-size:0.8em;font-weight:bold;color:#F57F17;">TRANSITION + NEW EPISODE' + epLabel + '</div>';
          html += '<pre style="margin:3px 0;white-space:pre-wrap;font-size:0.88em;">' + esc(content) + '</pre></div>';
        }} else {{
          html += '<div class="msg"><div class="role">USER' + epLabel + '</div><pre>'+esc(content)+'</pre></div>';
        }}
      }}
    }}
  }}

  el.innerHTML = html;
}}

function fmtGT(gt) {{
  if (!gt || !Object.keys(gt).length) return '\\u2014';
  if (gt.target_word && !gt.start_word) return gt.target_word;
  if (gt.ground_truth) {{
    const p = gt.ground_truth;
    if (typeof p === 'object') {{
      let best = '', bestV = -1;
      for (const [k,v] of Object.entries(p)) {{ if (v > bestV) {{ best=k; bestV=v; }} }}
      return best + '=' + bestV.toFixed(2);
    }}
  }}
  if (gt.secret_code) return JSON.stringify(gt.secret_code);
  if (gt.draws) {{
    const d = gt.draws;
    const mx = d.indexOf(Math.max(...d));
    return 'max@' + mx;
  }}
  if (gt.start_word) return gt.start_word + '\\u2192' + (gt.target_word||'?');
  return JSON.stringify(gt).substring(0, 40);
}}

function onReviewChange() {{
  if (!filtered[idx]) return;
  const t = filtered[idx];
  const checked = document.getElementById('rev-checked').checked;
  const priSel = document.getElementById('rev-priority');
  const bhSel = document.getElementById('rev-behavior');
  priSel.disabled = !checked;
  bhSel.disabled = !checked;
  const r = {{
    checked: checked,
    priority: priSel.value,
    behavior: bhSel.value,
    remarks: document.getElementById('rev-remarks').value
  }};
  const ok = saveReview(t, r);
  const saved = document.getElementById('review-saved');
  if (saved) {{ saved.textContent = ok ? '✓ Saved' : '⚠ Save failed'; setTimeout(() => {{ if(saved) saved.textContent = ''; }}, 1500); }}
}}

// Wire up filters
['f-model','f-env','f-latent','f-prompt','f-feedback','f-reward-op','f-reward-val','f-checked','f-priority','f-behavior'].forEach(id => {{
  const el = document.getElementById(id);
  if (el) el.addEventListener(el.tagName === 'INPUT' ? 'input' : 'change', applyFilters);
}});

// Init reviewer name
if (!REVIEWER) {{
  const name = prompt('Enter your reviewer name (letters, numbers, _ only):\\nThis is saved locally in your browser.', '');
  if (name) {{
    REVIEWER = name.replace(/[^a-zA-Z0-9_]/g, '') || 'anonymous';
    localStorage.setItem('reviewer_name', REVIEWER);
  }} else {{
    REVIEWER = 'anonymous';
  }}
}}
document.getElementById('reviewer-name').textContent = REVIEWER;

// Load other reviewers' files from the server directory, then render
reloadOthers();
</script>
</body>
</html>"""

    def render_interactive_html_from_dir(
        self,
        data_dir: str,
        title: str = "Trajectory Explorer",
        max_trajectories: int = 10000,
    ) -> str:
        """Load all trajectories from a DataStore directory and render interactive HTML.

        Args:
            data_dir: DataStore output directory
            title: Page title
            max_trajectories: Safety limit on number of trajectories to embed

        Returns:
            Standalone HTML string
        """
        trajs = self.browse(data_dir, max_results=max_trajectories)
        return self.render_interactive_html(trajs, title=title)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _format_ground_truth(self, gt: Dict[str, Any]) -> str:
        """Format ground truth dict for text display."""
        if not gt:
            return "—"
        # Env-specific formatting
        if "target_word" in gt:
            return f"target_word={gt['target_word']}"
        if "ground_truth" in gt:
            probs = gt["ground_truth"]
            if isinstance(probs, dict):
                best = max(probs, key=probs.get)
                return f"best={best} ({probs[best]:.2f}), probs={{{', '.join(f'{k}:{v:.2f}' for k, v in probs.items())}}}"
        if "secret_code" in gt:
            return f"secret_code={gt['secret_code']}"
        if "draws" in gt:
            draws = gt["draws"]
            max_idx = draws.index(max(draws)) if draws else -1
            return f"max_at={max_idx}, draws=[{', '.join(f'{d:.3f}' for d in draws[:5])}{'...' if len(draws) > 5 else ''}]"
        if "start_word" in gt:
            return f"{gt['start_word']} → {gt.get('target_word', '?')}"
        # Generic fallback
        return str(gt)[:120]

    def _format_ground_truth_short(self, gt: Dict[str, Any]) -> str:
        """Short ground truth for HTML table cells."""
        if not gt:
            return "—"
        if "target_word" in gt and "start_word" not in gt:
            return gt["target_word"]
        if "ground_truth" in gt:
            probs = gt["ground_truth"]
            if isinstance(probs, dict):
                best = max(probs, key=probs.get)
                return f"{best}={probs[best]:.2f}"
        if "secret_code" in gt:
            return str(gt["secret_code"])
        if "draws" in gt:
            draws = gt["draws"]
            max_idx = draws.index(max(draws)) if draws else -1
            return f"max@{max_idx}"
        if "start_word" in gt:
            return f"{gt['start_word']}→{gt.get('target_word', '?')}"
        return str(gt)[:40]

    @staticmethod
    def _load_json(path: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
