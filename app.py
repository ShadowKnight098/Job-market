"""
Gradio UI for Job Market Strategy RL Agent.
Entry point for Hugging Face Spaces.
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import numpy as np
import random
import pandas as pd

from job_market_env import (
    JobMarketEnv, ACTIONS, SKILL_CURRICULUM,
    get_skill_recommendations
)
from agent import QLearningAgent
from train import train

MODEL_PATH = "models/qtable.pkl"

# ── Load or train agent ────────────────────────────────────────────────

def load_agent() -> QLearningAgent:
    agent = QLearningAgent(epsilon=0.0)
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
    else:
        print("⚠️ Model not found, using untrained agent")
    agent.epsilon = 0.0
    return agent

agent = load_agent()

DIFFICULTY_MAP = {"Startup 🏠": 0, "Mid-size 🏢": 1, "FAANG 🏦": 2}
ACTION_EMOJI   = {0: "📨 Apply small", 1: "🚀 Apply big", 2: "📚 Upskill"}


# ── Format skill recommendations as markdown ───────────────────────────

def format_recommendations(recs: dict | None, rejection_reason: str | None, step: int) -> str:
    if recs is None:
        return ""
    lines = [f"### Step {step} — Feedback & Recommendations\n"]
    if rejection_reason:
        lines.append("**❌ Why you were rejected:**")
        lines.append(f"> {rejection_reason}\n")
    if recs.get("priority"):
        lines.append("**📌 What to work on next:**")
        for item in recs["priority"]:
            lines.append(f"- {item}")
        lines.append("")
    if recs.get("resources"):
        lines.append("**🔗 Resources:**")
        for r in recs["resources"]:
            lines.append(f"- {r}")
        lines.append("")
    if recs.get("timeline"):
        lines.append(f"**⏱ Estimated prep time:** {recs['timeline']}")
    return "\n".join(lines)


# ── Main episode runner ────────────────────────────────────────────────

def run_episode(skills: int, apps_left: int, difficulty_label: str, agent_mode: str):
    difficulty = DIFFICULTY_MAP[difficulty_label]
    env   = JobMarketEnv()
    state = env.reset()
    state.skills_level       = skills
    state.applications_left  = apps_left
    state.company_difficulty = difficulty

    log_rows     = []
    feedback_log = []
    total_reward = 0.0
    step = 0

    while not state.done:
        step += 1
        s_tuple = state.to_tuple()

        if agent_mode == "🤖 Trained Q-Agent":
            action = agent.choose_action(s_tuple)
        elif agent_mode == "🎲 Random Agent":
            action = random.randint(0, 2)
        else:
            s, a, d = s_tuple
            action = 2 if (s < 3 and a > 5) else (1 if s >= 4 and d == 2 else 0)

        state, reward, done, info = env.step(action)
        total_reward += reward

        msg = info.get("message", "")
        if "🌟" in msg:    result = "🌟 Big shortlist"
        elif "✅" in msg:  result = "✅ Shortlisted"
        elif "❌" in msg:  result = "❌ Rejected"
        elif "📚" in msg:  result = "📚 Upskilled"
        else:              result = "⚠️ Wasted"

        log_rows.append({
            "Step":      step,
            "Action":    ACTION_EMOJI[action],
            "Result":    result,
            "Reward":    f"{reward:+.1f}",
            "Skills":    f"{state.skills_level}/5",
            "Apps Left": state.applications_left,
        })

        rejection = info.get("rejection_reason")
        recs      = info.get("skill_recommendations")
        if rejection or (action == 2 and recs):
            feedback_log.append(format_recommendations(recs, rejection, step))

    # Summary
    outcome_bar = "🟢" * state.shortlisted_count + "🔴" * state.wasted_count + "🔵" * state.upskill_count
    summary = (
        f"## Episode Complete\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Total Reward | `{total_reward:+.2f}` |\n"
        f"| Shortlisted | {state.shortlisted_count} |\n"
        f"| Failed Attempts | {state.wasted_count} |\n"
        f"| Upskills | {state.upskill_count} |\n"
        f"| Final Skill Level | {state.skills_level}/5 |\n\n"
        f"🟢 shortlist &nbsp; 🔴 reject &nbsp; 🔵 upskill\n\n{outcome_bar}"
    )

    df = pd.DataFrame(log_rows)
    combined_feedback = "\n\n---\n\n".join(feedback_log) if feedback_log else "_No rejections this episode!_ 🎉"
    return df, summary, combined_feedback


# ── Skill gap analyser ─────────────────────────────────────────────────

def analyse_skill_gap(skills: int, difficulty_label: str):
    difficulty = DIFFICULTY_MAP[difficulty_label]
    curriculum = SKILL_CURRICULUM[difficulty]
    recs       = get_skill_recommendations(difficulty, skills)
    required   = curriculum["required_skills"]
    nice       = curriculum["nice_to_have"]
    min_lvl    = curriculum["min_skill_level"]
    gap        = max(0, min_lvl - skills)

    lines = [f"## Skill Gap Analysis — {curriculum['name']}\n",
             f"**Your level:** {skills}/5 &nbsp;&nbsp; **Required minimum:** {min_lvl}/5 &nbsp;&nbsp; **Gap:** {gap} level(s)\n"]
    skill_bar = "█" * skills + "░" * (5 - skills)
    req_bar   = "█" * min_lvl + "░" * (5 - min_lvl)
    lines.append(f"```\nYours:    [{skill_bar}] {skills}/5\nRequired: [{req_bar}] {min_lvl}/5\n```\n")
    lines.append("### ✅ Required skills")
    for i, s in enumerate(required):
        check = "✔️" if skills > i else "❌"
        lines.append(f"- {check} {s}")
    lines.append("\n### ⭐ Nice to have")
    for s in nice:
        lines.append(f"- ✨ {s}")
    lines.append("\n### 📌 Your action plan")
    for p in recs.get("priority", []):
        lines.append(f"- {p}")
    lines.append("\n### 🔗 Resources")
    for r in recs.get("resources", []):
        lines.append(f"- {r}")
    lines.append(f"\n### ⏱ Timeline\n{recs.get('timeline', 'Varies')}")
    return "\n".join(lines)


# ── Q-value inspector ─────────────────────────────────────────────────

def show_q_values(skills: int, apps: int, difficulty_label: str):
    difficulty   = DIFFICULTY_MAP[difficulty_label]
    q_vals       = agent.q_table[skills, apps, difficulty]
    best         = int(np.argmax(q_vals))
    action_names = ["apply_small_company", "apply_big_company", "upskill"]
    bars         = ["▁","▂","▃","▄","▅","▆","▇","█"]
    max_q        = max(abs(q) for q in q_vals) or 1
    lines        = [f"### Q-Values — skills={skills}, apps={apps}, market={difficulty_label}\n"]
    for i, (name, val) in enumerate(zip(action_names, q_vals)):
        norm   = int((val / max_q + 1) / 2 * 7)
        bar    = bars[max(0, min(norm, 7))]
        marker = " ← **best**" if i == best else ""
        lines.append(f"- `{name}`: {bar} `{val:.4f}`{marker}")
    lines.append(f"\n> Agent will choose: **{action_names[best]}**")
    return "\n".join(lines)

def make_card(rejection_reason, step):
    return f"""
    <div style="
        border:1px solid #e5e7eb;
        border-left:5px solid #f97316;
        border-radius:10px;
        padding:15px;
        margin-bottom:12px;
        background:#fff7ed;
    ">
        <b>Step {step} — ❌ Rejected</b><br><br>
        {rejection_reason}
    </div>
    """
# ── Gradio UI ──────────────────────────────────────────────────────────

with gr.Blocks(title="Job Application RL Agent", theme=gr.themes.Soft(primary_hue='orange')) as demo:

    gr.Markdown("""
    #  Job Application Strategy — RL Agent
    **OpenEnv |&nbsp; `JobMarketStrategy-v1`

    An RL agent that learns *when* to apply, *when* to upskill — with **real rejection reasons** and **personalised skill recommendations** at every step.
    """)

    with gr.Tabs():

        # ── Tab 1: Run Episode ──────────────────────────────────────────
        with gr.Tab("▶ Run Episode"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Setup")
                    skills_slider = gr.Slider(0, 5, value=2, step=1, label="Starting Skill Level")
                    apps_slider   = gr.Slider(1, 10, value=10, step=1, label="Applications Available")
                    diff_radio    = gr.Radio(list(DIFFICULTY_MAP.keys()), value="Mid-size 🏢", label="Market")
                    agent_radio   = gr.Radio(
                        [" Trained Q-Agent", " Random Agent", " Heuristic Agent"],
                        value="🤖 Trained Q-Agent", label="Agent"
                    )
                    run_btn = gr.Button("▶ Run Episode", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Decision Log")
                    log_table  = gr.Dataframe(
                        headers=["Step", "Action", "Result", "Reward", "Skills", "Apps Left"],
                        interactive=False,
                    )
                    summary_md = gr.Markdown()

            gr.Markdown("---")
            gr.Markdown("### 🔍 Rejection Reasons & Skill Recommendations")
            gr.Markdown("_Each rejection comes with a specific reason and a targeted learning plan._")
            feedback_md = gr.Markdown(value="_Run an episode above to see personalised feedback._")

            run_btn.click(
                fn=run_episode,
                inputs=[skills_slider, apps_slider, diff_radio, agent_radio],
                outputs=[log_table, summary_md, feedback_md],
            )

        # ── Tab 2: Skill Gap Analyser ───────────────────────────────────
        with gr.Tab("📊 Skill Gap Analyser"):
            gr.Markdown("### Know exactly what's missing for your target company")
            with gr.Row():
                gap_skills = gr.Slider(0, 5, value=2, step=1, label="Your Skill Level")
                gap_diff   = gr.Radio(list(DIFFICULTY_MAP.keys()), value="FAANG 🏦", label="Target")
            gap_btn    = gr.Button("Analyse My Gap", variant="primary")
            gap_output = gr.Markdown()

            gap_btn.click(fn=analyse_skill_gap, inputs=[gap_skills, gap_diff], outputs=[gap_output])
            demo.load(fn=analyse_skill_gap, inputs=[gap_skills, gap_diff], outputs=[gap_output])

        # ── Tab 3: Q-Value Inspector ────────────────────────────────────
        with gr.Tab("🔬 Q-Value Inspector"):
            gr.Markdown("### See what the trained agent thinks about any state")
            with gr.Row():
                q_skills = gr.Slider(0, 5, value=2, step=1, label="Skills")
                q_apps   = gr.Slider(0, 10, value=5, step=1, label="Apps Left")
                q_diff   = gr.Radio(list(DIFFICULTY_MAP.keys()), value="Mid-size 🏢", label="Market")
            q_btn    = gr.Button("Show Q-Values", variant="secondary")
            q_output = gr.Markdown()
            q_btn.click(fn=show_q_values, inputs=[q_skills, q_apps, q_diff], outputs=[q_output])

    gr.Markdown("---\n**Environment:** `JobMarketStrategy-v1` | **Algorithm:** Q-Learning | **Hackathon:** OpenEnv × Scaler")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)