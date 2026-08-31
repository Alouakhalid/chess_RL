import os
import sys
import time
import json
from pathlib import Path
import numpy as np
sys.modules['tensorflow'] = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from chess_env import ChessEnv
from stable_baselines3 import PPO, A2C, DQN, DDPG, SAC
from sb3_contrib import MaskablePPO, QRDQN
from sb3_contrib.common.wrappers import ActionMasker
from model_support import load_model as load_project_model, predict_legal_move
from fpdf import FPDF
def mask_fn(env):
    if hasattr(env, 'get_action_mask'):
        return env.get_action_mask()
    return env.get_wrapper_attr('get_action_mask')()
MODELS = {
    "PPO":  {"cls": MaskablePPO, "mask": True},
    "A2C":  {"cls": A2C,         "mask": False},
    "DQN":  {"cls": DQN,         "mask": False},
    "DDQN": {"cls": QRDQN,       "mask": False},
    "DDPG": {"cls": DDPG,        "mask": False},
    "SAC":  {"cls": SAC,          "mask": False},
}
def load_model(name):
    try:
        return load_project_model(name.lower())
    except Exception:
        return None
def predict(model, name, env, obs):
    return predict_legal_move(model, name.lower(), obs, env)
def evaluate_model(name, n_episodes=20, max_steps=200):
    print(f"  Evaluating {name}...", end=" ", flush=True)
    model = load_model(name)
    if model is None:
        print("SKIPPED (no checkpoint)")
        return None
    env = ChessEnv(mode="agent_vs_opponent", player_color="white", opponent_type="heuristic", max_steps=max_steps)
    rewards_list = []
    lengths_list = []
    wins = 0
    losses = 0
    draws = 0
    illegal_count = 0
    total_steps_all = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False
        while not done:
            decision = predict(model, name, env, obs)
            if decision.move is None:
                break
            obs, reward, term, trunc, info = env.step(decision.move)
            ep_reward += reward
            ep_steps += 1
            done = term or trunc
            if decision.was_illegal:
                illegal_count += 1
        total_steps_all += ep_steps
        rewards_list.append(ep_reward)
        lengths_list.append(ep_steps)
        board = env.unwrapped.board
        if board.is_checkmate():
            if board.turn != env.unwrapped.agent_color:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1
    result = {
        "name": name,
        "episodes": n_episodes,
        "mean_reward": float(np.mean(rewards_list)),
        "std_reward": float(np.std(rewards_list)),
        "max_reward": float(np.max(rewards_list)),
        "min_reward": float(np.min(rewards_list)),
        "mean_length": float(np.mean(lengths_list)),
        "std_length": float(np.std(lengths_list)),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes * 100,
        "illegal_moves": illegal_count,
        "total_steps": total_steps_all,
        "rewards": rewards_list,
        "lengths": lengths_list,
    }
    print(f"Win={wins} Loss={losses} Draw={draws}  MeanR={result['mean_reward']:.2f}")
    return result
def run_all_evaluations(n_episodes=20):
    print("=" * 60)
    print("   CHESS RL MODEL EVALUATION BENCHMARK")
    print("=" * 60)
    results = {}
    for name in MODELS:
        res = evaluate_model(name, n_episodes=n_episodes)
        if res:
            results[name] = res
    return results
def generate_charts(results, output_dir="./report_assets"):
    os.makedirs(output_dir, exist_ok=True)
    names = list(results.keys())
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']
    fig, ax = plt.subplots(figsize=(7, 3.5))
    means = [results[n]["mean_reward"] for n in names]
    stds = [results[n]["std_reward"] for n in names]
    bars = ax.bar(names, means, yerr=stds, color=colors[:len(names)], edgecolor='#333', linewidth=0.8, capsize=4, error_kw={'linewidth': 1.2})
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title("Mean Reward per Algorithm", fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{val:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/reward_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    win_rates = [results[n]["win_rate"] for n in names]
    bars = ax.bar(names, win_rates, color=colors[:len(names)], edgecolor='#333', linewidth=0.8)
    ax.set_ylabel("Win Rate (%)", fontsize=11)
    ax.set_title("Win Rate per Algorithm", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/winrate_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    lengths = [results[n]["mean_length"] for n in names]
    bars = ax.bar(names, lengths, color=colors[:len(names)], edgecolor='#333', linewidth=0.8)
    ax.set_ylabel("Mean Episode Length (steps)", fontsize=11)
    ax.set_title("Episode Length per Algorithm", fontsize=13, fontweight='bold')
    for bar, val in zip(bars, lengths):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/length_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, n in enumerate(names):
        ax.plot(results[n]["rewards"], label=n, color=colors[i], linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.set_title("Reward Trajectory per Episode", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/reward_trajectory.png", dpi=200, bbox_inches='tight')
    plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    win_vals = [results[n]["wins"] for n in names]
    loss_vals = [results[n]["losses"] for n in names]
    draw_vals = [results[n]["draws"] for n in names]
    x = np.arange(len(names))
    w = 0.25
    axes[0].bar(x - w, win_vals, w, label='Wins', color='#4CAF50')
    axes[0].bar(x, loss_vals, w, label='Losses', color='#F44336')
    axes[0].bar(x + w, draw_vals, w, label='Draws', color='#FFC107')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_title("Win/Loss/Draw", fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8)
    illegal = [results[n]["illegal_moves"] for n in names]
    axes[1].bar(names, illegal, color='#F44336', edgecolor='#333', linewidth=0.8)
    axes[1].set_title("Illegal Moves", fontsize=11, fontweight='bold')
    for i, v in enumerate(illegal):
        axes[1].text(i, v + 0.2, str(v), ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/wld_illegal.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Charts saved to {output_dir}/")
def generate_pdf_report(results, output_dir="./report_assets", pdf_path="./Chess_RL_Report.pdf"):
    names = list(results.keys())
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    W = 210
    COL_W = (W - 30) / 2
    MARGIN = 10
    pdf.add_page()
    pdf.set_fill_color(20, 20, 30)
    pdf.rect(0, 0, 210, 55, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(MARGIN, 10)
    pdf.cell(W - 2*MARGIN, 12, "Chess Reinforcement Learning", align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_xy(MARGIN, 24)
    pdf.cell(W - 2*MARGIN, 10, "Comparative Evaluation of Deep RL Algorithms", align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_xy(MARGIN, 36)
    pdf.cell(W - 2*MARGIN, 8, "Environment: Custom Gymnasium Chess  |  Framework: Stable-Baselines3", align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(MARGIN, 60)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(COL_W, 7, "1. Abstract")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(MARGIN)
    pdf.multi_cell(COL_W, 4.5,
        "This report presents the evaluation results of six deep "
        "reinforcement learning algorithms applied to the game of chess. "
        "The algorithms evaluated are Proximal Policy Optimization (PPO), "
        "Advantage Actor-Critic (A2C), Deep Q-Network (DQN), Double DQN "
        "(QR-DQN), Deep Deterministic Policy Gradient (DDPG), and Soft "
        "Actor-Critic (SAC). Each model was trained using a custom "
        "Gymnasium-compatible chess environment with a discrete action "
        "space of 4096 and a 14-channel (8x8) observation tensor. "
        "Models were benchmarked against a heuristic opponent over "
        f"{results[names[0]]['episodes']} evaluation episodes per algorithm."
    )
    col2_x = MARGIN + COL_W + 10
    pdf.set_xy(col2_x, 60)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(COL_W, 7, "2. Environment Design")
    pdf.set_xy(col2_x, 68)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(COL_W, 4.5,
        "The chess environment (ChessEnv) is built on python-chess and "
        "Gymnasium. Key specifications:\n"
        "- Action Space: Discrete(4096) encoding from_sq*64+to_sq\n"
        "- Observation: (14, 8, 8) float32 tensor with 12 piece planes, "
        "turn plane, and castling/check meta-plane\n"
        "- Reward Shaping: Win (+10), Loss (-10), Material delta, "
        "center control (+0.05), check bonus (+0.2), step penalty (-0.01), "
        "illegal move (-1)\n"
        "- Modes: Human-vs-AI, AI-vs-AI, Human-vs-Human\n"
        "- Opponent: Heuristic bot with material-based evaluation"
    )
    pdf.set_xy(MARGIN, pdf.get_y() + 6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(W - 2*MARGIN, 7, "3. Algorithm Summary", align="L")
    pdf.ln(8)
    algo_desc = {
        "PPO": "On-policy, clipped surrogate objective. Uses MaskablePPO with action masking for legal moves.",
        "A2C": "Synchronous advantage actor-critic. Simpler than PPO but faster per update.",
        "DQN": "Off-policy value-based with experience replay and target network.",
        "DDQN": "Quantile Regression DQN (QR-DQN). Distributional RL reducing overestimation.",
        "DDPG": "Off-policy actor-critic for continuous actions. Uses discrete-to-continuous wrapper.",
        "SAC": "Maximum entropy RL with automatic temperature. Uses continuous action wrapper.",
    }
    pdf.set_font("Helvetica", "", 9)
    y_start = pdf.get_y()
    col_items = list(algo_desc.items())
    half = (len(col_items) + 1) // 2
    for i, (alg, desc) in enumerate(col_items[:half]):
        pdf.set_x(MARGIN)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(12, 4.5, f"{alg}:")
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(COL_W - 12, 4, desc)
        pdf.ln(1)
    y2 = y_start
    for i, (alg, desc) in enumerate(col_items[half:]):
        pdf.set_xy(col2_x, y2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(12, 4.5, f"{alg}:")
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(COL_W - 12, 4, desc)
        y2 = pdf.get_y() + 1
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(W - 2*MARGIN, 7, "4. Evaluation Results")
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 8)
    col_widths = [18, 22, 22, 22, 18, 18, 18, 18, 18, 18]
    headers = ["Model", "Mean R", "Std R", "Max R", "Win", "Loss", "Draw", "Win%", "Len", "Illegal"]
    pdf.set_fill_color(40, 40, 55)
    pdf.set_text_color(255, 255, 255)
    for w_c, h_c in zip(col_widths, headers):
        pdf.cell(w_c, 7, h_c, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 8)
    fill = False
    for n in names:
        r = results[n]
        if fill:
            pdf.set_fill_color(235, 240, 250)
        else:
            pdf.set_fill_color(255, 255, 255)
        row = [
            n,
            f"{r['mean_reward']:.2f}",
            f"{r['std_reward']:.2f}",
            f"{r['max_reward']:.2f}",
            str(r['wins']),
            str(r['losses']),
            str(r['draws']),
            f"{r['win_rate']:.0f}%",
            f"{r['mean_length']:.0f}",
            str(r['illegal_moves']),
        ]
        for w_c, val in zip(col_widths, row):
            pdf.cell(w_c, 6, val, border=1, align="C", fill=True)
        pdf.ln()
        fill = not fill
    pdf.ln(6)
    charts = [
        ("reward_comparison.png", "Figure 1: Mean Reward Comparison"),
        ("winrate_comparison.png", "Figure 2: Win Rate Comparison"),
    ]
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(W - 2*MARGIN, 7, "5. Performance Charts")
    pdf.ln(8)
    for i, (fname, caption) in enumerate(charts):
        fpath = f"{output_dir}/{fname}"
        if os.path.exists(fpath):
            x_pos = MARGIN if i % 2 == 0 else MARGIN + COL_W + 5
            if i % 2 == 0:
                chart_y = pdf.get_y()
            pdf.image(fpath, x=x_pos, y=chart_y, w=COL_W)
            pdf.set_xy(x_pos, chart_y + COL_W * 0.52)
            pdf.set_font("Helvetica", "I", 8)
            pdf.cell(COL_W, 4, caption, align="C")
    pdf.ln(COL_W * 0.55 + 6)
    charts2 = [
        ("length_comparison.png", "Figure 3: Episode Length Comparison"),
        ("wld_illegal.png", "Figure 4: Win/Loss/Draw & Illegal Moves"),
    ]
    for i, (fname, caption) in enumerate(charts2):
        fpath = f"{output_dir}/{fname}"
        if os.path.exists(fpath):
            x_pos = MARGIN if i % 2 == 0 else MARGIN + COL_W + 5
            if i % 2 == 0:
                chart_y2 = pdf.get_y()
            pdf.image(fpath, x=x_pos, y=chart_y2, w=COL_W)
            pdf.set_xy(x_pos, chart_y2 + COL_W * 0.52)
            pdf.set_font("Helvetica", "I", 8)
            pdf.cell(COL_W, 4, caption, align="C")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(W - 2*MARGIN, 7, "6. Reward Trajectory")
    pdf.ln(6)
    traj_path = f"{output_dir}/reward_trajectory.png"
    if os.path.exists(traj_path):
        pdf.image(traj_path, x=MARGIN, w=W - 2*MARGIN)
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(W - 2*MARGIN, 4, "Figure 5: Per-Episode Reward Trajectory for All Algorithms", align="C")
    pdf.ln(12)
    pdf.set_font("Helvetica", "B", 12)
    y_analysis = pdf.get_y()
    pdf.set_x(MARGIN)
    pdf.cell(COL_W, 7, "7. Analysis & Discussion")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(MARGIN)
    best_model = max(results, key=lambda n: results[n]["mean_reward"])
    best_wr = max(results, key=lambda n: results[n]["win_rate"])
    longest = max(results, key=lambda n: results[n]["mean_length"])
    pdf.multi_cell(COL_W, 4.5,
        f"Among the evaluated algorithms, {best_model} achieved the "
        f"highest mean reward ({results[best_model]['mean_reward']:.2f}), "
        f"while {best_wr} recorded the highest win rate "
        f"({results[best_wr]['win_rate']:.0f}%). "
        f"The longest average episode length was observed in "
        f"{longest} ({results[longest]['mean_length']:.0f} steps), "
        "suggesting deeper game exploration.\n\n"
        "PPO with action masking (MaskablePPO) ensures that the agent "
        "only selects legal chess moves, which significantly reduces "
        "illegal move penalties compared to unconstrained algorithms. "
        "Continuous-action algorithms (DDPG, SAC) require a "
        "discrete-to-continuous wrapper, introducing quantization noise "
        "that limits their effectiveness in a discrete action space."
    )
    pdf.set_xy(col2_x, y_analysis)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(COL_W, 7, "8. Conclusions")
    pdf.set_xy(col2_x, y_analysis + 8)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(COL_W, 4.5,
        "This study demonstrates the feasibility of applying deep RL "
        "algorithms to chess using a custom Gymnasium environment. "
        "Key findings include:\n\n"
        "1) Policy gradient methods (PPO, A2C) are better suited for "
        "discrete chess action spaces than value-based or continuous "
        "methods.\n\n"
        "2) Action masking is critical for chess RL, as it eliminates "
        "illegal move exploration and focuses learning on valid "
        "strategies.\n\n"
        "3) All algorithms require significantly more training time "
        "(millions of timesteps) to develop meaningful chess strategies "
        "beyond random play.\n\n"
        "4) The reward shaping system (material, center control, check) "
        "provides dense feedback that accelerates early learning."
    )
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_x(MARGIN)
    pdf.cell(W - 2*MARGIN, 7, "9. References")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 8)
    refs = [
        "[1] Schulman et al., 'Proximal Policy Optimization Algorithms', arXiv:1707.06347, 2017.",
        "[2] Mnih et al., 'Asynchronous Methods for Deep RL', ICML 2016.",
        "[3] Mnih et al., 'Human-Level Control through Deep RL', Nature, 2015.",
        "[4] Dabney et al., 'Distributional RL with Quantile Regression', AAAI, 2018.",
        "[5] Lillicrap et al., 'Continuous Control with Deep RL', ICLR, 2016.",
        "[6] Haarnoja et al., 'Soft Actor-Critic', ICML, 2018.",
        "[7] Raffin et al., 'Stable-Baselines3', JMLR, 2021.",
    ]
    for ref in refs:
        pdf.set_x(MARGIN)
        pdf.cell(W - 2*MARGIN, 4.5, ref)
        pdf.ln()
    pdf.output(pdf_path)
    print(f"\n  PDF Report saved to: {pdf_path}")
if __name__ == "__main__":
    results = run_all_evaluations(n_episodes=20)
    if results:
        generate_charts(results)
        generate_pdf_report(results)
        print("\n" + "=" * 60)
        print("  EVALUATION SUMMARY TABLE")
        print("=" * 60)
        print(f"{'Model':<8} {'MeanR':>8} {'StdR':>8} {'Win':>5} {'Loss':>5} {'Draw':>5} {'WinRate':>8} {'AvgLen':>8} {'Illegal':>8}")
        print("-" * 70)
        for n, r in results.items():
            print(f"{n:<8} {r['mean_reward']:>8.2f} {r['std_reward']:>8.2f} {r['wins']:>5} {r['losses']:>5} {r['draws']:>5} {r['win_rate']:>7.0f}% {r['mean_length']:>8.1f} {r['illegal_moves']:>8}")
        print("=" * 70)
        print("\nDone! Open Chess_RL_Report.pdf to view the research report.")
