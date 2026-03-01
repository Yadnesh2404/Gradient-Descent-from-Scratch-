# Gradient Descent Comparison: Batch GD vs Stochastic GD
# Animated matplotlib charts showing loss, step sizes, and R² score

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from batch_gadient_descent_from_scratch import GD as BatchGD
from stochastic_gadient_descent_from_scratch import SGD

# ── Data ─────────────────────────────────────────────────────────────────────
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# ── Train ─────────────────────────────────────────────────────────────────────
EPOCHS = 100

bgd = BatchGD(learning_rate=0.1, epochs=EPOCHS)
bgd.fit(X_train, y_train)
bgd_r2 = r2_score(y_test, bgd.predict(X_test))

np.random.seed(42)
sgd = SGD(learning_rate=0.01, epochs=EPOCHS)
sgd.fit(X_train, y_train)
sgd_r2 = r2_score(y_test, sgd.predict(X_test))

print(f"Batch GD      → R²: {bgd_r2:.4f} | Final Loss: {bgd.loss_history[-1]:.2f}")
print(f"Stochastic GD → R²: {sgd_r2:.4f} | Final Loss: {sgd.loss_history[-1]:.2f}")

# ── Pre-compute step sizes (|ΔLoss| per epoch) ───────────────────────────────
bgd_steps = [abs(bgd.loss_history[i] - bgd.loss_history[i - 1]) for i in range(1, EPOCHS)]
sgd_steps = [abs(sgd.loss_history[i] - sgd.loss_history[i - 1]) for i in range(1, EPOCHS)]

# ── Dark theme setup ──────────────────────────────────────────────────────────
BG_DARK  = '#0d0d1a'
BG_PANEL = '#12122a'
C_BGD    = '#00d4ff'   # cyan  → Batch GD
C_SGD    = '#ff6b6b'   # coral → SGD
C_TEXT   = '#e0e0e0'

plt.rcParams.update({
    'text.color':      C_TEXT,
    'axes.labelcolor': C_TEXT,
    'xtick.color':     C_TEXT,
    'ytick.color':     C_TEXT,
    'font.family':     'DejaVu Sans',
})

fig = plt.figure(figsize=(18, 9), facecolor=BG_DARK)
fig.suptitle('Batch GD  vs  Stochastic GD — Live Convergence', fontsize=18,
             fontweight='bold', color=C_TEXT, y=0.97)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Axes ──────────────────────────────────────────────────────────────────────
def styled_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color=C_TEXT, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.grid(color='#222244', linestyle='--', linewidth=0.5)
    return ax

ax_loss = styled_ax(fig.add_subplot(gs[0, :2]), 'MSE Loss over Epochs',       'Epoch', 'MSE Loss')
ax_step = styled_ax(fig.add_subplot(gs[1, :2]), 'Step Size  |ΔLoss| / Epoch', 'Epoch', '|ΔLoss|')
ax_r2   = styled_ax(fig.add_subplot(gs[:, 2]),  'R² Score (filling up)',       'Model', 'R²')

# ── Loss plot ─────────────────────────────────────────────────────────────────
loss_max = max(bgd.loss_history[0], sgd.loss_history[0]) * 1.05
ax_loss.set_xlim(0, EPOCHS)
ax_loss.set_ylim(0, loss_max)

line_bgd_loss, = ax_loss.plot([], [], color=C_BGD, lw=2.2, label='Batch GD')
line_sgd_loss, = ax_loss.plot([], [], color=C_SGD, lw=2.2, label='SGD',
                               alpha=0.85, linestyle='--')
epoch_label = ax_loss.text(0.98, 0.92, '', transform=ax_loss.transAxes,
                            ha='right', color=C_TEXT, fontsize=11, fontweight='bold')
ax_loss.legend(facecolor=BG_PANEL, edgecolor='#333355', labelcolor=C_TEXT, fontsize=10)

# ── Step-size plot ────────────────────────────────────────────────────────────
step_max = max(max(bgd_steps), max(sgd_steps)) * 1.1
ax_step.set_xlim(0, EPOCHS - 1)
ax_step.set_ylim(0, step_max)

line_bgd_step, = ax_step.plot([], [], color=C_BGD, lw=2, label='Batch GD')
line_sgd_step, = ax_step.plot([], [], color=C_SGD, lw=2, label='SGD',
                               alpha=0.85, linestyle='--')
ax_step.legend(facecolor=BG_PANEL, edgecolor='#333355', labelcolor=C_TEXT, fontsize=10)

# ── R² bar chart ──────────────────────────────────────────────────────────────
ax_r2.set_xlim(-0.5, 1.5)
ax_r2.set_ylim(0, 1.05)
ax_r2.set_xticks([0, 1])
ax_r2.set_xticklabels(['Batch GD', 'SGD'], fontsize=11)

bar_bgd = ax_r2.bar(0, 0, width=0.5, color=C_BGD, alpha=0.85, zorder=3)
bar_sgd = ax_r2.bar(1, 0, width=0.5, color=C_SGD, alpha=0.85, zorder=3)
txt_bgd = ax_r2.text(0, 0.02, '', ha='center', color='white', fontsize=12, fontweight='bold', zorder=4)
txt_sgd = ax_r2.text(1, 0.02, '', ha='center', color='white', fontsize=12, fontweight='bold', zorder=4)
ax_r2.axhline(y=max(bgd_r2, sgd_r2), color='#ffffff33', linestyle=':', linewidth=1.5)

# ── Animation ─────────────────────────────────────────────────────────────────
def animate(frame):
    f = frame + 1

    line_bgd_loss.set_data(range(f), bgd.loss_history[:f])
    line_sgd_loss.set_data(range(f), sgd.loss_history[:f])
    epoch_label.set_text(f'Epoch {f}/{EPOCHS}')

    if f > 1:
        line_bgd_step.set_data(range(f - 1), bgd_steps[:f - 1])
        line_sgd_step.set_data(range(f - 1), sgd_steps[:f - 1])

    progress = f / EPOCHS
    h_bgd = bgd_r2 * progress
    h_sgd = sgd_r2 * progress
    bar_bgd[0].set_height(h_bgd)
    bar_sgd[0].set_height(h_sgd)
    txt_bgd.set_position((0, h_bgd + 0.02))
    txt_sgd.set_position((1, h_sgd + 0.02))
    txt_bgd.set_text(f'{h_bgd:.3f}')
    txt_sgd.set_text(f'{h_sgd:.3f}')

    return (line_bgd_loss, line_sgd_loss, epoch_label,
            line_bgd_step, line_sgd_step,
            bar_bgd[0], bar_sgd[0], txt_bgd, txt_sgd)


ani = animation.FuncAnimation(fig, animate, frames=EPOCHS,
                               interval=60, blit=True, repeat=True, repeat_delay=1500)

plt.show()
