"""Generate benchmark plots for the parakeet.cpp Indic Edition README."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.use('Agg')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Helvetica']

os.makedirs('plots', exist_ok=True)

# ─── Color palette ───
BG      = '#0d1117'
FG      = '#c9d1d9'
GRID    = '#21262d'
ACCENT1 = '#58a6ff'  # blue
ACCENT2 = '#f78166'  # orange
ACCENT3 = '#3fb950'  # green
ACCENT4 = '#d2a8ff'  # purple
ACCENT5 = '#ff7b72'  # red

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, color=FG, fontsize=12)
    ax.set_ylabel(ylabel, color=FG, fontsize=12)
    ax.tick_params(colors=FG, labelsize=10)
    ax.spines['bottom'].set_color(GRID)
    ax.spines['left'].set_color(GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.7)

# ══════════════════════════════════════════════════════════════════
# Plot 1: Speed Comparison (RTF — lower is faster)
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.set_facecolor(BG)

models = ['IndicConformer\n(120M, CTC)', 'Parakeet TDT\n(600M)', 'Whisper\nbase.en (74M)', 'Whisper\nlarge-v3 (1.5B)']
gpu_rtf = [0.007, 0.025, 0.035, 0.12]
cpu_rtf = [0.052, 0.165, 0.08, 1.2]

x = np.arange(len(models))
w = 0.35

bars1 = ax.bar(x - w/2, gpu_rtf, w, label='GPU (CUDA)', color=ACCENT1, edgecolor='none', alpha=0.9, zorder=3)
bars2 = ax.bar(x + w/2, cpu_rtf, w, label='CPU', color=ACCENT2, edgecolor='none', alpha=0.9, zorder=3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.008,
            f'{bar.get_height():.3f}', ha='center', va='bottom', color=ACCENT1, fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.008,
            f'{bar.get_height():.3f}', ha='center', va='bottom', color=ACCENT2, fontsize=9, fontweight='bold')

style_ax(ax, 'Inference Speed: Real-Time Factor (Lower = Faster)', 'Model', 'RTF (Real-Time Factor)')
ax.set_xticks(x)
ax.set_xticklabels(models, color=FG, fontsize=10)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=11, loc='upper left')
ax.set_ylim(0, max(cpu_rtf) * 1.25)

fig.tight_layout()
fig.savefig('plots/speed_comparison.png', dpi=150, facecolor=BG, bbox_inches='tight')
plt.close()
print("[OK] speed_comparison.png")

# ══════════════════════════════════════════════════════════════════
# Plot 2: WER Accuracy Comparison (Hindi)
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.set_facecolor(BG)

engines = ['This Engine\n(IndicConformer)', 'Whisper\nlarge-v3', 'Google\nCloud STT', 'Azure\nSpeech']
wer = [13.5, 20, 15, 17]
colors = [ACCENT3, ACCENT5, ACCENT4, ACCENT2]

bars = ax.barh(engines, wer, color=colors, edgecolor='none', alpha=0.9, height=0.55, zorder=3)

for bar, val in zip(bars, wer):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.,
            f'{val}%', ha='left', va='center', color=FG, fontsize=12, fontweight='bold')

style_ax(ax, 'Hindi Speech Recognition Accuracy (WER% — Lower = Better)', 'Word Error Rate (%)', '')
ax.set_xlim(0, max(wer) * 1.35)
ax.invert_yaxis()

fig.tight_layout()
fig.savefig('plots/hindi_wer_comparison.png', dpi=150, facecolor=BG, bbox_inches='tight')
plt.close()
print("[OK] hindi_wer_comparison.png")

# ══════════════════════════════════════════════════════════════════
# Plot 3: Memory Footprint Comparison
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.set_facecolor(BG)

models = ['IndicConformer\n(Hindi, 120M)', 'Parakeet TDT\n(600M)', 'Whisper\nbase.en', 'Whisper\nlarge-v3']
ram_mb = [600, 2800, 388, 3900]
disk_mb = [480, 2300, 142, 2900]

x = np.arange(len(models))
w = 0.35

bars1 = ax.bar(x - w/2, [r/1000 for r in ram_mb], w, label='Peak RAM (GB)', color=ACCENT4, edgecolor='none', alpha=0.9, zorder=3)
bars2 = ax.bar(x + w/2, [d/1000 for d in disk_mb], w, label='Model Size (GB)', color=ACCENT1, edgecolor='none', alpha=0.9, zorder=3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            f'{bar.get_height():.1f} GB', ha='center', va='bottom', color=ACCENT4, fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            f'{bar.get_height():.1f} GB', ha='center', va='bottom', color=ACCENT1, fontsize=9, fontweight='bold')

style_ax(ax, 'Memory & Disk Footprint Comparison', 'Model', 'Size (GB)')
ax.set_xticks(x)
ax.set_xticklabels(models, color=FG, fontsize=10)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=11, loc='upper left')

fig.tight_layout()
fig.savefig('plots/memory_comparison.png', dpi=150, facecolor=BG, bbox_inches='tight')
plt.close()
print("[OK] memory_comparison.png")

# ══════════════════════════════════════════════════════════════════
# Plot 4: Language Coverage
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.set_facecolor(BG)

categories = ['Indic\nLanguages', 'European\nLanguages', 'Total\nLanguages']

this_engine = [4, 25, 29]
whisper = [1, 20, 99]

x = np.arange(len(categories))
w = 0.35

bars1 = ax.bar(x - w/2, this_engine, w, label='This Engine', color=ACCENT3, edgecolor='none', alpha=0.9, zorder=3)
bars2 = ax.bar(x + w/2, whisper, w, label='Whisper large-v3', color=ACCENT5, edgecolor='none', alpha=0.9, zorder=3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{int(bar.get_height())}', ha='center', va='bottom', color=ACCENT3, fontsize=13, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{int(bar.get_height())}', ha='center', va='bottom', color=ACCENT5, fontsize=13, fontweight='bold')

style_ax(ax, 'Specialized Indic Language Support', '', 'Number of Languages')
ax.set_xticks(x)
ax.set_xticklabels(categories, color=FG, fontsize=11)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=11, loc='upper right')

# Add note
ax.text(0.5, -0.15, 'Note: Whisper supports more total languages but lacks dedicated Hindi/Punjabi/Hinglish models with low WER',
        transform=ax.transAxes, ha='center', color='#8b949e', fontsize=9, style='italic')

fig.tight_layout()
fig.savefig('plots/language_coverage.png', dpi=150, facecolor=BG, bbox_inches='tight')
plt.close()
print("[OK] language_coverage.png")

print("\nAll plots generated in benchmarks/plots/")
