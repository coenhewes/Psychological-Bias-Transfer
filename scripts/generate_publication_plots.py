import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import os
import seaborn as sns

# Set publication-ready style (white background, high contrast for arXiv/PDF)
plt.style.use('default')
sns.set_theme(style="whitegrid", font="serif")
sns.set_context("paper", font_scale=1.5)

RESULTS_DIR = "/home/forge/Psychological-Bias-Transfer/results"
PLOTS_DIR = "/home/forge/Psychological-Bias-Transfer/results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load data
marker_freq = pd.read_csv(os.path.join(RESULTS_DIR, "marker_frequencies.csv"))
paired_ttests = pd.read_csv(os.path.join(RESULTS_DIR, "paired_ttests.csv"))
seed_variance = pd.read_csv(os.path.join(RESULTS_DIR, "seed_variance_check.csv"))

# --- 1. Main Results Figure (Marker Frequencies) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
markers = ['rumination', 'catastrophizing', 'doom_framing', 'certainty_collapse']
titles = ['Rumination', 'Catastrophizing', 'Doom Framing', 'Certainty Collapse']

palette = {'control': '#4C72B0', 'treatment': '#C44E52'}

for i, marker in enumerate(markers):
    ax = axes[i//2, i%2]
    data = marker_freq[marker_freq['marker'] == marker]
    
    sns.boxplot(x='model', y='freq_per_1k', hue='corpus', data=data, ax=ax, palette=palette, showfliers=False)
    sns.stripplot(x='model', y='freq_per_1k', hue='corpus', data=data, ax=ax, dodge=True, palette='dark:black', alpha=0.4, jitter=0.2)
    
    ax.set_title(titles[i], fontweight='bold')
    ax.set_ylabel("Frequency (per 1k tokens)")
    ax.set_xlabel("")
    
    # Only keep legend on first plot to avoid clutter
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[0:2], labels[0:2], title="Corpus", loc='upper left')
    else:
        ax.legend_.remove()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig1_marker_frequencies.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(PLOTS_DIR, "fig1_marker_frequencies.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- 2. Effect Size Plot (Cohen's d) ---
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x='marker', y='cohens_d', hue='model', data=paired_ttests, palette='viridis')
ax.axhline(0.8, ls='--', color='red', alpha=0.5, label='Large Effect (>0.8)')
ax.axhline(0, color='black', lw=1)

ax.set_title("Treatment Effect Size (Cohen's d) by Marker and Architecture", fontweight='bold')
ax.set_ylabel("Cohen's d")
ax.set_xlabel("")
ax.set_xticklabels(titles)
ax.legend(title='Model')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig2_effect_sizes.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(PLOTS_DIR, "fig2_effect_sizes.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- 3. Seed Variance Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x='marker', y='within_seed_var', hue='model', data=seed_variance, palette='mako')

ax.set_title("Initialization Instability (Within-Seed Variance)", fontweight='bold')
ax.set_ylabel("Variance")
ax.set_xlabel("")
ax.set_xticklabels(titles)
ax.legend(title='Model')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig3_seed_variance.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(PLOTS_DIR, "fig3_seed_variance.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"Generated publication-ready visuals in {PLOTS_DIR}")
