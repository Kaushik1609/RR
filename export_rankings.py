"""
export_rankings.py
Builds official rankings for SupeRR Selector Season 2:
- top_batters_under_pressure.csv (Top 30 Batters)
- top_bowlers_under_pressure.csv (Top 30 Bowlers)
- pressure_player_rankings.csv (Top 10 Batters & Bowlers formatted for Unstop dropdowns)
- top_batters_pressure_matrix.png
- top_bowlers_pressure_matrix.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print("Loading scored_balls.parquet...")
df = pd.read_parquet('scored_balls.parquet')
clean = df[df['in_scope_clean']].copy()

# Load official player mapping for canonical names
mapping = pd.read_csv('competition_team_player_mapping.csv')
name_map = mapping.drop_duplicates('player_id').set_index('player_id')['player_name'].to_dict()

# ==============================================================================
# 1. BATTING RANKINGS UNDER PRESSURE
# ==============================================================================
print("Aggregating batter metrics...")

# HP Chases: 2nd innings, pressure_pct >= 0.80, clean matches
bat_hp_data = clean[clean['is_high_pressure_chase'] & (clean['is_legal'] | (clean['batter_runs'] > 0))].copy()
bat_hp = bat_hp_data.groupby('striker_id').agg(
    hp_balls=('batter_runs', 'count'),
    hp_runs=('batter_runs', 'sum'),
    hp_outs=('is_batter_out', 'sum'),
    hp_bat_rae=('bat_rae', 'sum'),
    hp_bat_out_ae=('bat_out_ae', 'sum')
).reset_index()

# Overall Clean Deliveries (Innings 1 & 2)
bat_all_data = clean[clean['is_legal'] | (clean['batter_runs'] > 0)].copy()
bat_all = bat_all_data.groupby('striker_id').agg(
    ov_balls=('batter_runs', 'count'),
    ov_runs=('batter_runs', 'sum'),
    ov_outs=('is_batter_out', 'sum'),
    ov_bat_rae=('bat_rae', 'sum'),
    wt_bat_rae=('bat_rae', lambda x: (x * bat_all_data.loc[x.index, 'pressure_score_w']).sum())
).reset_index()

bat_comb = pd.merge(bat_hp, bat_all, on='striker_id', how='inner')
bat_comb['player_name'] = bat_comb['striker_id'].map(name_map).fillna(
    df.drop_duplicates('striker_id').set_index('striker_id')['striker_name']
)

# Derived rates
bat_comb['hp_strike_rate'] = np.round((bat_comb['hp_runs'] / bat_comb['hp_balls']) * 100, 2)
bat_comb['hp_average'] = np.round(bat_comb['hp_runs'] / bat_comb['hp_outs'].replace(0, 1), 2)
bat_comb['hp_rae_per_ball'] = np.round(bat_comb['hp_bat_rae'] / bat_comb['hp_balls'], 3)

bat_comb['ov_strike_rate'] = np.round((bat_comb['ov_runs'] / bat_comb['ov_balls']) * 100, 2)
bat_comb['ov_average'] = np.round(bat_comb['ov_runs'] / bat_comb['ov_outs'].replace(0, 1), 2)
bat_comb['ov_rae_per_ball'] = np.round(bat_comb['ov_bat_rae'] / bat_comb['ov_balls'], 3)

# Clutch differential (Performance lift in high pressure vs overall baseline)
bat_comb['rae_delta'] = np.round(bat_comb['hp_rae_per_ball'] - bat_comb['ov_rae_per_ball'], 3)
bat_comb['sr_delta'] = np.round(bat_comb['hp_strike_rate'] - bat_comb['ov_strike_rate'], 2)

# Filter minimum sample size: >= 60 HP balls faced
top_batters = bat_comb[bat_comb['hp_balls'] >= 60].sort_values('hp_bat_rae', ascending=False).reset_index(drop=True)
top_batters['pressure_rank'] = top_batters.index + 1

# Round float columns
for col in ['hp_bat_rae', 'hp_bat_out_ae', 'ov_bat_rae', 'wt_bat_rae']:
    top_batters[col] = np.round(top_batters[col], 2)

top_batters_out = top_batters.head(30)[[
    'pressure_rank', 'striker_id', 'player_name',
    'hp_balls', 'hp_runs', 'hp_outs', 'hp_average', 'hp_strike_rate',
    'hp_bat_rae', 'hp_rae_per_ball', 'rae_delta', 'wt_bat_rae',
    'ov_balls', 'ov_runs', 'ov_strike_rate'
]]
top_batters_out.to_csv('top_batters_under_pressure.csv', index=False)
print("Saved top_batters_under_pressure.csv (30 players).")

# ==============================================================================
# 2. BOWLING RANKINGS UNDER PRESSURE
# ==============================================================================
print("Aggregating bowler metrics...")

bowl_hp_data = clean[clean['is_high_pressure_chase']].copy()
bowl_hp = bowl_hp_data.groupby('bowler_id').agg(
    hp_balls=('bowler_runs_conceded', 'count'),
    hp_runs_conceded=('bowler_runs_conceded', 'sum'),
    hp_wkts=('is_bowler_wicket', 'sum'),
    hp_bowl_rae=('bowl_rae', 'sum'),
    hp_bowl_wae=('bowl_wae', 'sum')
).reset_index()

bowl_all_data = clean.copy()
bowl_all = bowl_all_data.groupby('bowler_id').agg(
    ov_balls=('bowler_runs_conceded', 'count'),
    ov_runs_conceded=('bowler_runs_conceded', 'sum'),
    ov_wkts=('is_bowler_wicket', 'sum'),
    ov_bowl_rae=('bowl_rae', 'sum'),
    ov_bowl_wae=('bowl_wae', 'sum'),
    wt_bowl_rae=('bowl_rae', lambda x: (x * bowl_all_data.loc[x.index, 'pressure_score_w']).sum()),
    wt_bowl_wae=('bowl_wae', lambda x: (x * bowl_all_data.loc[x.index, 'pressure_score_w']).sum())
).reset_index()

bowl_comb = pd.merge(bowl_hp, bowl_all, on='bowler_id', how='inner')
bowl_comb['player_name'] = bowl_comb['bowler_id'].map(name_map).fillna(
    df.drop_duplicates('bowler_id').set_index('bowler_id')['bowler_name']
)

# Derived rates
bowl_comb['hp_economy'] = np.round((bowl_comb['hp_runs_conceded'] / bowl_comb['hp_balls']) * 6, 2)
bowl_comb['hp_rae_per_ball'] = np.round(bowl_comb['hp_bowl_rae'] / bowl_comb['hp_balls'], 3)
bowl_comb['ov_economy'] = np.round((bowl_comb['ov_runs_conceded'] / bowl_comb['ov_balls']) * 6, 2)
bowl_comb['ov_rae_per_ball'] = np.round(bowl_comb['ov_bowl_rae'] / bowl_comb['ov_balls'], 3)

# Clutch differential (positive means more runs saved per ball under pressure than usual)
bowl_comb['rae_delta'] = np.round(bowl_comb['hp_rae_per_ball'] - bowl_comb['ov_rae_per_ball'], 3)
bowl_comb['econ_delta'] = np.round(bowl_comb['hp_economy'] - bowl_comb['ov_economy'], 2)

# Filter minimum sample size: >= 60 HP balls bowled
top_bowlers = bowl_comb[bowl_comb['hp_balls'] >= 60].sort_values('hp_bowl_rae', ascending=False).reset_index(drop=True)
top_bowlers['pressure_rank'] = top_bowlers.index + 1

# Round float columns
for col in ['hp_bowl_rae', 'hp_bowl_wae', 'ov_bowl_rae', 'ov_bowl_wae', 'wt_bowl_rae', 'wt_bowl_wae']:
    top_bowlers[col] = np.round(top_bowlers[col], 2)

top_bowlers_out = top_bowlers.head(30)[[
    'pressure_rank', 'bowler_id', 'player_name',
    'hp_balls', 'hp_runs_conceded', 'hp_wkts', 'hp_economy',
    'hp_bowl_rae', 'hp_bowl_wae', 'hp_rae_per_ball', 'rae_delta', 'wt_bowl_rae',
    'ov_balls', 'ov_runs_conceded', 'ov_wkts', 'ov_economy'
]]
top_bowlers_out.to_csv('top_bowlers_under_pressure.csv', index=False)
print("Saved top_bowlers_under_pressure.csv (30 players).")

# ==============================================================================
# 3. UNSTOP TOP 10 DROPDOWN SELECTION TABLE
# ==============================================================================
top10_bat = top_batters_out.head(10).copy()
top10_bowl = top_bowlers_out.head(10).copy()

unstop_table = pd.DataFrame({
    'Rank': range(1, 11),
    'Batter_Name': top10_bat['player_name'].values,
    'Batter_ID': top10_bat['striker_id'].values,
    'Batter_HP_RAE': top10_bat['hp_bat_rae'].values,
    'Batter_HP_SR': top10_bat['hp_strike_rate'].values,
    'Bowler_Name': top10_bowl['player_name'].values,
    'Bowler_ID': top10_bowl['bowler_id'].values,
    'Bowler_HP_RAE': top10_bowl['hp_bowl_rae'].values,
    'Bowler_HP_Econ': top10_bowl['hp_economy'].values
})
unstop_table.to_csv('pressure_player_rankings.csv', index=False)
print("Saved pressure_player_rankings.csv (Top 10 Unstop Selection).")

# ==============================================================================
# 4. PUBLICATION-QUALITY VISUALIZATIONS
# ==============================================================================
print("Generating publication visualizations...")

# --- Batter Matrix Plot ---
fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
b15 = top_batters_out.head(15).copy().sort_values('hp_bat_rae', ascending=True)

y_pos = np.arange(len(b15))
bars = ax.barh(y_pos, b15['hp_bat_rae'], color='#1E88E5', alpha=0.88, edgecolor='#0D47A1', height=0.65)

ax.set_yticks(y_pos)
labels = [f"{name} ({id_})" for name, id_ in zip(b15['player_name'], b15['striker_id'])]
ax.set_yticklabels(labels, fontsize=10, fontweight='medium')

for bar, rae, sr, balls in zip(bars, b15['hp_bat_rae'], b15['hp_strike_rate'], b15['hp_balls']):
    ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height()/2,
            f"+{rae:.1f} RAE | SR {sr:.1f} ({balls}b)",
            va='center', ha='left', fontsize=9, color='#1A237E', fontweight='bold')

ax.set_xlim(0, max(b15['hp_bat_rae']) + 22)
ax.set_xlabel('High-Pressure Runs Added Above Expected (hp_bat_rae)', fontsize=11, fontweight='bold', labelpad=8)
ax.set_title('Top 15 Batters Under High Pressure (2nd Innings Chases, Top 20% Leverage)\nEvaluated via Out-of-Fold Context Baseline (LightGBM)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('top_batters_pressure_matrix.png')
plt.close()

# --- Bowler Matrix Plot ---
fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
bw15 = top_bowlers_out.head(15).copy().sort_values('hp_bowl_rae', ascending=True)

y_pos = np.arange(len(bw15))
bars = ax.barh(y_pos, bw15['hp_bowl_rae'], color='#43A047', alpha=0.88, edgecolor='#1B5E20', height=0.65)

ax.set_yticks(y_pos)
labels = [f"{name} ({id_})" for name, id_ in zip(bw15['player_name'], bw15['bowler_id'])]
ax.set_yticklabels(labels, fontsize=10, fontweight='medium')

for bar, rae, econ, balls in zip(bars, bw15['hp_bowl_rae'], bw15['hp_economy'], bw15['hp_balls']):
    ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height()/2,
            f"+{rae:.1f} Saved | Econ {econ:.2f} ({balls}b)",
            va='center', ha='left', fontsize=9, color='#1B5E20', fontweight='bold')

ax.set_xlim(0, max(bw15['hp_bowl_rae']) + 18)
ax.set_xlabel('High-Pressure Runs Saved Below Expected (hp_bowl_rae)', fontsize=11, fontweight='bold', labelpad=8)
ax.set_title('Top 15 Bowlers Under High Pressure (2nd Innings Chases, Top 20% Leverage)\nEvaluated via Out-of-Fold Context Baseline (LightGBM)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('top_bowlers_pressure_matrix.png')
plt.close()

print("\n================================================================================")
print("OFFICIAL TOP 10 UNSTOP DROPDOWN RANKINGS:")
print("================================================================================")
print(unstop_table.to_string(index=False))
print("================================================================================")

