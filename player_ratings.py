"""
player_ratings.py
Empirical Bayes shrunk ratings under pressure for SupeRR Selector Season 2.
Input: scored_balls.parquet
Outputs:
  - ratings_batters_v2.csv
  - ratings_bowlers_v2.csv
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, log_loss

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
W_WICKET = 15.0       # Runs per wicket (tested with 10.0 and 20.0)
MIN_NEFF = 60.0       # Minimum effective sample size
WEIGHT_LEVEL = 0.5    # Weight for pressure_level in final rating
WEIGHT_DELTA = 0.5    # Weight for pressure_delta in final rating
N_BOOTSTRAP = 500     # Bootstrap iterations over matches
RNG_SEED = 42

def load_data():
    parquet_path = 'scored_balls.parquet'
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found.")
        sys.exit(1)
    df = pd.read_parquet(parquet_path)
    return df

def step0_transparency(df):
    print("=" * 80)
    print("STEP 0 - TRANSPARENCY & DATA AUDIT")
    print("=" * 80)

    clean = df[df['in_scope_clean']].copy()

    # (a) OOF RMSE & LogLoss
    clean_bat = clean[clean['is_legal'] | (clean['batter_runs'] > 0)]
    
    y_bat_runs = clean_bat['batter_runs'].values
    p_bat_runs = clean_bat['xBatRuns'].values
    rmse_bat_runs = root_mean_squared_error(y_bat_runs, p_bat_runs)
    rmse_base_bat_runs = root_mean_squared_error(y_bat_runs, np.full_like(y_bat_runs, y_bat_runs.mean()))

    y_bat_out = clean_bat['is_batter_out'].values
    p_bat_out = clean_bat['xBatOut'].values
    ll_bat_out = log_loss(y_bat_out, p_bat_out)
    ll_base_bat_out = log_loss(y_bat_out, np.full_like(y_bat_out, y_bat_out.mean()))

    y_bowl_runs = clean['bowler_runs_conceded'].values
    p_bowl_runs = clean['xBowlRuns'].values
    rmse_bowl_runs = root_mean_squared_error(y_bowl_runs, p_bowl_runs)
    rmse_base_bowl_runs = root_mean_squared_error(y_bowl_runs, np.full_like(y_bowl_runs, y_bowl_runs.mean()))

    y_bowl_wkt = clean['is_bowler_wkt_int'].values
    p_bowl_wkt = clean['xBowlWkt'].values
    ll_bowl_wkt = log_loss(y_bowl_wkt, p_bowl_wkt)
    ll_base_bowl_wkt = log_loss(y_bowl_wkt, np.full_like(y_bowl_wkt, y_bowl_wkt.mean()))

    print("\n(a) Model Diagnostics & Residual Means (in_scope_clean):")
    print(f"  xBatRuns:  RMSE = {rmse_bat_runs:.4f} vs Mean Baseline = {rmse_base_bat_runs:.4f} (Δ = {rmse_base_bat_runs - rmse_bat_runs:+.4f})")
    print(f"  xBatOut:   LogLoss = {ll_bat_out:.4f} vs BaseRate = {ll_base_bat_out:.4f} (Δ = {ll_base_bat_out - ll_bat_out:+.4f})")
    print(f"  xBowlRuns: RMSE = {rmse_bowl_runs:.4f} vs Mean Baseline = {rmse_base_bowl_runs:.4f} (Δ = {rmse_base_bowl_runs - rmse_bowl_runs:+.4f})")
    print(f"  xBowlWkt:  LogLoss = {ll_bowl_wkt:.4f} vs BaseRate = {ll_base_bowl_wkt:.4f} (Δ = {ll_base_bowl_wkt - ll_bowl_wkt:+.4f})")
    print(f"  Residual Means: bat_rae = {clean_bat['bat_rae'].mean():+.6f}, bat_out_ae = {clean_bat['bat_out_ae'].mean():+.6f}, bowl_rae = {clean['bowl_rae'].mean():+.6f}, bowl_wae = {clean['bowl_wae'].mean():+.6f}")

    # (b) Dropped Matches Breakdown
    # Find original 41 gap flagged matches from pressure_model logic
    in_scope_m = df[df['in_scope']].copy()
    inn2_m = in_scope_m[in_scope_m['innings'] == 2]
    inn2_last = inn2_m.sort_values(['match_id', 'innings_ball_num']).groupby('match_id').last().reset_index()
    inn2_last['final_total'] = inn2_last['runs_so_far'] + inn2_last['total_runs_this_ball']
    inn2_last['gap'] = inn2_last['final_total'] - inn2_last['target']
    
    # 41 gap flagged matches
    orig_gap_matches = set(inn2_last[
        ((inn2_last['batting_team_won'] == 1) & (inn2_last['gap'] < -3)) |
        ((inn2_last['batting_team_won'] == 0) & (inn2_last['gap'] >= 0))
    ]['match_id'])
    
    all_flagged_matches = set(df[df['data_quality_flag']]['match_id'].unique())
    extra_target_matches = all_flagged_matches - orig_gap_matches

    match_meta = df.drop_duplicates('match_id').set_index('match_id')
    print(f"\n(b) Dropped Matches Breakdown (Total Flagged = {len(all_flagged_matches)}):")
    print(f"  - Chase-Gap Anomaly Matches: {len(orig_gap_matches)}")
    print(f"  - Extra Target-Overrun Matches (runs_so_far >= target): {len(extra_target_matches)}")
    
    drop_records = []
    for mid in all_flagged_matches:
        fmt = match_meta.loc[mid, 'comp_format']
        won = match_meta.loc[mid, 'batting_team_won']
        reason = 'Chase Gap Flag' if mid in orig_gap_matches else 'Target Overrun'
        drop_records.append({'match_id': mid, 'reason': reason, 'comp_format': fmt, 'chase_result': 'Won' if won else 'Lost'})
    
    drop_df = pd.DataFrame(drop_records)
    crosstab_drop = pd.crosstab(drop_df['reason'], [drop_df['comp_format'], drop_df['chase_result']])
    print(crosstab_drop.to_string())

    # (c) Old Top-10 Reference
    print("\n(c) Previous Top-10 Rankings (pressure_player_rankings.csv):")
    if os.path.exists('pressure_player_rankings.csv'):
        old_p = pd.read_csv('pressure_player_rankings.csv')
        print(old_p[['Rank', 'Batter_Name', 'Batter_ID', 'Batter_HP_RAE', 'Bowler_Name', 'Bowler_ID', 'Bowler_HP_RAE']].to_string(index=False))
    else:
        print("  pressure_player_rankings.csv not found.")

def compute_player_metrics(df_inn2, role='batter', W=15.0):
    """
    Computes per-player net values, pressure levels, ordinary levels, n_eff, HP stats, and WPA.
    """
    if role == 'batter':
        # Exclude wides: is_legal or batter_runs > 0
        balls = df_inn2[df_inn2['is_legal'] | (df_inn2['batter_runs'] > 0)].copy()
        balls['net_val'] = balls['bat_rae'] - W * balls['bat_out_ae']
        id_col = 'striker_id'
        name_col = 'striker_name'
        wpa_col = 'delta_wp'
    else:
        # Bowler: all deliveries
        balls = df_inn2.copy()
        balls['net_val'] = balls['bowl_rae'] + W * balls['bowl_wae']
        id_col = 'bowler_id'
        name_col = 'bowler_name'
        wpa_col = 'neg_delta_wp'

    balls['w'] = balls['pressure_score_w']
    balls['w2'] = balls['w'] ** 2
    balls['wnet'] = balls['w'] * balls['net_val']
    balls['is_ord'] = (balls['pressure_pct'] < 0.5).astype(float)
    balls['ord_net'] = balls['is_ord'] * balls['net_val']
    balls['is_hp'] = balls['is_high_pressure_chase'].astype(float)
    
    # Vectorized hp_runs
    runs_col = 'batter_runs' if role == 'batter' else 'bowler_runs_conceded'
    balls['hp_runs'] = balls['is_hp'] * balls[runs_col]

    # Per-ball variance across league
    sigma2_net = balls['net_val'].var()

    # Pre-aggregate by player and match for fast bootstrap & player level aggregation
    match_grp = balls.groupby([id_col, 'match_id']).agg(
        w=('w', 'sum'),
        w2=('w2', 'sum'),
        wnet=('wnet', 'sum'),
        ord_cnt=('is_ord', 'sum'),
        ord_net=('ord_net', 'sum'),
        hp_balls=('is_hp', 'sum'),
        hp_runs=('hp_runs', 'sum'),
        wpa=(wpa_col, 'sum'),
        total_balls=('w', 'count')
    ).reset_index()

    # Player totals
    p_totals = match_grp.groupby(id_col).agg(
        tot_w=('w', 'sum'),
        tot_w2=('w2', 'sum'),
        tot_wnet=('wnet', 'sum'),
        tot_ord_cnt=('ord_cnt', 'sum'),
        tot_ord_net=('ord_net', 'sum'),
        tot_hp_balls=('hp_balls', 'sum'),
        tot_hp_runs=('hp_runs', 'sum'),
        tot_wpa=('wpa', 'sum'),
        tot_balls=('total_balls', 'sum')
    ).reset_index()

    p_totals['n_eff'] = (p_totals['tot_w'] ** 2) / p_totals['tot_w2']
    p_totals['pressure_level'] = p_totals['tot_wnet'] / p_totals['tot_w']
    p_totals['ordinary_level'] = np.where(p_totals['tot_ord_cnt'] > 0, p_totals['tot_ord_net'] / p_totals['tot_ord_cnt'], np.nan)
    p_totals['pressure_delta'] = p_totals['pressure_level'] - p_totals['ordinary_level']

    if role == 'batter':
        p_totals['hp_rate'] = np.where(p_totals['tot_hp_balls'] > 0, (p_totals['tot_hp_runs'] / p_totals['tot_hp_balls']) * 100, np.nan) # SR
    else:
        p_totals['hp_rate'] = np.where(p_totals['tot_hp_balls'] > 0, (p_totals['tot_hp_runs'] / p_totals['tot_hp_balls']) * 6, np.nan) # Econ

    # Map player names
    id_name_map = balls.drop_duplicates(id_col).set_index(id_col)[name_col].to_dict()
    p_totals['player_name'] = p_totals[id_col].map(id_name_map)

    return p_totals, match_grp, sigma2_net, balls

def run_eb_shrinkage(p_stats, sigma2_net, min_neff=60.0):
    """
    Applies Empirical Bayes shrinkage on pressure_level and pressure_delta.
    """
    qual = p_stats[(p_stats['n_eff'] >= min_neff) & (p_stats['pressure_delta'].notna())].copy()
    
    # 1. pressure_level shrinkage
    var_p_level = qual['pressure_level'].var()
    mean_samp_var_lev = (sigma2_net / qual['n_eff']).mean()
    tau2_lev = max(0.0, var_p_level - mean_samp_var_lev)
    k_lev = sigma2_net / tau2_lev if tau2_lev > 0 else 1e9
    mean_league_lev = qual['pressure_level'].mean()
    
    qual['shrunk_pressure_level'] = (
        (qual['n_eff'] / (qual['n_eff'] + k_lev)) * qual['pressure_level'] +
        (k_lev / (qual['n_eff'] + k_lev)) * mean_league_lev
    )

    # 2. pressure_delta shrinkage
    var_p_del = qual['pressure_delta'].var()
    mean_samp_var_del = (sigma2_net / qual['n_eff']).mean()
    tau2_del = max(0.0, var_p_del - mean_samp_var_del)
    k_del = sigma2_net / tau2_del if tau2_del > 0 else 1e9
    mean_league_del = qual['pressure_delta'].mean()

    qual['shrunk_pressure_delta'] = (
        (qual['n_eff'] / (qual['n_eff'] + k_del)) * qual['pressure_delta'] +
        (k_del / (qual['n_eff'] + k_del)) * mean_league_del
    )

    # Final rating: 0.5 * z(shrunk_level) + 0.5 * z(shrunk_delta)
    z_lev = (qual['shrunk_pressure_level'] - qual['shrunk_pressure_level'].mean()) / qual['shrunk_pressure_level'].std()
    z_del = (qual['shrunk_pressure_delta'] - qual['shrunk_pressure_delta'].mean()) / qual['shrunk_pressure_delta'].std()
    qual['rating'] = WEIGHT_LEVEL * z_lev + WEIGHT_DELTA * z_del

    shrink_params = {
        'sigma2_net': sigma2_net,
        'tau2_lev': tau2_lev, 'k_lev': k_lev, 'mean_league_lev': mean_league_lev,
        'tau2_del': tau2_del, 'k_del': k_del, 'mean_league_del': mean_league_del
    }
    return qual, shrink_params

def bootstrap_ratings(qual_players, match_grp, shrink_params, role='batter', n_reps=500, rng_seed=42):
    """
    Fast bootstrap resampling of matches per player to compute 90% CI and top-10 probability.
    """
    id_col = 'striker_id' if role == 'batter' else 'bowler_id'
    qual_ids = qual_players[id_col].values
    n_players = len(qual_ids)
    
    # Pre-group match data by player into numpy arrays for speed
    # columns: w, w2, wnet, ord_cnt, ord_net
    player_data = {}
    for pid in qual_ids:
        sub = match_grp[match_grp[id_col] == pid]
        player_data[pid] = sub[['w', 'w2', 'wnet', 'ord_cnt', 'ord_net']].values

    ratings_matrix = np.zeros((n_reps, n_players))
    in_top10_matrix = np.zeros((n_reps, n_players), dtype=bool)

    rng = np.random.default_rng(rng_seed)

    k_lev = shrink_params['k_lev']
    mean_lev = shrink_params['mean_league_lev']
    k_del = shrink_params['k_del']
    mean_del = shrink_params['mean_league_del']

    for b in range(n_reps):
        b_levels = np.zeros(n_players)
        b_deltas = np.zeros(n_players)
        b_neffs = np.zeros(n_players)

        for i, pid in enumerate(qual_ids):
            arr = player_data[pid]
            n_m = len(arr)
            # Resample matches with replacement
            idx = rng.choice(n_m, size=n_m, replace=True)
            samp = arr[idx].sum(axis=0)
            
            sw, sw2, swnet, socnt, sonet = samp
            neff = (sw ** 2) / sw2 if sw2 > 0 else 1e-6
            plev = swnet / sw if sw > 0 else 0
            ordlev = sonet / socnt if socnt > 0 else plev
            
            b_neffs[i] = neff
            b_levels[i] = plev
            b_deltas[i] = plev - ordlev

        # Apply shrinkage
        shrunk_l = (b_neffs / (b_neffs + k_lev)) * b_levels + (k_lev / (b_neffs + k_lev)) * mean_lev
        shrunk_d = (b_neffs / (b_neffs + k_del)) * b_deltas + (k_del / (b_neffs + k_del)) * mean_del

        # Standardize
        std_l = shrunk_l.std()
        std_d = shrunk_d.std()
        z_l = (shrunk_l - shrunk_l.mean()) / (std_l if std_l > 0 else 1.0)
        z_d = (shrunk_d - shrunk_d.mean()) / (std_d if std_d > 0 else 1.0)
        
        rep_rating = WEIGHT_LEVEL * z_l + WEIGHT_DELTA * z_d
        ratings_matrix[b] = rep_rating

        # Top 10 indices
        top10_idx = np.argsort(rep_rating)[-10:]
        in_top10_matrix[b, top10_idx] = True

    ci_low = np.percentile(ratings_matrix, 5, axis=0)
    ci_high = np.percentile(ratings_matrix, 95, axis=0)
    top10_prob = in_top10_matrix.mean(axis=0)

    res_df = pd.DataFrame({
        id_col: qual_ids,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'top10_prob': top10_prob
    })
    return res_df

def get_player_team_map():
    mapping = pd.read_csv('competition_team_player_mapping.csv')
    # Extract distinct teams per player
    team_dict = {}
    for pid, grp in mapping.groupby('player_id'):
        teams = grp['team_name'].unique().tolist()
        # Sort or prioritize IPL franchise and National team
        team_str = " / ".join(teams[:2]) if len(teams) > 1 else teams[0]
        team_dict[pid] = team_str
    return team_dict

def opposition_check(qual_df, scored_inn2, role='batter'):
    id_col = 'striker_id' if role == 'batter' else 'bowler_id'
    opp_team_col = 'bowling_team_id' if role == 'batter' else 'batting_team_id'

    teams = pd.read_csv('teams.csv').set_index('team_id')['team_name'].to_dict()

    top15 = qual_df.sort_values('rating', ascending=False).head(15)
    print(f"\n--- Opposition Analysis for Top 15 {role.capitalize()}s ---")
    
    rows = []
    for _, row in top15.iterrows():
        pid = row[id_col]
        p_name = row['player_name']
        
        p_balls = scored_inn2[scored_inn2[id_col] == pid].copy()
        tot_w = p_balls['pressure_score_w'].sum()
        
        ipl_w = p_balls.loc[p_balls['comp_format'] == 'IPL', 'pressure_score_w'].sum()
        ipl_share = (ipl_w / tot_w * 100) if tot_w > 0 else 0
        t20i_share = 100.0 - ipl_share
        
        t20i_balls = p_balls[p_balls['comp_format'] == 'T20I']
        t20i_opps = t20i_balls[opp_team_col].map(teams).unique()
        opp_str = ", ".join([str(o) for o in t20i_opps if pd.notna(o)]) if len(t20i_opps) > 0 else "None"
        
        rows.append({
            'Player': p_name,
            'IPL_Share%': round(ipl_share, 1),
            'T20I_Share%': round(t20i_share, 1),
            'T20I_Opponents': opp_str
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # Sensitivity: Restrict to IPL + T20I matches where BOTH teams are among top 12
    t20i_m = scored_inn2[scored_inn2['comp_format'] == 'T20I'].drop_duplicates('match_id')
    team_counts = pd.concat([t20i_m['batting_team_id'], t20i_m['bowling_team_id']]).value_counts()
    top12_ids = set(team_counts.head(12).index)
    
    t20i_both_top12 = t20i_m[t20i_m['batting_team_id'].isin(top12_ids) & t20i_m['bowling_team_id'].isin(top12_ids)]['match_id']
    ipl_m = scored_inn2[scored_inn2['comp_format'] == 'IPL']['match_id'].unique()
    sens_matches = set(ipl_m).union(set(t20i_both_top12))

    sens_inn2 = scored_inn2[scored_inn2['match_id'].isin(sens_matches)].copy()
    p_sens, _, s2_sens, _ = compute_player_metrics(sens_inn2, role=role, W=W_WICKET)
    qual_sens, _ = run_eb_shrinkage(p_sens, s2_sens, min_neff=MIN_NEFF)
    
    main_top10 = set(top15.head(10)[id_col])
    sens_top10 = set(qual_sens.sort_values('rating', ascending=False).head(10)[id_col])
    overlap = len(main_top10.intersection(sens_top10))
    print(f"\nSensitivity Test (IPL + Top-12 T20I Only): Top 10 Overlap = {overlap}/10 players retained.")

def run_pipeline():
    print("=" * 80)
    print("PLAYER RATINGS UNDER PRESSURE (V2 - EMPIRICAL BAYES & WPA)")
    print("=" * 80)
    
    df = load_data()
    
    # Transparency
    step0_transparency(df)

    # Filter: Innings 2 in_scope_clean balls
    clean_inn2 = df[df['in_scope_clean'] & (df['innings'] == 2)].copy()
    clean_inn2 = clean_inn2.sort_values(['match_id', 'innings_ball_num'])
    
    # Calculate delivery-level WPA
    clean_inn2['wp_next'] = clean_inn2.groupby('match_id')['wp'].shift(-1)
    clean_inn2['wp_next'] = clean_inn2['wp_next'].fillna(clean_inn2['batting_team_won'].astype(float))
    clean_inn2['delta_wp'] = clean_inn2['wp_next'] - clean_inn2['wp']
    clean_inn2['neg_delta_wp'] = -clean_inn2['delta_wp']

    team_map = get_player_team_map()
    old_rankings = pd.read_csv('pressure_player_rankings.csv') if os.path.exists('pressure_player_rankings.csv') else None

    # Test W parameter sensitivity (10, 15, 20)
    print("\n" + "=" * 80)
    print("STEP 2 - W PARAMETER SENSITIVITY (W = 10, 15, 20 runs per wicket)")
    print("=" * 80)
    for test_w in [10.0, 15.0, 20.0]:
        p_b, _, s2_b, _ = compute_player_metrics(clean_inn2, role='batter', W=test_w)
        q_b, _ = run_eb_shrinkage(p_b, s2_b, min_neff=60.0)
        top5_b = q_b.sort_values('rating', ascending=False).head(5)['player_name'].tolist()
        
        p_bw, _, s2_bw, _ = compute_player_metrics(clean_inn2, role='bowler', W=test_w)
        q_bw, _ = run_eb_shrinkage(p_bw, s2_bw, min_neff=60.0)
        top5_bw = q_bw.sort_values('rating', ascending=False).head(5)['player_name'].tolist()
        print(f"  W = {test_w:2.0f} | Top 5 Batters: {', '.join(top5_b)}")
        print(f"         | Top 5 Bowlers: {', '.join(top5_bw)}")

    # ==========================================================================
    # 1. BATTERS V2 PIPELINE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 3 to 6 - BATTERS RATINGS V2 (W = 15.0)")
    print("=" * 80)
    p_bat, match_grp_bat, s2_bat, bat_balls = compute_player_metrics(clean_inn2, role='batter', W=W_WICKET)
    
    print("\nBatters Eligibility by n_eff:")
    for cut in [40, 60, 100]:
        cnt = (p_bat['n_eff'] >= cut).sum()
        print(f"  Batters with n_eff >= {cut}: {cnt} players")

    qual_bat, shrink_bat = run_eb_shrinkage(p_bat, s2_bat, min_neff=MIN_NEFF)
    print("\nBatters Empirical Bayes Shrinkage Parameters:")
    print(f"  Per-ball net variance (sigma2): {shrink_bat['sigma2_net']:.4f}")
    print(f"  pressure_level: tau2 = {shrink_bat['tau2_lev']:.4f}, k = {shrink_bat['k_lev']:.2f}, league_mean = {shrink_bat['mean_league_lev']:.4f}")
    print(f"  pressure_delta: tau2 = {shrink_bat['tau2_del']:.4f}, k = {shrink_bat['k_del']:.2f}, league_mean = {shrink_bat['mean_league_del']:.4f}")

    print(f"\nRunning Match-Cluster Bootstrap (500 reps) for Batters...")
    boot_bat = bootstrap_ratings(qual_bat, match_grp_bat, shrink_bat, role='batter', n_reps=N_BOOTSTRAP, rng_seed=RNG_SEED)
    qual_bat = pd.merge(qual_bat, boot_bat, on='striker_id')

    qual_bat['team'] = qual_bat['striker_id'].map(team_map).fillna('Unknown')
    qual_bat = qual_bat.sort_values('rating', ascending=False).reset_index(drop=True)
    qual_bat['rank_v2'] = qual_bat.index + 1

    # Map old rank
    if old_rankings is not None:
        old_bat_ranks = old_rankings.set_index('Batter_ID')['Rank'].to_dict()
        qual_bat['old_rank'] = qual_bat['striker_id'].map(old_bat_ranks).fillna('-')
    else:
        qual_bat['old_rank'] = '-'

    # Save ratings_batters_v2.csv
    bat_out_cols = [
        'rank_v2', 'old_rank', 'striker_id', 'player_name', 'team',
        'n_eff', 'tot_hp_balls', 'hp_rate',
        'pressure_level', 'ordinary_level', 'pressure_delta',
        'shrunk_pressure_level', 'shrunk_pressure_delta',
        'rating', 'ci_low', 'ci_high', 'top10_prob', 'tot_wpa'
    ]
    qual_bat_export = qual_bat[bat_out_cols].copy()
    qual_bat_export.columns = [
        'rank_v2', 'old_rank', 'player_id', 'name', 'team',
        'n_eff', 'hp_balls', 'hp_strike_rate',
        'pressure_level', 'ordinary_level', 'delta',
        'shrunk_pressure_level', 'shrunk_pressure_delta',
        'rating', 'ci_low', 'ci_high', 'top10_prob', 'WPA'
    ]
    # Round floats
    for c in ['n_eff', 'hp_strike_rate', 'pressure_level', 'ordinary_level', 'delta', 'shrunk_pressure_level', 'shrunk_pressure_delta', 'rating', 'ci_low', 'ci_high', 'top10_prob', 'WPA']:
        qual_bat_export[c] = np.round(qual_bat_export[c], 3)
    
    qual_bat_export.to_csv('ratings_batters_v2.csv', index=False)
    print("Saved ratings_batters_v2.csv.")

    # Opposition check
    opposition_check(qual_bat, clean_inn2, role='batter')

    # ==========================================================================
    # 2. BOWLERS V2 PIPELINE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 3 to 6 - BOWLERS RATINGS V2 (W = 15.0)")
    print("=" * 80)
    p_bowl, match_grp_bowl, s2_bowl, bowl_balls = compute_player_metrics(clean_inn2, role='bowler', W=W_WICKET)
    
    print("\nBowlers Eligibility by n_eff:")
    for cut in [40, 60, 100]:
        cnt = (p_bowl['n_eff'] >= cut).sum()
        print(f"  Bowlers with n_eff >= {cut}: {cnt} players")

    qual_bowl, shrink_bowl = run_eb_shrinkage(p_bowl, s2_bowl, min_neff=MIN_NEFF)
    print("\nBowlers Empirical Bayes Shrinkage Parameters:")
    print(f"  Per-ball net variance (sigma2): {shrink_bowl['sigma2_net']:.4f}")
    print(f"  pressure_level: tau2 = {shrink_bowl['tau2_lev']:.4f}, k = {shrink_bowl['k_lev']:.2f}, league_mean = {shrink_bowl['mean_league_lev']:.4f}")
    print(f"  pressure_delta: tau2 = {shrink_bowl['tau2_del']:.4f}, k = {shrink_bowl['k_del']:.2f}, league_mean = {shrink_bowl['mean_league_del']:.4f}")

    print(f"\nRunning Match-Cluster Bootstrap (500 reps) for Bowlers...")
    boot_bowl = bootstrap_ratings(qual_bowl, match_grp_bowl, shrink_bowl, role='bowler', n_reps=N_BOOTSTRAP, rng_seed=RNG_SEED)
    qual_bowl = pd.merge(qual_bowl, boot_bowl, on='bowler_id')

    qual_bowl['team'] = qual_bowl['bowler_id'].map(team_map).fillna('Unknown')
    qual_bowl = qual_bowl.sort_values('rating', ascending=False).reset_index(drop=True)
    qual_bowl['rank_v2'] = qual_bowl.index + 1

    # Map old rank
    if old_rankings is not None:
        old_bowl_ranks = old_rankings.set_index('Bowler_ID')['Rank'].to_dict()
        qual_bowl['old_rank'] = qual_bowl['bowler_id'].map(old_bowl_ranks).fillna('-')
    else:
        qual_bowl['old_rank'] = '-'

    # Save ratings_bowlers_v2.csv
    bowl_out_cols = [
        'rank_v2', 'old_rank', 'bowler_id', 'player_name', 'team',
        'n_eff', 'tot_hp_balls', 'hp_rate',
        'pressure_level', 'ordinary_level', 'pressure_delta',
        'shrunk_pressure_level', 'shrunk_pressure_delta',
        'rating', 'ci_low', 'ci_high', 'top10_prob', 'tot_wpa'
    ]
    qual_bowl_export = qual_bowl[bowl_out_cols].copy()
    qual_bowl_export.columns = [
        'rank_v2', 'old_rank', 'player_id', 'name', 'team',
        'n_eff', 'hp_balls', 'hp_economy',
        'pressure_level', 'ordinary_level', 'delta',
        'shrunk_pressure_level', 'shrunk_pressure_delta',
        'rating', 'ci_low', 'ci_high', 'top10_prob', 'WPA'
    ]
    for c in ['n_eff', 'hp_economy', 'pressure_level', 'ordinary_level', 'delta', 'shrunk_pressure_level', 'shrunk_pressure_delta', 'rating', 'ci_low', 'ci_high', 'top10_prob', 'WPA']:
        qual_bowl_export[c] = np.round(qual_bowl_export[c], 3)
    
    qual_bowl_export.to_csv('ratings_bowlers_v2.csv', index=False)
    print("Saved ratings_bowlers_v2.csv.")

    # Opposition check
    opposition_check(qual_bowl, clean_inn2, role='bowler')

    # ==========================================================================
    # FINAL DISPLAY: TOP 15 COMPARISON TABLE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TOP 15 BATTERS (NEW V2 vs OLD RANK):")
    print("=" * 80)
    display_bat = qual_bat_export.head(15)[[
        'rank_v2', 'old_rank', 'player_id', 'name', 'n_eff', 'hp_balls', 'hp_strike_rate',
        'delta', 'rating', 'ci_low', 'ci_high', 'top10_prob', 'WPA'
    ]]
    print(display_bat.to_string(index=False))

    print("\n" + "=" * 80)
    print("TOP 15 BOWLERS (NEW V2 vs OLD RANK):")
    print("=" * 80)
    display_bowl = qual_bowl_export.head(15)[[
        'rank_v2', 'old_rank', 'player_id', 'name', 'n_eff', 'hp_balls', 'hp_economy',
        'delta', 'rating', 'ci_low', 'ci_high', 'top10_prob', 'WPA'
    ]]
    print(display_bowl.to_string(index=False))
    print("=" * 80)

if __name__ == '__main__':
    run_pipeline()
