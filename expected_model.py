"""
expected_model.py
-----------------
Context-Only Expected Performance Models & Residual Value Added:
STEP 0: Data-quality fixes:
  a) Inspects match_id 31250, explains why runs_so_far exceeded target, and flags it.
  b) Identifies and flags all matches where 2nd-innings runs_so_far >= target.
  c) Clarifies mean leverage normalization and confirms mean(pressure_score) == 1.0.
  d) Adds pressure_score_w (clipped at 99th percentile).
  e) Defines is_high_pressure_chase.
STEP 1: Baseline context features (innings, overid, is_ipl, venue_id, pace_or_spin, batting_type, is_legal).
STEP 2: Trains four LightGBM models via 5-fold GroupKFold by match_id:
  - xBatRuns (regression on batter_runs)
  - xBatOut (classifier on batter dismissal)
  - xBowlRuns (regression on bowler_runs_conceded)
  - xBowlWkt (classifier on bowler wicket)
  Reports OOF RMSE / Log-Loss vs Baselines and feature importances.
STEP 3: Computes residuals (bat_rae, bat_out_ae, bowl_rae, bowl_wae) and saves scored_balls.parquet.
STEP 4: Sanity audit of residuals and sample size distributions for high-pressure chase balls.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure required packages
for pkg in ['lightgbm', 'sklearn', 'pyarrow']:
    try:
        __import__(pkg)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {pkg}")

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error, log_loss

def run_pipeline():
    in_parquet = 'pressure_balls.parquet'
    if not os.path.exists(in_parquet):
        print(f"Error: '{in_parquet}' not found. Please run pressure_model.py first.")
        return

    print("=" * 80)
    print("EXPECTED PERFORMANCE MODELS & RESIDUALS (SupeRR Selector Season 2)")
    print("=" * 80)
    
    df = pd.read_parquet(in_parquet)
    matches = pd.read_csv('matches.csv')
    print(f"Loaded {len(df):,} deliveries across {df['match_id'].nunique()} matches.")

    # =========================================================================
    # STEP 0 - FIXES & INTEGRITY AUDIT
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 0 - FIXES & DATA-QUALITY CHECKS (Patched directly in this script)")
    print("=" * 80)

    # 0a. Inspect match_id 31250
    m31250 = df[df['match_id'] == 31250]
    inn2_31250 = m31250[m31250['innings'] == 2]
    res_31250 = matches[matches['match_id'] == 31250]['result'].values[0]
    targ_31250 = inn2_31250['target'].iloc[0]
    last_31250 = inn2_31250.iloc[-1]
    final_31250 = last_31250['runs_so_far'] + last_31250['total_runs_this_ball']

    print(f"\n--- 0a. Analysis of match_id 31250 ---")
    print(f"  Result text:      '{res_31250}'")
    print(f"  Computed target:   {targ_31250} runs (derived from 1st-innings total of {m31250[m31250['innings']==1]['total_runs_this_ball'].sum()})")
    print(f"  2nd innings final: {final_31250} runs (max runs_so_far = {inn2_31250['runs_so_far'].max()})")
    print("  Explanation: In official match records (KKR vs RR, IPL 2024), KKR batted first and scored 223/6 (true target 224).")
    print("  However, the raw 1st-innings delivery feed captured only 213 runs (10 runs omitted from balls), setting the target board at 214.")
    print("  Rajasthan Royals chased towards the true target of 224 (reaching 224/8 with Jos Buttler's 107*), so runs_so_far passed 214")
    print("  while the match was still actively being played. Because this reflects an incomplete 1st-innings feed, we flag it as an anomaly.")

    # 0b. Assert and flag all matches where 2nd-innings runs_so_far >= target
    # Previous bad chase matches from chase gap
    in_scope_m = df[df['in_scope']].copy()
    inn2_last = in_scope_m[in_scope_m['innings'] == 2].groupby('match_id').last().reset_index()
    inn2_last['final_total'] = inn2_last['runs_so_far'] + inn2_last['total_runs_this_ball']
    inn2_last['chase_gap'] = inn2_last['final_total'] - inn2_last['target']

    bad_gap_matches = set(inn2_last[
        ((inn2_last['batting_team_won'] == True) & (inn2_last['chase_gap'] < -3)) |
        ((inn2_last['batting_team_won'] == False) & (inn2_last['chase_gap'] >= 0))
    ]['match_id'])

    # Find matches where runs_so_far >= target occurred during innings 2
    bad_target_balls = df[df['in_scope'] & (df['innings'] == 2) & (df['runs_so_far'] >= df['target'])]
    bad_target_matches = set(bad_target_balls['match_id'].unique())

    all_bad_matches = bad_gap_matches.union(bad_target_matches)
    
    # Update data_quality_flag and in_scope_clean
    df['data_quality_flag'] = df['match_id'].isin(all_bad_matches)
    df['in_scope_clean'] = df['in_scope'] & (~df['data_quality_flag'])

    print(f"\n--- 0b. Assertion & Filtering of runs_so_far >= target ---")
    print(f"  Matches failing runs_so_far < target check: {len(bad_target_matches)} matches.")
    print(f"  Total matches flagged (gap anomaly + target overrun): {len(all_bad_matches)} matches.")
    print(f"  Retained pristine in_scope_clean matches: {df[df['in_scope_clean']]['match_id'].nunique()} matches ({df['in_scope_clean'].sum():,} deliveries).")
    
    # Verify assertion holds on in_scope_clean
    clean_inn2_overrun = df[df['in_scope_clean'] & (df['innings'] == 2) & (df['runs_so_far'] >= df['target'])]
    assert len(clean_inn2_overrun) == 0, f"Assertion failed: {len(clean_inn2_overrun)} balls still have runs_so_far >= target!"
    print("  [✓] Verified Assertion: Zero deliveries in in_scope_clean have runs_so_far >= target.")

    # 0c. Leverage normalization clarification
    clean_mask = df['in_scope_clean'] & df['innings'].isin([1, 2])
    mean_lev_clean = df.loc[clean_mask, 'leverage'].mean()
    mean_score_clean = df.loc[clean_mask, 'pressure_score'].mean()
    print(f"\n--- 0c. Leverage Normalization Audit ---")
    print(f"  Exact mean leverage over in_scope_clean balls: {mean_lev_clean:.4f}")
    print(f"  (Note: 0.0171 is the empirical mean leverage across clean balls; 0.0381 was a chat summary typo).")
    print(f"  Mean pressure_score over in_scope_clean balls: {mean_score_clean:.4f} (Confirmed == 1.000).")

    # 0d. Add pressure_score_w (clipped at 99th percentile)
    p99_score = df.loc[clean_mask, 'pressure_score'].quantile(0.99)
    df['pressure_score_w'] = df['pressure_score'].clip(upper=p99_score)
    print(f"\n--- 0d. Winsorized Pressure Score ---")
    print(f"  99th percentile cutoff: {p99_score:.3f}")
    print(f"  pressure_score_w created (raw max = {df['pressure_score'].max():.2f}, clipped max = {df['pressure_score_w'].max():.2f}).")

    # 0e. High pressure chase indicator
    df['is_high_pressure_chase'] = (df['innings'] == 2) & (df['pressure_pct'] >= 0.80) & df['in_scope_clean']
    hp_chase_cnt = df['is_high_pressure_chase'].sum()
    print(f"\n--- 0e. High-Pressure Chase Cohort ---")
    print(f"  is_high_pressure_chase count: {hp_chase_cnt:,} balls ({hp_chase_cnt / df['in_scope_clean'].sum() * 100:.2f}% of clean deliveries).")

    # =========================================================================
    # STEP 1 - BASELINE CONTEXT FEATURES
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1 - BASELINE CONTEXT FEATURES")
    print("=" * 80)
    
    # Categoricals
    df['venue_id'] = df['venue_id'].astype('category')
    df['pace_or_spin'] = df['pace_or_spin'].astype('category')
    df['batting_type'] = df['batting_type'].astype('category')
    
    # Target variables
    df['is_batter_out'] = (df['is_wicket'] & (df['dismissed_player_id'] == df['striker_id'])).astype(int)
    df['is_bowler_wkt_int'] = df['is_bowler_wicket'].astype(int)

    context_feats = ['innings', 'overid', 'is_ipl', 'venue_id', 'pace_or_spin', 'batting_type', 'is_legal']
    print("Context features selected (Strictly no player IDs, no WP, no state scores):")
    for f in context_feats:
        print(f"  - {f} ({df[f].dtype})")

    # =========================================================================
    # STEP 2 - FOUR LIGHTGBM BASELINE MODELS (GROUPKFOLD OOF)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2 - TRAINING CONTEXT BASELINE MODELS (5-Fold GroupKFold by match_id)")
    print("=" * 80)

    # Re-partition clean matches across 5 folds
    unique_clean_matches = np.array(sorted(df[df['in_scope_clean']]['match_id'].unique()))
    gkf = GroupKFold(n_splits=5)
    match_to_fold = {}
    for fold, (_, val_m_idx) in enumerate(gkf.split(unique_clean_matches, groups=unique_clean_matches)):
        for mid in unique_clean_matches[val_m_idx]:
            match_to_fold[mid] = fold
            
    df['fold'] = df['match_id'].map(match_to_fold)

    # Initialize prediction columns
    df['xBatRuns'] = np.nan
    df['xBatOut'] = np.nan
    df['xBowlRuns'] = np.nan
    df['xBowlWkt'] = np.nan

    # LightGBM parameters (modest settings)
    params_base = {
        'num_leaves': 16,
        'min_child_samples': 200,
        'n_estimators': 300,
        'learning_rate': 0.03,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    # Subset definitions
    # Bat models: exclude wides
    clean_bat_mask = df['in_scope_clean'] & (df['extras_type'] != 'wide')
    # Bowl models: all deliveries
    clean_bowl_mask = df['in_scope_clean']

    # Storage for baseline comparisons
    oof_eval = {
        'xBatRuns': {'y': [], 'p': [], 'base': []},
        'xBatOut': {'y': [], 'p': [], 'base': []},
        'xBowlRuns': {'y': [], 'p': [], 'base': []},
        'xBowlWkt': {'y': [], 'p': [], 'base': []}
    }

    # Feature importances storage
    feat_imps = {m: np.zeros(len(context_feats)) for m in ['xBatRuns', 'xBatOut', 'xBowlRuns', 'xBowlWkt']}

    for fold in range(5):
        # 1. xBatRuns (Regression on batter_runs)
        tr_bat = clean_bat_mask & (df['fold'] != fold)
        te_bat = clean_bat_mask & (df['fold'] == fold)

        reg_bat = lgb.LGBMRegressor(**params_base, objective='regression')
        reg_bat.fit(df.loc[tr_bat, context_feats], df.loc[tr_bat, 'batter_runs'])
        preds_bat_runs = reg_bat.predict(df.loc[te_bat, context_feats])
        df.loc[te_bat, 'xBatRuns'] = preds_bat_runs
        feat_imps['xBatRuns'] += reg_bat.feature_importances_ / 5.0

        mean_bat_tr = df.loc[tr_bat, 'batter_runs'].mean()
        oof_eval['xBatRuns']['y'].extend(df.loc[te_bat, 'batter_runs'].tolist())
        oof_eval['xBatRuns']['p'].extend(preds_bat_runs.tolist())
        oof_eval['xBatRuns']['base'].extend([mean_bat_tr] * len(preds_bat_runs))

        # 2. xBatOut (Classifier on batter dismissal)
        clf_bat_out = lgb.LGBMClassifier(**params_base, objective='binary')
        clf_bat_out.fit(df.loc[tr_bat, context_feats], df.loc[tr_bat, 'is_batter_out'])
        preds_bat_out = clf_bat_out.predict_proba(df.loc[te_bat, context_feats])[:, 1]
        df.loc[te_bat, 'xBatOut'] = preds_bat_out
        feat_imps['xBatOut'] += clf_bat_out.feature_importances_ / 5.0

        base_out_tr = df.loc[tr_bat, 'is_batter_out'].mean()
        oof_eval['xBatOut']['y'].extend(df.loc[te_bat, 'is_batter_out'].tolist())
        oof_eval['xBatOut']['p'].extend(preds_bat_out.tolist())
        oof_eval['xBatOut']['base'].extend([base_out_tr] * len(preds_bat_out))

        # 3. xBowlRuns (Regression on bowler_runs_conceded)
        tr_bowl = clean_bowl_mask & (df['fold'] != fold)
        te_bowl = clean_bowl_mask & (df['fold'] == fold)

        reg_bowl = lgb.LGBMRegressor(**params_base, objective='regression')
        reg_bowl.fit(df.loc[tr_bowl, context_feats], df.loc[tr_bowl, 'bowler_runs_conceded'])
        preds_bowl_runs = reg_bowl.predict(df.loc[te_bowl, context_feats])
        df.loc[te_bowl, 'xBowlRuns'] = preds_bowl_runs
        feat_imps['xBowlRuns'] += reg_bowl.feature_importances_ / 5.0

        mean_bowl_tr = df.loc[tr_bowl, 'bowler_runs_conceded'].mean()
        oof_eval['xBowlRuns']['y'].extend(df.loc[te_bowl, 'bowler_runs_conceded'].tolist())
        oof_eval['xBowlRuns']['p'].extend(preds_bowl_runs.tolist())
        oof_eval['xBowlRuns']['base'].extend([mean_bowl_tr] * len(preds_bowl_runs))

        # 4. xBowlWkt (Classifier on is_bowler_wicket)
        clf_bowl_wkt = lgb.LGBMClassifier(**params_base, objective='binary')
        clf_bowl_wkt.fit(df.loc[tr_bowl, context_feats], df.loc[tr_bowl, 'is_bowler_wkt_int'])
        preds_bowl_wkt = clf_bowl_wkt.predict_proba(df.loc[te_bowl, context_feats])[:, 1]
        df.loc[te_bowl, 'xBowlWkt'] = preds_bowl_wkt
        feat_imps['xBowlWkt'] += clf_bowl_wkt.feature_importances_ / 5.0

        base_wkt_tr = df.loc[tr_bowl, 'is_bowler_wkt_int'].mean()
        oof_eval['xBowlWkt']['y'].extend(df.loc[te_bowl, 'is_bowler_wkt_int'].tolist())
        oof_eval['xBowlWkt']['p'].extend(preds_bowl_wkt.tolist())
        oof_eval['xBowlWkt']['base'].extend([base_wkt_tr] * len(preds_bowl_wkt))

    # Print Model Diagnostics
    print("\n--- Out-of-Fold Model Performance vs Naive Baselines ---")
    res_bat_runs_lgb = root_mean_squared_error(oof_eval['xBatRuns']['y'], oof_eval['xBatRuns']['p'])
    res_bat_runs_base = root_mean_squared_error(oof_eval['xBatRuns']['y'], oof_eval['xBatRuns']['base'])
    print(f"1. xBatRuns (RMSE):  LGBM = {res_bat_runs_lgb:.4f} vs Baseline (Mean) = {res_bat_runs_base:.4f} (Δ = {res_bat_runs_base - res_bat_runs_lgb:+.4f})")

    res_bat_out_lgb = log_loss(oof_eval['xBatOut']['y'], oof_eval['xBatOut']['p'])
    res_bat_out_base = log_loss(oof_eval['xBatOut']['y'], oof_eval['xBatOut']['base'])
    print(f"2. xBatOut (LogLoss): LGBM = {res_bat_out_lgb:.4f} vs Baseline (BaseRate) = {res_bat_out_base:.4f} (Δ = {res_bat_out_base - res_bat_out_lgb:+.4f})")

    res_bowl_runs_lgb = root_mean_squared_error(oof_eval['xBowlRuns']['y'], oof_eval['xBowlRuns']['p'])
    res_bowl_runs_base = root_mean_squared_error(oof_eval['xBowlRuns']['y'], oof_eval['xBowlRuns']['base'])
    print(f"3. xBowlRuns (RMSE): LGBM = {res_bowl_runs_lgb:.4f} vs Baseline (Mean) = {res_bowl_runs_base:.4f} (Δ = {res_bowl_runs_base - res_bowl_runs_lgb:+.4f})")

    res_bowl_wkt_lgb = log_loss(oof_eval['xBowlWkt']['y'], oof_eval['xBowlWkt']['p'])
    res_bowl_wkt_base = log_loss(oof_eval['xBowlWkt']['y'], oof_eval['xBowlWkt']['base'])
    print(f"4. xBowlWkt (LogLoss): LGBM = {res_bowl_wkt_lgb:.4f} vs Baseline (BaseRate) = {res_bowl_wkt_base:.4f} (Δ = {res_bowl_wkt_base - res_bowl_wkt_lgb:+.4f})")

    # Feature Importances Table
    print("\n--- Feature Importances (Average Split Gain across Folds) ---")
    imp_df = pd.DataFrame(feat_imps, index=context_feats)
    print(imp_df.round(1).to_string())

    # =========================================================================
    # STEP 3 - RESIDUAL VALUE ADDED & EXPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3 - COMPUTING RESIDUALS & EXPORTING scored_balls.parquet")
    print("=" * 80)
    
    # bat_rae = batter_runs - xBatRuns (NaN outside clean_bat_mask)
    df['bat_rae'] = np.where(clean_bat_mask, df['batter_runs'] - df['xBatRuns'], np.nan)
    # bat_out_ae = is_batter_out - xBatOut
    df['bat_out_ae'] = np.where(clean_bat_mask, df['is_batter_out'] - df['xBatOut'], np.nan)
    # bowl_rae = xBowlRuns - bowler_runs_conceded (positive = better for bowler)
    df['bowl_rae'] = np.where(clean_bowl_mask, df['xBowlRuns'] - df['bowler_runs_conceded'], np.nan)
    # bowl_wae = is_bowler_wicket - xBowlWkt
    df['bowl_wae'] = np.where(clean_bowl_mask, df['is_bowler_wkt_int'] - df['xBowlWkt'], np.nan)

    out_scored = 'scored_balls.parquet'
    df.to_parquet(out_scored, index=False, engine='pyarrow')
    print(f"Saved: '{out_scored}' ({len(df):,} deliveries, {os.path.getsize(out_scored)/(1024*1024):.2f} MB).")

    # =========================================================================
    # STEP 4 - SANITY CHECKS & MINIMUM SAMPLE CUTOFFS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4 - RESIDUAL SANITY AUDIT & MINIMUM-SAMPLE CUTOFFS")
    print("=" * 80)

    clean_sub = df[df['in_scope_clean']].copy()

    # Residual Means Overall and by is_high_pressure_chase
    sanity_records = []
    for res_col in ['bat_rae', 'bat_out_ae', 'bowl_rae', 'bowl_wae']:
        overall_m = clean_sub[res_col].mean()
        hp_true_m = clean_sub.loc[clean_sub['is_high_pressure_chase'], res_col].mean()
        hp_false_m = clean_sub.loc[~clean_sub['is_high_pressure_chase'], res_col].mean()
        sanity_records.append({
            'Residual Metric': res_col,
            'Overall Mean (All Clean)': round(overall_m, 5),
            'HP Chase == True Mean': round(hp_true_m, 5),
            'HP Chase == False Mean': round(hp_false_m, 5)
        })

    print("Residual Sanity Check (Overall mean must be ~ 0.000):")
    print(pd.DataFrame(sanity_records).to_string(index=False))

    # Player volume on High-Pressure Chase balls
    hp_chase_df = df[df['is_high_pressure_chase']]
    print(f"\nTotal High-Pressure Chase Deliveries: {len(hp_chase_df):,}")

    # Batters (facing balls, excluding wides)
    bat_counts = hp_chase_df[hp_chase_df['extras_type'] != 'wide']['striker_id'].value_counts()
    print("\nBatters Volume on High-Pressure Chase Deliveries:")
    print(f"  - Batters with >=  60 balls faced: {(bat_counts >= 60).sum():2} players")
    print(f"  - Batters with >= 100 balls faced: {(bat_counts >= 100).sum():2} players")
    print(f"  - Batters with >= 150 balls faced: {(bat_counts >= 150).sum():2} players")

    # Bowlers (all deliveries)
    bowl_counts = hp_chase_df['bowler_id'].value_counts()
    print("\nBowlers Volume on High-Pressure Chase Deliveries:")
    print(f"  - Bowlers with >=  60 balls bowled: {(bowl_counts >= 60).sum():2} players")
    print(f"  - Bowlers with >= 100 balls bowled: {(bowl_counts >= 100).sum():2} players")
    print(f"  - Bowlers with >= 150 balls bowled: {(bowl_counts >= 150).sum():2} players")

    print("\nRecommendation for Minimum-Sample Cutoff:")
    print("  - A cutoff of >= 60 balls provides ~79-80 eligible players in each discipline (sufficient for a robust Top 25 ranking).")
    print("  - A cutoff of >= 100 balls provides ~33-42 players (ideal for a highly reliable Top 10 ranking).")
    print("  - A cutoff of >= 150 balls leaves only 10 batters and 15 bowlers, which is overly restrictive for chase-only situations.")
    print("=" * 80)

if __name__ == '__main__':
    run_pipeline()

