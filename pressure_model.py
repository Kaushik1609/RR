"""
pressure_model.py
-----------------
Win Probability & Leverage-based Pressure Modeling Pipeline:
0. Identifies and flags match-level data-quality anomalies (chase gaps).
1. Engineers before-the-ball match states, phases, and competition indicators.
2. Trains two Win Probability LightGBM classifiers (Innings 1 & Innings 2) with
   monotone constraints using 5-fold GroupKFold by match_id.
3. Computes Out-Of-Fold WP predictions and estimates next-ball Leverage Index across
   7 empirical next-state outcomes (wicket, 0, 1, 2, 3, 4, 6).
4. Generates Pressure Score, Closeness, and Percentile Ranks (is_high_pressure >= 0.80).
5. Produces comprehensive validation tables (by decile, phase-stratified, top balls).
6. Saves diagnostic plots (calibration_wp.png, pressure_by_decile.png) and
   exports enriched dataset to pressure_balls.parquet.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure required packages
for pkg in ['lightgbm', 'sklearn', 'matplotlib']:
    try:
        __import__(pkg)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {pkg}")

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

def run_pressure_model():
    parquet_path = 'merged_balls.parquet'
    if not os.path.exists(parquet_path):
        print(f"Error: '{parquet_path}' not found. Please run prepare_data.py first.")
        return

    print("=" * 80)
    print("WIN-PROBABILITY & LEVERAGE PRESSURE MODEL (SupeRR Selector Season 2)")
    print("=" * 80)
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} deliveries across {df['match_id'].nunique()} matches.")

    # =========================================================================
    # 0. DATA QUALITY FLAG
    # =========================================================================
    # For in_scope matches, compute chase gap on last 2nd-innings ball
    in_scope_m = df[df['in_scope']].copy()
    inn2_last = in_scope_m[in_scope_m['innings'] == 2].sort_values(
        ['match_id', 'overid', 'over_ball_num', 'innings_ball_num']
    ).groupby('match_id').last().reset_index()

    inn2_last['final_total'] = inn2_last['runs_so_far'] + inn2_last['total_runs_this_ball']
    inn2_last['chase_gap'] = inn2_last['final_total'] - inn2_last['target']

    # Flag anomaly: won chase with gap < -3 OR lost chase with gap >= 0
    bad_chases = inn2_last[
        ((inn2_last['batting_team_won'] == True) & (inn2_last['chase_gap'] < -3)) |
        ((inn2_last['batting_team_won'] == False) & (inn2_last['chase_gap'] >= 0))
    ]
    bad_match_ids = set(bad_chases['match_id'])
    
    df['data_quality_flag'] = df['match_id'].isin(bad_match_ids)
    df['in_scope_clean'] = df['in_scope'] & (~df['data_quality_flag'])
    
    print("\n--- 0. DATA QUALITY AUDIT ---")
    print(f"Flagged {len(bad_match_ids)} anomaly matches (data_quality_flag = True):")
    print(f"  - Won chases with recorded gap < -3: {((inn2_last['batting_team_won'] == True) & (inn2_last['chase_gap'] < -3)).sum()} matches")
    print(f"  - Lost chases with recorded gap >= 0: {((inn2_last['batting_team_won'] == False) & (inn2_last['chase_gap'] >= 0)).sum()} matches")
    print(f"Retained in_scope_clean: {df[df['in_scope_clean']]['match_id'].nunique()} matches ({df['in_scope_clean'].sum():,} deliveries)")

    # =========================================================================
    # 1. FEATURE ENGINEERING (STATE BEFORE THE BALL)
    # =========================================================================
    df['wickets_in_hand'] = np.maximum(0, 10 - df['wickets_so_far']).astype(int)
    df['is_ipl'] = (df['comp_format'] == 'IPL').astype(int)
    df['runs_needed'] = np.where(df['innings'] == 2, np.maximum(0, df['target'] - df['runs_so_far']), np.nan)
    
    df['phase'] = np.where(
        df['overid'] <= 6, 'powerplay',
        np.where(df['overid'] <= 15, 'middle', 'death')
    )

    # =========================================================================
    # 2. LIGHTGBM WIN PROBABILITY MODELS (OOF GROUPKFOLD)
    # =========================================================================
    print("\n--- 2. TRAINING WIN-PROBABILITY CLASSIFIERS ---")
    
    # Clean legal subset for training
    clean_legal = df[df['in_scope_clean'] & df['is_legal'] & df['innings'].isin([1, 2])].copy()
    
    # Establish consistent 5-fold match splits on in_scope_clean
    unique_clean_matches = np.array(sorted(clean_legal['match_id'].unique()))
    gkf = GroupKFold(n_splits=5)
    
    match_to_fold = {}
    for fold, (_, val_m_idx) in enumerate(gkf.split(unique_clean_matches, groups=unique_clean_matches)):
        for mid in unique_clean_matches[val_m_idx]:
            match_to_fold[mid] = fold
            
    # For out-of-scope / flagged matches, assign fold deterministically
    all_unique_matches = df['match_id'].unique()
    for mid in all_unique_matches:
        if mid not in match_to_fold:
            match_to_fold[mid] = int(mid) % 5

    df['fold'] = df['match_id'].map(match_to_fold)

    # Feature definitions and monotone constraints
    feats_inn2 = ['runs_needed', 'balls_remaining', 'wickets_in_hand', 'required_run_rate', 'current_run_rate', 'target', 'is_ipl']
    mono_inn2 = [-1, 1, 1, -1, 0, 0, 0]

    feats_inn1 = ['runs_so_far', 'wickets_in_hand', 'balls_remaining', 'current_run_rate', 'is_ipl']
    mono_inn1 = [1, 1, 0, 0, 0]

    params_lgb = {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'num_leaves': 8,
        'min_child_samples': 300,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_lambda': 5,
        'random_state': 42,
        'objective': 'binary',
        'verbose': -1,
        'n_jobs': -1
    }

    # Storage for predictions and fold models
    df['wp'] = np.nan
    df['wp_baseline'] = np.nan

    models_inn1 = {}
    models_inn2 = {}

    # Store OOF validation series for reporting
    oof_eval = {'inn1': {'y': [], 'p_lgb': [], 'p_base': []},
                'inn2': {'y': [], 'p_lgb': [], 'p_base': []}}

    for fold in range(5):
        # ----------------- INNINGS 1 MODEL -----------------
        train_mask_inn1 = df['in_scope_clean'] & df['is_legal'] & (df['innings'] == 1) & (df['fold'] != fold)
        test_mask_inn1 = (df['innings'] == 1) & (df['fold'] == fold)
        
        clf_inn1 = lgb.LGBMClassifier(**params_lgb, monotone_constraints=mono_inn1)
        X_tr1 = df.loc[train_mask_inn1, feats_inn1]
        y_tr1 = df.loc[train_mask_inn1, 'batting_team_won'].astype(int)
        clf_inn1.fit(X_tr1, y_tr1)
        models_inn1[fold] = clf_inn1

        # Predict for all deliveries in held-out matches
        X_te1 = df.loc[test_mask_inn1, feats_inn1]
        preds_inn1 = clf_inn1.predict_proba(X_te1)[:, 1]
        df.loc[test_mask_inn1, 'wp'] = preds_inn1

        # Baseline Innings 1: Constant base rate
        base_rate_inn1 = y_tr1.mean()
        df.loc[test_mask_inn1, 'wp_baseline'] = base_rate_inn1

        # OOF evaluation mask (clean legal only)
        eval_mask_inn1 = test_mask_inn1 & df['in_scope_clean'] & df['is_legal']
        oof_eval['inn1']['y'].extend(df.loc[eval_mask_inn1, 'batting_team_won'].astype(int).tolist())
        oof_eval['inn1']['p_lgb'].extend(df.loc[eval_mask_inn1, 'wp'].tolist())
        oof_eval['inn1']['p_base'].extend(df.loc[eval_mask_inn1, 'wp_baseline'].tolist())

        # ----------------- INNINGS 2 MODEL -----------------
        train_mask_inn2 = df['in_scope_clean'] & df['is_legal'] & (df['innings'] == 2) & (df['fold'] != fold)
        test_mask_inn2 = (df['innings'] == 2) & (df['fold'] == fold)

        clf_inn2 = lgb.LGBMClassifier(**params_lgb, monotone_constraints=mono_inn2)
        X_tr2 = df.loc[train_mask_inn2, feats_inn2]
        y_tr2 = df.loc[train_mask_inn2, 'batting_team_won'].astype(int)
        clf_inn2.fit(X_tr2, y_tr2)
        models_inn2[fold] = clf_inn2

        X_te2 = df.loc[test_mask_inn2, feats_inn2]
        preds_inn2 = clf_inn2.predict_proba(X_te2)[:, 1]
        df.loc[test_mask_inn2, 'wp'] = preds_inn2

        # Baseline Innings 2: Logistic Regression on runs_needed, balls_remaining, wickets_in_hand
        lr_feats = ['runs_needed', 'balls_remaining', 'wickets_in_hand']
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(df.loc[train_mask_inn2, lr_feats], y_tr2)
        df.loc[test_mask_inn2, 'wp_baseline'] = lr.predict_proba(X_te2[lr_feats])[:, 1]

        eval_mask_inn2 = test_mask_inn2 & df['in_scope_clean'] & df['is_legal']
        oof_eval['inn2']['y'].extend(df.loc[eval_mask_inn2, 'batting_team_won'].astype(int).tolist())
        oof_eval['inn2']['p_lgb'].extend(df.loc[eval_mask_inn2, 'wp'].tolist())
        oof_eval['inn2']['p_base'].extend(df.loc[eval_mask_inn2, 'wp_baseline'].tolist())

    # Report OOF Metrics
    metrics_summary = []
    for inn_key, inn_name in [('inn1', 'Innings 1'), ('inn2', 'Innings 2')]:
        y_true = np.array(oof_eval[inn_key]['y'])
        p_lgb = np.array(oof_eval[inn_key]['p_lgb'])
        p_base = np.array(oof_eval[inn_key]['p_base'])
        
        metrics_summary.append({
            'Innings': inn_name,
            'Model': 'LightGBM (OOF)',
            'Log-Loss': round(log_loss(y_true, p_lgb), 4),
            'AUC': round(roc_auc_score(y_true, p_lgb), 4),
            'Brier Score': round(brier_score_loss(y_true, p_lgb), 4)
        })
        metrics_summary.append({
            'Innings': inn_name,
            'Model': 'Baseline',
            'Log-Loss': round(log_loss(y_true, p_base), 4),
            'AUC': round(roc_auc_score(y_true, p_base), 4) if len(set(p_base)) > 1 else 0.5000,
            'Brier Score': round(brier_score_loss(y_true, p_base), 4)
        })

    print(pd.DataFrame(metrics_summary).to_string(index=False))

    # Save calibration curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=250)
    for i, (inn_key, inn_name) in enumerate([('inn1', 'Innings 1'), ('inn2', 'Innings 2')]):
        y_true = np.array(oof_eval[inn_key]['y'])
        p_lgb = np.array(oof_eval[inn_key]['p_lgb'])
        p_base = np.array(oof_eval[inn_key]['p_base'])
        
        prob_true_lgb, prob_pred_lgb = calibration_curve(y_true, p_lgb, n_bins=10)
        axes[i].plot(prob_pred_lgb, prob_true_lgb, marker='o', lw=2, label='LightGBM (OOF)', color='#1f77b4')
        
        if len(set(p_base)) > 1:
            prob_true_b, prob_pred_b = calibration_curve(y_true, p_base, n_bins=10)
            axes[i].plot(prob_pred_b, prob_true_b, marker='s', lw=1.5, ls='--', label='Baseline', color='#ff7f0e')
        else:
            axes[i].axvline(p_base[0], color='#ff7f0e', ls='--', label='Constant Baseline')
            
        axes[i].plot([0, 1], [0, 1], 'k:', lw=1.2, label='Perfect Calibration')
        axes[i].set_title(f'Win Probability Calibration: {inn_name}', fontweight='bold')
        axes[i].set_xlabel('Predicted Win Probability')
        axes[i].set_ylabel('Empirical Win Fraction')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='lower right')

    plt.tight_layout()
    calib_file = 'calibration_wp.png'
    plt.savefig(calib_file)
    plt.close()
    print(f"Saved: '{calib_file}'")

    # =========================================================================
    # 3. LEVERAGE & PRESSURE SCORE COMPUTATION
    # =========================================================================
    print("\n--- 3. ESTIMATING LEVERAGE & PRESSURE SCORES ---")
    
    # Next-ball outcome probabilities from clean legal balls
    clean_legal['outcome'] = np.where(
        clean_legal['is_wicket'],
        'wicket',
        np.where(clean_legal['total_runs_this_ball'].isin([0, 1, 2, 3, 4, 6]), clean_legal['total_runs_this_ball'].astype(str), np.nan)
    )
    outcome_probs = clean_legal.dropna(subset=['outcome']).groupby(['innings', 'phase'])['outcome'].value_counts(normalize=True).to_dict()

    # The 7 discrete outcomes
    outcomes = ['wicket', '0', '1', '2', '3', '4', '6']
    
    # Pre-allocate leverage
    df['leverage'] = np.nan
    
    # Process each fold
    for fold in range(5):
        clf_1 = models_inn1[fold]
        clf_2 = models_inn2[fold]
        
        fold_mask = (df['fold'] == fold) & df['innings'].isin([1, 2])
        sub_df = df[fold_mask]
        if len(sub_df) == 0:
            continue
            
        sub_indices = sub_df.index
        n_balls = len(sub_df)
        
        # Current WP
        curr_wp = sub_df['wp'].values
        
        inn = sub_df['innings'].values
        ph = sub_df['phase'].values
        r_so_far = sub_df['runs_so_far'].values
        w_so_far = sub_df['wickets_so_far'].values
        b_rem = sub_df['balls_remaining'].values
        targ = sub_df['target'].values
        is_ipl_val = sub_df['is_ipl'].values
        
        # Accumulate weighted absolute difference: sum_o p(o) * |wp_next - wp|
        lev_accum = np.zeros(n_balls, dtype=np.float32)
        
        for outcome in outcomes:
            is_w = (outcome == 'wicket')
            r_val = 0 if is_w else int(outcome)
            
            # Outcome probabilities array for each ball
            p_outcome = np.array([outcome_probs.get((inn[j], ph[j], outcome), 0.0) for j in range(n_balls)], dtype=np.float32)
            
            # Next states
            r_next = r_so_far + r_val
            w_next = w_so_far + (1 if is_w else 0)
            b_rem_next = np.maximum(0, b_rem - 1)
            b_bowled_next = 120 - b_rem_next
            w_hand_next = np.maximum(0, 10 - w_next)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                crr_next = np.where(b_bowled_next > 0, (r_next / b_bowled_next) * 6.0, 0.0)
            
            # Separate by innings for model inference
            mask_1 = (inn == 1)
            mask_2 = (inn == 2)
            
            wp_next = np.zeros(n_balls, dtype=np.float32)
            
            # --- Innings 1 evaluation ---
            if np.any(mask_1):
                X_next_1 = pd.DataFrame({
                    'runs_so_far': r_next[mask_1],
                    'wickets_in_hand': w_hand_next[mask_1],
                    'balls_remaining': b_rem_next[mask_1],
                    'current_run_rate': crr_next[mask_1],
                    'is_ipl': is_ipl_val[mask_1]
                })
                wp_next[mask_1] = clf_1.predict_proba(X_next_1)[:, 1]
                
            # --- Innings 2 evaluation ---
            if np.any(mask_2):
                r_need_next = np.maximum(0, targ[mask_2] - r_next[mask_2])
                with np.errstate(divide='ignore', invalid='ignore'):
                    rrr_next = np.where(b_rem_next[mask_2] > 0, (r_need_next / b_rem_next[mask_2]) * 6.0, 0.0)
                
                # Terminal condition masks
                win_term = (r_next[mask_2] >= targ[mask_2])
                loss_term = (w_next[mask_2] >= 10) | ((b_rem_next[mask_2] == 0) & (r_next[mask_2] < targ[mask_2]))
                active = (~win_term) & (~loss_term)
                
                wp_sub_2 = np.zeros(np.sum(mask_2), dtype=np.float32)
                wp_sub_2[win_term] = 1.0
                wp_sub_2[loss_term] = 0.0
                
                if np.any(active):
                    X_next_2 = pd.DataFrame({
                        'runs_needed': r_need_next[active],
                        'balls_remaining': b_rem_next[mask_2][active],
                        'wickets_in_hand': w_hand_next[mask_2][active],
                        'required_run_rate': rrr_next[active],
                        'current_run_rate': crr_next[mask_2][active],
                        'target': targ[mask_2][active],
                        'is_ipl': is_ipl_val[mask_2][active]
                    })
                    wp_sub_2[active] = clf_2.predict_proba(X_next_2)[:, 1]
                    
                wp_next[mask_2] = wp_sub_2
                
            lev_accum += p_outcome * np.abs(wp_next - curr_wp)
            
        df.loc[sub_indices, 'leverage'] = lev_accum

    # Compute Pressure Score: leverage / mean(leverage on in_scope_clean)
    clean_mask = df['in_scope_clean'] & df['innings'].isin([1, 2])
    mean_leverage = df.loc[clean_mask, 'leverage'].mean()
    df['pressure_score'] = df['leverage'] / mean_leverage
    
    # Diagnostic closeness
    df['closeness'] = 1.0 - 2.0 * np.abs(df['wp'] - 0.5)

    # Percentile rank within each innings (in_scope_clean only)
    df['pressure_pct'] = np.nan
    for inn_val in [1, 2]:
        inn_clean_mask = clean_mask & (df['innings'] == inn_val)
        df.loc[inn_clean_mask, 'pressure_pct'] = df.loc[inn_clean_mask, 'pressure_score'].rank(pct=True)

    # High pressure flag: pressure_pct >= 0.80
    df['is_high_pressure'] = df['pressure_pct'] >= 0.80

    print(f"Mean Baseline Leverage: {mean_leverage:.4f}")
    print(f"High-Pressure Deliveries (is_high_pressure == True): {df['is_high_pressure'].sum():,} ({df.loc[clean_mask, 'is_high_pressure'].mean()*100:.1f}% of clean balls)")

    # =========================================================================
    # 4. VALIDATION TABLES & REPORTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. VALIDATION TABLES (IN_SCOPE_CLEAN)")
    print("=" * 80)
    
    # Table A: By Pressure Decile
    clean_df = df[clean_mask].copy()
    clean_df['pressure_decile'] = pd.qcut(clean_df['pressure_pct'], q=10, labels=range(1, 11))

    for inn_val in [1, 2]:
        sub_inn = clean_df[clean_df['innings'] == inn_val]
        decile_table = sub_inn.groupby('pressure_decile', observed=False).agg(
            balls=('runs', 'count'),
            runs_per_ball=('total_runs_this_ball', 'mean'),
            dot_pct=('total_runs_this_ball', lambda x: (x == 0).mean() * 100),
            boundary_pct=('is_four', lambda x: (sub_inn.loc[x.index, 'is_four'] | sub_inn.loc[x.index, 'is_six']).mean() * 100),
            wicket_pct=('is_wicket', lambda x: x.mean() * 100),
            mean_wp=('wp', 'mean'),
            mean_balls_rem=('balls_remaining', 'mean'),
            mean_pressure=('pressure_score', 'mean')
        ).reset_index()

        for c in ['runs_per_ball', 'mean_wp', 'mean_balls_rem', 'mean_pressure']:
            decile_table[c] = decile_table[c].round(2)
        for c in ['dot_pct', 'boundary_pct', 'wicket_pct']:
            decile_table[c] = decile_table[c].round(1)

        print(f"\n>>> Table 4a. Pressure Decile Breakdown (Innings {inn_val}):")
        print(decile_table.to_string(index=False))

    # Table B: High vs Ordinary Pressure within each phase
    clean_df['pressure_regime'] = np.where(clean_df['is_high_pressure'], 'High Pressure (Top 20%)', 'Ordinary Pressure')
    phase_order = ['powerplay', 'middle', 'death']

    for inn_val in [1, 2]:
        sub_inn = clean_df[clean_df['innings'] == inn_val]
        phase_table = sub_inn.groupby(['phase', 'pressure_regime'], observed=False).agg(
            balls=('runs', 'count'),
            runs_per_ball=('total_runs_this_ball', 'mean'),
            dot_pct=('total_runs_this_ball', lambda x: (x == 0).mean() * 100),
            boundary_pct=('is_four', lambda x: (sub_inn.loc[x.index, 'is_four'] | sub_inn.loc[x.index, 'is_six']).mean() * 100),
            wicket_pct=('is_wicket', lambda x: x.mean() * 100),
            mean_wp=('wp', 'mean'),
            mean_pressure=('pressure_score', 'mean')
        ).reset_index()

        phase_table['phase'] = pd.Categorical(phase_table['phase'], categories=phase_order, ordered=True)
        phase_table = phase_table.sort_values(['phase', 'pressure_regime'])

        for c in ['runs_per_ball', 'mean_wp', 'mean_pressure']:
            phase_table[c] = phase_table[c].round(2)
        for c in ['dot_pct', 'boundary_pct', 'wicket_pct']:
            phase_table[c] = phase_table[c].round(1)

        print(f"\n>>> Table 4b. Phase-Stratified High vs Ordinary Pressure (Innings {inn_val}):")
        print(phase_table.to_string(index=False))

    # Table C: Sample Balls and Distribution
    cols_sample = ['match_id', 'innings', 'overid', 'over_ball_num', 'runs_so_far', 'wickets_so_far', 'target', 'wp', 'pressure_score']
    
    top5_balls = clean_df.sort_values('pressure_score', ascending=False)[cols_sample].head(5)
    np.random.seed(42)
    low_balls_pool = clean_df[clean_df['pressure_pct'] <= 0.20]
    random5_low = low_balls_pool.sample(5, random_state=42)[cols_sample]

    for c in ['wp', 'pressure_score']:
        top5_balls[c] = top5_balls[c].round(3)
        random5_low[c] = random5_low[c].round(3)

    print("\n>>> Table 4c-1. 5 Highest-Pressure Deliveries in Entire Dataset:")
    print(top5_balls.to_string(index=False))

    print("\n>>> Table 4c-2. 5 Random Low-Pressure Deliveries:")
    print(random5_low.to_string(index=False))

    # High pressure share by phase and innings
    share_breakdown = clean_df.groupby(['innings', 'phase'])['is_high_pressure'].agg(
        total_balls='count',
        hp_balls='sum',
        hp_share=lambda x: x.mean() * 100
    ).reset_index()
    share_breakdown['hp_share'] = share_breakdown['hp_share'].round(1)
    print("\n>>> Share of High-Pressure Deliveries by Phase & Innings:")
    print(share_breakdown.to_string(index=False))

    # Save pressure_by_decile.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=250)
    for i, inn_val in enumerate([1, 2]):
        sub_inn = clean_df[clean_df['innings'] == inn_val]
        dec_agg = sub_inn.groupby('pressure_decile', observed=False).agg(
            dot_pct=('total_runs_this_ball', lambda x: (x == 0).mean() * 100),
            boundary_pct=('is_four', lambda x: (sub_inn.loc[x.index, 'is_four'] | sub_inn.loc[x.index, 'is_six']).mean() * 100),
            wicket_pct=('is_wicket', lambda x: x.mean() * 100),
            runs_per_ball=('total_runs_this_ball', 'mean')
        ).reset_index()
        
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        ax1.plot(dec_agg['pressure_decile'], dec_agg['dot_pct'], marker='o', color='#2ca02c', label='Dot Ball %')
        ax1.plot(dec_agg['pressure_decile'], dec_agg['boundary_pct'], marker='s', color='#1f77b4', label='Boundary %')
        ax1.plot(dec_agg['pressure_decile'], dec_agg['wicket_pct'], marker='^', color='#d62728', label='Wicket %')
        ax2.plot(dec_agg['pressure_decile'], dec_agg['runs_per_ball'], marker='d', color='#9467bd', ls='--', label='Runs / Ball')
        
        ax1.set_title(f'Cricket Dynamics Across Pressure Deciles: Innings {inn_val}', fontweight='bold')
        ax1.set_xlabel('Pressure Decile (1 = Lowest, 10 = Highest)')
        ax1.set_ylabel('Percentage (%)')
        ax2.set_ylabel('Runs per Ball (Purple)', color='#9467bd')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

    plt.tight_layout()
    dec_file = 'pressure_by_decile.png'
    plt.savefig(dec_file)
    plt.close()
    print(f"\nSaved: '{dec_file}'")

    # =========================================================================
    # 5. EXPORT PRESSURE_BALLS.PARQUET
    # =========================================================================
    out_parquet = 'pressure_balls.parquet'
    df.to_parquet(out_parquet, index=False, engine='pyarrow')
    print(f"\nSUCCESS: Exported {len(df):,} deliveries to '{out_parquet}' ({os.path.getsize(out_parquet)/(1024*1024):.2f} MB)")
    print("New columns included: wp, leverage, pressure_score, closeness, pressure_pct, is_high_pressure, phase, data_quality_flag, in_scope_clean.")
    print("=" * 80)

if __name__ == '__main__':
    run_pressure_model()
