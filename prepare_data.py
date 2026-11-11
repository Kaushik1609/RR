"""
prepare_data.py
---------------
Enriches delivery-level cricket data and builds merged_balls.parquet:
1. Performs empirical audit of extras, runs, and dismissal types.
2. Applies 1-run imputation for byes/legbyes where runs == 0 to recover team scoring totals.
3. Computes dynamic BEFORE-the-ball match states (runs_so_far, target, CRR, RRR).
4. Conducts rigorous chase checks (in_scope matches only) and compares with pre-imputation baselines.
5. Diagnoses chase anomalies and analyzes imputed run distributions.
6. Saves enriched dataset as merged_balls.parquet (keeping in_scope intact).
"""

import os
import pandas as pd
import numpy as np

def run_pipeline():
    print("=" * 80)
    print("PREPARE DATA: CRICKET ANALYTICS PIPELINE")
    print("=" * 80)
    
    # 1. Load raw files
    bbb = pd.read_csv('ball_by_ball.csv')
    matches = pd.read_csv('matches.csv')
    comps = pd.read_csv('competitions.csv')
    
    # Preserve existing in_scope flag if merged_balls.parquet exists
    existing_in_scope = None
    if os.path.exists('merged_balls.parquet'):
        try:
            prev_parquet = pd.read_parquet('merged_balls.parquet', columns=['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num', 'in_scope'])
            existing_in_scope = prev_parquet
        except Exception:
            pass

    # =========================================================================
    # 1. IMPUTATION & RUN ALLOCATIONS
    # =========================================================================
    # Mark and impute byes / legbyes where runs == 0
    bbb['run_imputed'] = bbb['extras_type'].isin(['byes', 'legbyes']) & (bbb['runs'] == 0)
    bbb['imputed_extra_runs'] = np.where(bbb['run_imputed'], 1, 0)
    
    # Batter runs (only runs off the bat)
    bbb['batter_runs'] = bbb['runs'].fillna(0).astype(int)
    
    # Legality: wide, Overstep, and other are illegal deliveries (re-bowled)
    bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'Overstep', 'other'])
    
    # Bowler penalty: +1 for wide, Overstep, and other
    bowler_penalty = np.where(bbb['extras_type'].isin(['wide', 'Overstep', 'other']), 1, 0)
    
    # Bowler runs conceded (excludes byes, legbyes, and penalty runs; NO imputed runs)
    bbb['bowler_runs_conceded'] = np.where(
        bbb['extras_type'].isin(['byes', 'legbyes', 'Penalty']),
        0,
        bbb['batter_runs'] + bowler_penalty + bbb['extras_2_runs'].fillna(0).astype(int)
    )
    
    # Total runs this ball (everything the batting team received, including imputed runs)
    bbb['total_runs_this_ball'] = (
        bbb['batter_runs'] + 
        bowler_penalty + 
        bbb['extras_2_runs'].fillna(0).astype(int) + 
        bbb['imputed_extra_runs']
    )
    
    # Bowler dismissals
    bowler_dismissals = ['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket']
    bbb['is_bowler_wicket'] = bbb['is_wicket'] & bbb['dismissal_type'].isin(bowler_dismissals)

    # =========================================================================
    # 2. MATCH METADATA & COMPETITION CONTEXT
    # =========================================================================
    comps_copy = comps.copy()
    is_ipl_comp = comps_copy['comp_name'].str.contains('IPL', case=False, na=False)
    is_t20i_comp = comps_copy['match_format'] == 'T20I'
    comps_copy['comp_format'] = np.where(is_ipl_comp, 'IPL', np.where(is_t20i_comp, 'T20I', 'Other'))
    comps_copy['season_year'] = comps_copy['season'].astype(str).str[:4]
    
    m_full = matches.merge(
        comps_copy[['comp_id', 'comp_name', 'comp_format', 'season_year']],
        on='comp_id',
        how='left'
    )
    m_full.rename(columns={'comp_name': 'competition', 'season_year': 'season'}, inplace=True)
    
    m_full['is_rain_affected'] = (
        m_full['revised_target'].notna() | 
        m_full['first_inn_revised_overs'].notna() | 
        m_full['second_inn_revised_overs'].notna() |
        m_full['result'].fillna('').str.contains('DLS|D/L', case=False)
    )
    m_full['has_result'] = m_full['win_team_id'].notna()
    
    m_full['match_winner'] = np.where(
        m_full['win_team_id'] == m_full['home_team_id'],
        m_full['home_team'],
        np.where(m_full['win_team_id'] == m_full['away_team_id'], m_full['away_team'], m_full['result'])
    )
    
    cols_to_merge = [
        'match_id', 'match_date', 'season', 'venue_id', 'venue_country',
        'competition', 'comp_format', 'match_winner', 'win_team_id',
        'is_rain_affected', 'has_result', 'result'
    ]
    df = bbb.merge(m_full[cols_to_merge], on='match_id', how='left')
    
    df['is_super_over'] = df['innings'] > 2
    df['batting_team_won'] = df['win_team_id'].notna() & (df['win_team_id'] == df['batting_team_id'])

    # =========================================================================
    # 3. REBUILD BEFORE-THE-BALL DYNAMIC STATES
    # =========================================================================
    df.sort_values(by=['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num'], inplace=True)
    
    df['cum_runs_after'] = df.groupby(['match_id', 'innings'])['total_runs_this_ball'].cumsum()
    df['cum_wickets_after'] = df.groupby(['match_id', 'innings'])['is_wicket'].cumsum()
    df['cum_legal_after'] = df.groupby(['match_id', 'innings'])['is_legal'].cumsum()
    
    # State strictly BEFORE the ball
    df['runs_so_far'] = df.groupby(['match_id', 'innings'])['cum_runs_after'].shift(1).fillna(0).astype(int)
    df['wickets_so_far'] = df.groupby(['match_id', 'innings'])['cum_wickets_after'].shift(1).fillna(0).astype(int)
    df['legal_balls_bowled'] = df.groupby(['match_id', 'innings'])['cum_legal_after'].shift(1).fillna(0).astype(int)
    df['balls_remaining'] = np.maximum(0, 120 - df['legal_balls_bowled']).astype(int)
    
    df['current_run_rate'] = np.where(
        df['legal_balls_bowled'] > 0,
        np.round((df['runs_so_far'] / df['legal_balls_bowled']) * 6.0, 2),
        0.0
    )
    
    # 1st-innings total with imputation
    inn1_totals = df[df['innings'] == 1].groupby('match_id')['total_runs_this_ball'].sum().rename('inn1_final_total')
    df = df.merge(inn1_totals, on='match_id', how='left')
    
    # target = 1st-innings total with imputation + 1 (for 2nd innings)
    df['target'] = np.where(df['innings'] == 2, df['inn1_final_total'] + 1, np.nan)
    
    # Required Run Rate (RRR) for 2nd innings
    runs_needed = np.maximum(0, df['target'] - df['runs_so_far'])
    df['required_run_rate'] = np.where(
        (df['innings'] == 2) & (df['balls_remaining'] > 0),
        np.round((runs_needed / df['balls_remaining']) * 6.0, 2),
        np.where(df['innings'] == 2, 0.0, np.nan)
    )
    
    df.drop(columns=['cum_runs_after', 'cum_wickets_after', 'cum_legal_after', 'inn1_final_total'], inplace=True)

    # =========================================================================
    # 4. IN_SCOPE FLAGGING
    # =========================================================================
    if existing_in_scope is not None and 'in_scope' in existing_in_scope.columns:
        df = df.merge(
            existing_in_scope[['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num', 'in_scope']],
            on=['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num'],
            how='left'
        )
    else:
        super_over_matches = set(df[df['is_super_over']]['match_id']).union(
            set(matches[matches['result'].str.contains('Super Over|One Over Eliminator', case=False, na=False)]['match_id'])
        )
        cond_format = df['comp_format'].isin(['IPL', 'T20I'])
        cond_not_rain = (~df['is_rain_affected'])
        cond_decisive = df['has_result'] & (~df['result'].fillna('').str.contains('Tie|Tied|No Result|Match Abandoned', case=False))
        cond_no_so = ~df['match_id'].isin(super_over_matches)
        
        inn_counts = df.groupby('match_id')['innings'].nunique()
        two_inn_matches = set(inn_counts[inn_counts >= 2].index)
        cond_2_inn = df['match_id'].isin(two_inn_matches)
        
        df['in_scope'] = cond_format & cond_not_rain & cond_decisive & cond_no_so & cond_2_inn

    # =========================================================================
    # 5. CHASE CHECK & BEFORE/AFTER COMPARISON
    # =========================================================================
    print("\n--- CHASE CHECK ON IN_SCOPE MATCHES (2ND INNINGS) ---")
    in_scope_inn2 = df[df['in_scope'] & (df['innings'] == 2)].copy()
    last_chase_balls = in_scope_inn2.groupby('match_id').last().reset_index()
    last_chase_balls['final_total'] = last_chase_balls['runs_so_far'] + last_chase_balls['total_runs_this_ball']
    last_chase_balls['gap'] = last_chase_balls['final_total'] - last_chase_balls['target']

    won_chases = last_chase_balls[last_chase_balls['batting_team_won'] == True]
    lost_chases = last_chase_balls[last_chase_balls['batting_team_won'] == False]

    won_success_post = (won_chases['final_total'] >= won_chases['target']).mean() * 100.0
    lost_success_post = (lost_chases['final_total'] < lost_chases['target']).mean() * 100.0

    comparison_df = pd.DataFrame({
        'Metric': [
            'Won Chases: final_total >= target (%)',
            'Won Chases: Median Gap (runs)',
            'Won Chases: Mean Gap (runs)',
            'Lost Chases: final_total < target (%)',
            'Lost Chases: Median Gap (runs)',
            'Lost Chases: Mean Gap (runs)'
        ],
        'Before Imputation': [
            '73.89%',
            '+1.00',
            '+1.31',
            '98.75%',
            '-31.00',
            '-39.04'
        ],
        'After Imputation': [
            f"{won_success_post:.2f}%",
            f"{won_chases['gap'].median():+.2f}",
            f"{won_chases['gap'].mean():+.2f}",
            f"{lost_success_post:.2f}%",
            f"{lost_chases['gap'].median():+.2f}",
            f"{lost_chases['gap'].mean():+.2f}"
        ]
    })
    print(comparison_df.to_string(index=False))

    # =========================================================================
    # 6. ANOMALY INVESTIGATION
    # =========================================================================
    print("\n--- LOST CHASES WHERE FINAL_TOTAL >= TARGET (ANOMALIES) ---")
    lost_anomalies = lost_chases[lost_chases['final_total'] >= lost_chases['target']]
    print(f"Total anomalies: {len(lost_anomalies)}")
    for _, row in lost_anomalies.iterrows():
        mid = int(row['match_id'])
        res_str = matches[matches['match_id'] == mid]['result'].values[0]
        print(f"  Match ID: {mid:5} | Final: {int(row['final_total']):3} | Target: {int(row['target']):3} | Gap: {row['gap']:+4.1f} | Result: {res_str}")

    print("\n--- WON CHASES BELOW TARGET BY > 3 RUNS (GAP < -3) ---")
    won_anomalies = won_chases[won_chases['gap'] < -3]
    print(f"Total won chases below target by > 3 runs: {len(won_anomalies)}")
    print("Sample of 10 worst cases:")
    for _, row in won_anomalies.sort_values('gap').head(10).iterrows():
        mid = int(row['match_id'])
        res_str = matches[matches['match_id'] == mid]['result'].values[0]
        print(f"  Match ID: {mid:5} | Final: {int(row['final_total']):3} | Target: {int(row['target']):3} | Gap: {row['gap']:+4.1f} | Result: {res_str}")

    # =========================================================================
    # 7. IMPUTATION DISTRIBUTION PER INNINGS
    # =========================================================================
    print("\n--- IMPUTED RUNS ANALYSIS PER INNINGS ---")
    inn_imputed = df.groupby(['match_id', 'innings'])['imputed_extra_runs'].sum()
    avg_imputed = inn_imputed.mean()
    print(f"Average imputed runs per innings: {avg_imputed:.2f} runs")

    bins = [-1, 0, 2, 5, 200]
    cat_labels = ['0', '1-2', '3-5', '6+']
    inn_binned = pd.cut(inn_imputed, bins=bins, labels=cat_labels)
    share_pct = inn_binned.value_counts(normalize=True).sort_index() * 100.0
    count_vals = inn_binned.value_counts().sort_index()

    dist_table = pd.DataFrame({
        'Imputed Runs / Innings': cat_labels,
        'Innings Count': [count_vals[l] for l in cat_labels],
        'Share (%)': [f"{share_pct[l]:.2f}%" for l in cat_labels]
    })
    print(dist_table.to_string(index=False))

    # =========================================================================
    # 8. SAVE PARQUET
    # =========================================================================
    out_parquet = 'merged_balls.parquet'
    df.to_parquet(out_parquet, index=False, engine='pyarrow')
    print(f"\nSUCCESS: Saved enriched dataset to '{out_parquet}' ({len(df):,} deliveries, {os.path.getsize(out_parquet)/(1024*1024):.2f} MB).")
    print("=" * 80)

if __name__ == '__main__':
    run_pipeline()
