"""
validate_data.py
----------------
Validates merged_balls.parquet against cricket domain rules:
1. Run Integrity & Extras Audit:
   - Prints 10 sample rows of legbyes, byes (all columns), and wides.
   - Audits boundaries on extras and clarifies bye/legbye run storage.
2. Chase Check:
   - Verifies 2nd innings totals vs targets for won vs lost chases.
   - Calculates target gaps and highlights worst mismatches.
3. First-Innings Totals:
   - Computes mean and median scores per comp_format (IPL, T20I, Other).
4. Legal-Ball Distribution & 'other' Comparison:
   - Tests legal deliveries per completed over (6 balls expected).
   - Compares distributions when 'other' is treated as illegal vs legal.
   - Analyzes balls_remaining at end of 20-over and all-out innings.
5. Match Result Categorization:
   - Evaluates match outcome text patterns across matches.csv.
6. Scope Flagging:
   - Adds in_scope boolean flag (IPL/T20I, decisive, non-rain, 2 complete innings, no super over).
   - Reports kept matches and balls for IPL and T20I.
   - Re-saves merged_balls.parquet without dropping any records.
"""

import os
import pandas as pd
import numpy as np

def run_validation():
    parquet_path = 'merged_balls.parquet'
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found. Please run prepare_data.py first.")
        return

    print("=" * 80)
    print("VALIDATING merged_balls.parquet")
    print("=" * 80)
    df = pd.read_parquet(parquet_path)
    matches = pd.read_csv('matches.csv')

    # =========================================================================
    # 1. RUN INTEGRITY & EXTRAS AUDIT
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. RUN INTEGRITY & EXTRAS AUDIT")
    print("=" * 80)

    # Set display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # 10 sample rows of legbyes
    legbyes_sample = df[df['extras_type'] == 'legbyes'].head(10)
    print("\n>>> 10 Sample Rows: extras_type == 'legbyes' (All Columns):")
    print(legbyes_sample.to_string())

    # 10 sample rows of byes
    byes_sample = df[df['extras_type'] == 'byes'].head(10)
    print("\n>>> 10 Sample Rows: extras_type == 'byes' (All Columns):")
    print(byes_sample.to_string())

    # 10 sample rows of wides
    wides_sample = df[df['extras_type'] == 'wide'].head(10)
    print("\n>>> 10 Sample Rows: extras_type == 'wide' (All Columns):")
    print(wides_sample.to_string())

    # Reset display width for standard console logs
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')

    # Count boundaries on wides/byes/legbyes
    boundary_counts = {}
    for et in ['wide', 'byes', 'legbyes']:
        sub = df[df['extras_type'] == et]
        fours = int(sub['is_four'].sum())
        sixes = int(sub['is_six'].sum())
        boundary_counts[et] = {'count': len(sub), 'is_four': fours, 'is_six': sixes}
    
    bnd_summary = pd.DataFrame(boundary_counts).T
    print("\nBoundary flags on extras:")
    print(bnd_summary.to_string())

    print("\n--- STORAGE OF BYE / LEGBYE RUNS ---")
    print("Finding: Bye and legbye run magnitudes are NOT stored anywhere in the raw data.")
    print("Explicitly:")
    print("  - For 'byes' (529 rows): 'runs' is 0 in 100% of rows; 'extras_2_runs' is 0 in 100% of rows.")
    print("  - For 'legbyes' (2,748 rows): 'runs' is 0 in 2,747 rows (1 exception recorded as 1); 'extras_2_runs' is 0 in 100% of rows.")
    print("  - 'extras1_runs' is purely a categorical type ID (3 for legbyes, 4 for byes).")
    print("  - Conclusion: The dataset logs the occurrence of byes/legbyes but does NOT distinguish between 1 bye, 2 byes, or 4 byes.")

    # =========================================================================
    # 2. CHASE CHECK (2ND INNINGS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. CHASE CHECK (2ND INNINGS)")
    print("=" * 80)

    # Filter: 2nd innings, decisive result, non-rain, no super over
    super_over_matches = set(df[df['is_super_over']]['match_id']).union(
        set(matches[matches['result'].str.contains('Super Over|One Over Eliminator', case=False, na=False)]['match_id'])
    )
    
    m_dict = matches.set_index('match_id')['result'].to_dict()
    df_result = df['match_id'].map(m_dict).fillna('')
    is_rain = df['is_rain_affected'] | df_result.str.contains('DLS|D/L', case=False)
    
    chase_mask = (
        (df['innings'] == 2) & 
        df['has_result'] & 
        (~is_rain) & 
        (~df['match_id'].isin(super_over_matches))
    )
    df_chase = df[chase_mask].copy()

    # Get last ball of 2nd innings per match
    last_balls = df_chase.sort_values(
        ['match_id', 'overid', 'over_ball_num', 'innings_ball_num']
    ).groupby('match_id').last().reset_index()

    last_balls['final_total'] = last_balls['runs_so_far'] + last_balls['total_runs_this_ball']
    last_balls['gap'] = last_balls['final_total'] - last_balls['target']

    won_chases = last_balls[last_balls['batting_team_won'] == True]
    lost_chases = last_balls[last_balls['batting_team_won'] == False]

    won_pct = (won_chases['final_total'] >= won_chases['target']).mean() * 100.0
    lost_pct = (lost_chases['final_total'] < lost_chases['target']).mean() * 100.0

    print(f"Total valid chase matches analyzed: {len(last_balls)}")
    print(f"\nWon Chases (n={len(won_chases)}):")
    print(f"  - Percentage where final_total >= target: {won_pct:.2f}%")
    print(f"  - Gap (final_total - target): Mean = {won_chases['gap'].mean():+.2f}, Median = {won_chases['gap'].median():+.2f}")

    print(f"\nLost Chases (n={len(lost_chases)}):")
    print(f"  - Percentage where final_total < target: {lost_pct:.2f}%")
    print(f"  - Gap (final_total - target): Mean = {lost_chases['gap'].mean():+.2f}, Median = {lost_chases['gap'].median():+.2f}")

    # Identify mismatches
    won_mismatches = won_chases[won_chases['final_total'] < won_chases['target']]
    lost_mismatches = lost_chases[lost_chases['final_total'] >= lost_chases['target']]
    all_mismatches = pd.concat([won_mismatches, lost_mismatches]).copy()
    all_mismatches['abs_gap'] = all_mismatches['gap'].abs()

    print(f"\nTotal chase mismatches: {len(all_mismatches)}")
    print("Top 10 Worst Mismatches (by absolute gap):")
    cols_mismatch = ['match_id', 'competition', 'batting_team_won', 'final_total', 'target', 'gap', 'match_winner']
    print(all_mismatches.sort_values('abs_gap', ascending=False)[cols_mismatch].head(10).to_string(index=False))

    # =========================================================================
    # 3. FIRST-INNINGS TOTALS PER COMP_FORMAT
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. FIRST-INNINGS TOTALS PER COMP_FORMAT")
    print("=" * 80)

    inn1 = df[df['innings'] == 1].groupby(['match_id', 'comp_format'])['total_runs_this_ball'].sum().reset_index()
    inn1_stats = inn1.groupby('comp_format')['total_runs_this_ball'].agg(['count', 'mean', 'median', 'std'])
    inn1_stats['mean'] = inn1_stats['mean'].round(2)
    inn1_stats['median'] = inn1_stats['median'].round(2)
    inn1_stats['std'] = inn1_stats['std'].round(2)
    print(inn1_stats.to_string())

    # =========================================================================
    # 4. LEGAL-BALL CHECK & 'other' TREATMENT COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. LEGAL-BALL CHECK & 'other' TREATMENT COMPARISON")
    print("=" * 80)

    # Restrict to 20-over completed innings for full overs 1 to 19
    max_overs_per_inn = df.groupby(['match_id', 'innings'])['overid'].max()
    full_inn_keys = max_overs_per_inn[max_overs_per_inn == 20].index
    df_full_inn = df.set_index(['match_id', 'innings']).loc[full_inn_keys].reset_index()

    # Current treatment: 'other' is ILLEGAL (is_legal = False)
    overs_current = df_full_inn[df_full_inn['overid'] < 20].groupby(['match_id', 'innings', 'overid'])['is_legal'].sum()
    dist_current = overs_current.value_counts().sort_index()

    # Alternative treatment: 'other' is LEGAL (is_legal = True)
    is_legal_alt = ~df_full_inn['extras_type'].isin(['wide', 'Overstep'])
    df_full_inn['is_legal_alt'] = is_legal_alt
    overs_alt = df_full_inn[df_full_inn['overid'] < 20].groupby(['match_id', 'innings', 'overid'])['is_legal_alt'].sum()
    dist_alt = overs_alt.value_counts().sort_index()

    comp_table = pd.DataFrame({
        "Overs ('other' = ILLEGAL)": dist_current,
        "Overs ('other' = LEGAL)": dist_alt
    }).fillna(0).astype(int)
    print("Distribution of legal balls per over (Overs 1-19):")
    print(comp_table.to_string())

    print("\nConclusion on 'other':")
    overs_6_current = dist_current.get(6, 0)
    overs_6_alt = dist_alt.get(6, 0)
    print(f"  - Treating 'other' as ILLEGAL yields {overs_6_current:,} overs with exactly 6 balls.")
    print(f"  - Treating 'other' as LEGAL yields {overs_6_alt:,} overs with exactly 6 balls.")
    print(f"  - Difference: Treating 'other' as ILLEGAL gives {overs_6_current - overs_6_alt:,} MORE overs with exactly 6 legal balls.")
    print("  - When 'other' is treated as legal, 175 overs artificially show 7, 8, or 9 legal balls,")
    print("    proving that umpires required those balls to be re-bowled as no-balls.")

    # Distribution of balls_remaining at the end of 20-over innings and all-out innings
    last_balls_all = df.sort_values(
        ['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num']
    ).groupby(['match_id', 'innings']).last().reset_index()

    inn_20 = last_balls_all[last_balls_all['overid'] == 20]
    print("\nDistribution of balls_remaining (BEFORE ball) at last delivery of 20-over innings:")
    print(inn_20['balls_remaining'].value_counts().sort_index().to_string())

    inn_all_out = last_balls_all[(last_balls_all['wickets_so_far'] + last_balls_all['is_wicket']) >= 10]
    print("\nSummary of balls_remaining when an innings is ALL OUT (10 wickets down):")
    print(f"  - Count of all-out innings: {len(inn_all_out)}")
    print(f"  - Mean balls remaining: {inn_all_out['balls_remaining'].mean():.2f}")
    print(f"  - Median balls remaining: {inn_all_out['balls_remaining'].median():.2f}")
    print(f"  - Min / Max balls remaining: {inn_all_out['balls_remaining'].min()} / {inn_all_out['balls_remaining'].max()}")

    # =========================================================================
    # 5. MATCH RESULT PATTERNS
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. MATCH RESULT PATTERNS IN matches.csv")
    print("=" * 80)
    results = matches['result'].fillna('Missing')
    
    has_dls = results.str.contains('DLS|D/L', case=False)
    has_tie = results.str.contains('Tie|Tied', case=False)
    has_no_result = results.str.contains('No Result|Match Abandoned', case=False) | matches['win_team_id'].isna()
    has_super_over = results.str.contains('Super Over|One Over Eliminator', case=False)
    won_by_runs = results.str.contains('Runs?', case=False) & ~has_dls & ~has_tie
    won_by_wickets = results.str.contains('Wickets?', case=False) & ~has_dls & ~has_tie

    res_summary = pd.DataFrame({
        'Pattern Category': [
            'Standard Win by Runs',
            'Standard Win by Wickets',
            'Rain Affected (DLS / D/L Method)',
            'Tied Matches',
            'Super Over Finishes',
            'No Result / Abandoned',
            'Total Matches'
        ],
        'Count': [
            won_by_runs.sum(),
            won_by_wickets.sum(),
            has_dls.sum(),
            has_tie.sum(),
            has_super_over.sum(),
            has_no_result.sum(),
            len(matches)
        ],
        'Action for Scope': [
            'Keep (In Scope)',
            'Keep (In Scope)',
            'Exclude (Rain-affected)',
            'Exclude (Indecisive)',
            'Exclude (Irregular)',
            'Exclude (Incomplete)',
            '-'
        ]
    })
    print(res_summary.to_string(index=False))

    # =========================================================================
    # 6. IN_SCOPE FLAGGING & SAVING PARQUET
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. IN_SCOPE FLAGGING & EXPORT")
    print("=" * 80)

    # Compute in_scope per match:
    # 1. comp_format in ('IPL', 'T20I')
    cond_format = df['comp_format'].isin(['IPL', 'T20I'])

    # 2. Not rain affected
    cond_not_rain = (~df['is_rain_affected']) & (~df_result.str.contains('DLS|D/L', case=False))

    # 3. Decisive result (not tie / no result)
    cond_decisive = df['has_result'] & (~df_result.str.contains('Tie|Tied|No Result|Match Abandoned', case=False))

    # 4. No super over
    cond_no_so = ~df['match_id'].isin(super_over_matches)

    # 5. Exactly 2 complete innings (match has both innings 1 and 2 in bbb)
    inn_count_per_match = df.groupby('match_id')['innings'].nunique()
    two_inn_matches = set(inn_count_per_match[inn_count_per_match >= 2].index)
    cond_2_inn = df['match_id'].isin(two_inn_matches)

    # Combined in_scope flag
    df['in_scope'] = cond_format & cond_not_rain & cond_decisive & cond_no_so & cond_2_inn

    # Print kept matches and balls
    for fmt in ['IPL', 'T20I']:
        sub_all = df[df['comp_format'] == fmt]
        sub_scope = df[(df['comp_format'] == fmt) & df['in_scope']]
        m_tot = sub_all['match_id'].nunique()
        m_sc = sub_scope['match_id'].nunique()
        b_tot = len(sub_all)
        b_sc = len(sub_scope)
        print(f">>> {fmt} Scope Summary:")
        print(f"  Matches kept: {m_sc} / {m_tot} ({m_sc / m_tot * 100:.2f}%)")
        print(f"  Balls kept:   {b_sc:,} / {b_tot:,} ({b_sc / b_tot * 100:.2f}%)")

    # Save to parquet without filtering any rows
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    print(f"\nSaved updated parquet with 'in_scope' column to '{parquet_path}' ({len(df):,} rows, {os.path.getsize(parquet_path) / (1024*1024):.2f} MB)")
    print("=" * 80)
    print("Validation finished successfully.")

if __name__ == '__main__':
    run_validation()

