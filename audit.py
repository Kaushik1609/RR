"""
audit.py
--------
Comprehensive audit script for cricket analytics datasets:
1. Inspects shape, columns, dtypes, null counts, and 3 sample rows for each CSV.
2. Identifies join keys between ball_by_ball and matches.
3. Checks date range, IPL vs T20I match breakdown, and encoding of innings/over/ball/wicket/extras.
4. Builds and exports a merged ball-level dataframe to merged_balls.parquet with
   required in-game dynamic situational features.
"""

import os
import pandas as pd
import numpy as np

def audit_datasets():
    csv_files = [
        'ball_by_ball.csv',
        'matches.csv',
        'playing_eleven.csv',
        'teams.csv',
        'competitions.csv',
        'competition_team_player_mapping.csv'
    ]
    
    data = {}
    print("=" * 70)
    print("1. FILE AUDITS: SHAPE, COLUMNS, DTYPES, NULL COUNTS & SAMPLES")
    print("=" * 70)
    
    for f in csv_files:
        if not os.path.exists(f):
            print(f"Warning: {f} not found!")
            continue
        df = pd.read_csv(f)
        data[f] = df
        print(f"\n>>> File: {f}")
        print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Dtype': [str(t) for t in df.dtypes],
            'Nulls': df.isnull().sum().values
        })
        print("\nColumns, Dtypes & Null Counts:")
        print(info_df.to_string(index=False))
        
        print("\nSample (3 rows):")
        print(df.head(3).to_string(index=False))
        print("-" * 70)

    print("\n" + "=" * 70)
    print("2. JOIN KEYS BETWEEN ball_by_ball AND matches")
    print("=" * 70)
    bbb = data['ball_by_ball.csv']
    matches = data['matches.csv']
    
    common_cols = list(set(bbb.columns).intersection(set(matches.columns)))
    print(f"Common columns: {common_cols}")
    print(f"Primary Join Key: 'match_id'")
    
    bbb_matches = bbb['match_id'].nunique()
    m_matches = matches['match_id'].nunique()
    overlap = len(set(bbb['match_id']).intersection(set(matches['match_id'])))
    print(f"Unique match_ids in matches.csv: {m_matches}")
    print(f"Unique match_ids in ball_by_ball.csv: {bbb_matches}")
    print(f"Matching overlap: {overlap} / {bbb_matches} ({overlap / bbb_matches * 100:.1f}%)")

    print("\n" + "=" * 70)
    print("3. MATCH CONTEXT, FORMAT BREAKDOWN & BALL ENCODING")
    print("=" * 70)
    comps = data['competitions.csv']
    m_merged = matches.merge(comps, on='comp_id', how='left')
    
    min_date = m_merged['match_date'].min()
    max_date = m_merged['match_date'].max()
    print(f"Date Range: {min_date} to {max_date}")
    
    ipl_count = m_merged['comp_name'].str.contains('IPL', case=False, na=False).sum()
    t20i_count = (m_merged['match_format'] == 'T20I').sum()
    other_t20 = len(m_merged) - ipl_count - t20i_count
    print(f"Total Matches: {len(m_merged)}")
    print(f"  - IPL Matches: {ipl_count}")
    print(f"  - T20I Matches: {t20i_count}")
    print(f"  - Other T20 Matches: {other_t20}")
    
    print("\nEncoding of Key Ball-by-Ball Fields:")
    print(f"  - innings: {sorted(bbb['innings'].unique().tolist())} (Innings 1, 2, + Super Overs if any)")
    print(f"  - overid: min={bbb['overid'].min()}, max={bbb['overid'].max()} (1 to 20 overs)")
    print(f"  - over_ball_num: min={bbb['over_ball_num'].min()}, max={bbb['over_ball_num'].max()} (ball within over)")
    print(f"  - is_wicket: {bbb['is_wicket'].dtype} (True={bbb['is_wicket'].sum():,}, False={(~bbb['is_wicket']).sum():,})")
    print(f"  - extras_type: {bbb['extras_type'].value_counts(dropna=False).to_dict()}")
    print("  - Note on extras: 'extras1_runs' encodes extras category code (1: Overstep/No-ball, 2: Wide, 3: Legbyes, 4: Byes, 5: Other).")
    print("    'extras_2_runs' holds boundary or additional penalty runs.")

    print("\n" + "=" * 70)
    print("4. BUILDING AND SAVING merged_balls.parquet")
    print("=" * 70)
    
    # Enrich match context
    match_winner_series = np.where(
        matches['win_team_id'] == matches['home_team_id'],
        matches['home_team'],
        np.where(matches['win_team_id'] == matches['away_team_id'], matches['away_team'], matches['result'])
    )
    
    m_info = pd.DataFrame({
        'match_id': matches['match_id'],
        'comp_id': matches['comp_id'],
        'venue': matches['venue_country'],
        'match_winner': match_winner_series,
        'revised_target': matches['revised_target'],
        'first_inn_revised_overs': matches['first_inn_revised_overs'],
        'second_inn_revised_overs': matches['second_inn_revised_overs']
    }).merge(comps[['comp_id', 'comp_name']].rename(columns={'comp_name': 'competition'}), on='comp_id', how='left')
    
    merged = bbb.merge(m_info, on='match_id', how='left')
    
    # Delivery level calculation
    merged['is_legal'] = ~merged['extras_type'].isin(['wide', 'Overstep'])
    merged['batter_runs'] = merged['runs'].fillna(0).astype(int)
    merged['extra_runs'] = np.where(merged['extras_type'].isin(['wide', 'Overstep', 'byes', 'legbyes', 'other']), 1, 0) + merged['extras_2_runs'].fillna(0).astype(int)
    merged['ball_team_runs'] = merged['batter_runs'] + merged['extra_runs']
    
    # Chronological sort
    merged.sort_values(by=['match_id', 'innings', 'overid', 'over_ball_num'], inplace=True)
    
    # Cumulative states
    merged['runs_scored_so_far'] = merged.groupby(['match_id', 'innings'])['ball_team_runs'].cumsum()
    merged['wickets_fallen'] = merged.groupby(['match_id', 'innings'])['is_wicket'].cumsum()
    merged['legal_balls_bowled'] = merged.groupby(['match_id', 'innings'])['is_legal'].cumsum()
    
    # First innings totals
    inn1_totals = merged[merged['innings'] == 1].groupby('match_id')['ball_team_runs'].sum().rename('inn1_final')
    merged = merged.merge(inn1_totals, on='match_id', how='left')
    
    # Target (for 2nd innings)
    merged['target'] = np.where(
        merged['innings'] == 2,
        np.where(merged['revised_target'].notna(), merged['revised_target'], merged['inn1_final'] + 1),
        np.nan
    )
    
    # Total balls in innings (considering revised overs)
    total_balls_inn = np.where(
        merged['innings'] == 1,
        merged['first_inn_revised_overs'].fillna(120),
        merged['second_inn_revised_overs'].fillna(120)
    )
    merged['balls_remaining'] = np.maximum(0, total_balls_inn - merged['legal_balls_bowled']).astype(int)
    
    # Current Run Rate (CRR)
    merged['current_run_rate'] = np.where(
        merged['legal_balls_bowled'] > 0,
        np.round((merged['runs_scored_so_far'] / merged['legal_balls_bowled']) * 6.0, 2),
        0.0
    )
    
    # Required Run Rate (RRR) for 2nd innings
    runs_needed = np.maximum(0, merged['target'] - merged['runs_scored_so_far'])
    merged['required_run_rate'] = np.where(
        (merged['innings'] == 2) & (merged['balls_remaining'] > 0),
        np.round((runs_needed / merged['balls_remaining']) * 6.0, 2),
        np.where(merged['innings'] == 2, 0.0, np.nan)
    )
    
    # Drop intermediate scratch columns
    cols_to_drop = ['inn1_final', 'first_inn_revised_overs', 'second_inn_revised_overs', 'revised_target', 'legal_balls_bowled']
    merged.drop(columns=[c for c in cols_to_drop if c in merged.columns], inplace=True)
    
    out_parquet = 'merged_balls.parquet'
    merged.to_parquet(out_parquet, index=False, engine='pyarrow')
    print(f"\nSuccessfully generated '{out_parquet}':")
    print(f"Shape: {merged.shape[0]:,} rows x {merged.shape[1]} columns")
    print(f"File size: {os.path.getsize(out_parquet) / (1024*1024):.2f} MB")
    
    print("\nVerified columns present in merged_balls.parquet:")
    target_cols = [
        'innings', 'target', 'runs_scored_so_far', 'wickets_fallen',
        'balls_remaining', 'current_run_rate', 'required_run_rate',
        'competition', 'venue', 'match_winner'
    ]
    for c in target_cols:
        print(f"  [✓] {c} ({merged[c].dtype})")

    print("\nSample preview:")
    print(merged[['match_id', 'innings', 'overid', 'runs_scored_so_far', 'wickets_fallen', 'balls_remaining', 'current_run_rate', 'target', 'required_run_rate', 'competition', 'venue', 'match_winner']].head(3).to_string(index=False))
    print("\nAudit completed successfully.")

if __name__ == '__main__':
    audit_datasets()

