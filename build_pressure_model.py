import os
import pandas as pd
import numpy as np

def build_pressure_model():
    print("=" * 65)
    print("SupeRR Selector Season 2: Unified In-Scope Pressure Model")
    print("=" * 65)
    
    parquet_path = 'merged_balls.parquet'
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found. Please run prepare_data.py first.")
        return
        
    print("\n1. Loading preprocessed data from merged_balls.parquet...")
    df = pd.read_parquet(parquet_path)
    
    # Filter strictly on in-scope deliveries
    df = df[df['in_scope']].copy()
    print(f"Loaded {len(df):,} clean in-scope deliveries across {df['match_id'].nunique()} matches (IPL & T20I).")

    print("\n2. Applying High-Pressure Situational Filters...")
    # Innings 2 pressure logic (using BEFORE-the-ball state features)
    runs_needed = np.maximum(0, df['target'] - df['runs_so_far'])
    wickets_in_hand = np.maximum(0, 10 - df['wickets_so_far'])
    
    cond_chase_tight = (df['innings'] == 2) & (df['overid'] >= 15) & (runs_needed > 0) & (runs_needed <= 55) & (wickets_in_hand >= 1)
    cond_chase_rrr = (df['innings'] == 2) & (runs_needed > 0) & (df['required_run_rate'] >= 9.5) & (wickets_in_hand <= 7)
    cond_chase_collapse = (df['innings'] == 2) & (((df['overid'] <= 6) & (df['wickets_so_far'] >= 3)) | ((df['overid'] <= 11) & (df['wickets_so_far'] >= 5)))

    # Innings 1 pressure logic
    cond_inn1_collapse = (df['innings'] == 1) & (((df['overid'] <= 6) & (df['wickets_so_far'] >= 3)) | ((df['overid'] <= 10) & (df['wickets_so_far'] >= 4)))
    cond_inn1_death = (df['innings'] == 1) & (df['overid'] >= 16)

    df['is_high_pressure'] = cond_chase_tight | cond_chase_rrr | cond_chase_collapse | cond_inn1_collapse | cond_inn1_death
    
    hp_count = df['is_high_pressure'].sum()
    print(f"High-Pressure Deliveries: {hp_count:,} ({hp_count / len(df) * 100:.2f}%)")

    # 3. Batting Performance Metrics
    print("\n3. Calculating Batter Pressure vs Baseline Metrics...")
    df_hp = df[df['is_high_pressure']]
    
    # Aggregate HP stats
    b_hp = df_hp.groupby(['striker_id', 'striker_name']).agg(
        hp_balls=('is_legal', 'sum'),
        hp_runs=('batter_runs', 'sum'),
        hp_fours=('is_four', 'sum'),
        hp_sixes=('is_six', 'sum'),
        hp_dots=('batter_runs', lambda x: (x == 0).sum())
    ).reset_index()
    
    # HP dismissals (when striker is the dismissed player)
    hp_outs = df_hp[df_hp['is_wicket'] & (df_hp['dismissed_player_id'] == df_hp['striker_id'])].groupby('striker_id').size().rename('hp_outs')
    b_hp = b_hp.merge(hp_outs, on='striker_id', how='left')
    b_hp['hp_outs'] = b_hp['hp_outs'].fillna(0).astype(int)
    
    # Overall baseline batting stats
    b_ov = df.groupby('striker_id').agg(
        ov_balls=('is_legal', 'sum'),
        ov_runs=('batter_runs', 'sum'),
        ov_fours=('is_four', 'sum'),
        ov_sixes=('is_six', 'sum')
    ).reset_index()
    ov_outs = df[df['is_wicket'] & (df['dismissed_player_id'] == df['striker_id'])].groupby('striker_id').size().rename('ov_outs')
    b_ov = b_ov.merge(ov_outs, on='striker_id', how='left')
    b_ov['ov_outs'] = b_ov['ov_outs'].fillna(0).astype(int)
    
    batters = b_hp.merge(b_ov, on='striker_id', how='inner')
    
    # Rates
    batters['hp_strike_rate'] = (batters['hp_runs'] / np.maximum(1, batters['hp_balls'])) * 100.0
    batters['hp_average'] = batters['hp_runs'] / np.maximum(1, batters['hp_outs'])
    batters['hp_boundary_pct'] = ((batters['hp_fours'] + batters['hp_sixes']) / np.maximum(1, batters['hp_balls'])) * 100.0
    batters['hp_dot_pct'] = (batters['hp_dots'] / np.maximum(1, batters['hp_balls'])) * 100.0
    
    batters['ov_strike_rate'] = (batters['ov_runs'] / np.maximum(1, batters['ov_balls'])) * 100.0
    batters['ov_average'] = batters['ov_runs'] / np.maximum(1, batters['ov_outs'])
    
    # Elevation Delta
    batters['sr_delta'] = batters['hp_strike_rate'] - batters['ov_strike_rate']
    batters['avg_delta'] = batters['hp_average'] - batters['ov_average']
    
    # Minimum sample threshold: 150 High-Pressure balls
    eligible_batters = batters[batters['hp_balls'] >= 150].copy()
    
    # Composite Pressure Batting Score
    sr_score = np.clip((eligible_batters['hp_strike_rate'] - 110) / (200 - 110), 0, 1) * 100
    avg_score = np.clip((eligible_batters['hp_average'] - 15) / (50 - 15), 0, 1) * 100
    sr_delta_score = np.clip((eligible_batters['sr_delta'] + 10) / (35 + 10), 0, 1) * 100
    bnd_score = np.clip((eligible_batters['hp_boundary_pct'] - 12) / (28 - 12), 0, 1) * 100
    vol_weight = np.clip(np.log(eligible_batters['hp_balls']) / np.log(600), 0.75, 1.05)
    
    eligible_batters['pressure_batting_score'] = (
        0.35 * sr_score + 
        0.30 * avg_score + 
        0.20 * sr_delta_score + 
        0.15 * bnd_score
    ) * vol_weight
    
    eligible_batters['pressure_batting_score'] = eligible_batters['pressure_batting_score'].round(2)
    eligible_batters = eligible_batters.sort_values(by='pressure_batting_score', ascending=False).reset_index(drop=True)
    eligible_batters['pressure_rank'] = eligible_batters.index + 1

    # 4. Bowling Performance Metrics
    print("\n4. Calculating Bowler Pressure vs Baseline Metrics...")
    # HP bowling
    bowl_hp = df_hp.groupby(['bowler_id', 'bowler_name']).agg(
        hp_balls=('is_legal', 'sum'),
        hp_runs_conceded=('bowler_runs_conceded', 'sum'),
        hp_wickets=('is_bowler_wicket', 'sum'),
        hp_fours_conceded=('is_four', 'sum'),
        hp_sixes_conceded=('is_six', 'sum'),
        hp_dots=('bowler_runs_conceded', lambda x: (x == 0).sum())
    ).reset_index()
    
    # Baseline bowling
    bowl_ov = df.groupby('bowler_id').agg(
        ov_balls=('is_legal', 'sum'),
        ov_runs_conceded=('bowler_runs_conceded', 'sum'),
        ov_wickets=('is_bowler_wicket', 'sum')
    ).reset_index()
    
    bowlers = bowl_hp.merge(bowl_ov, on='bowler_id', how='inner')
    
    # Rates
    bowlers['hp_economy'] = (bowlers['hp_runs_conceded'] / np.maximum(1, bowlers['hp_balls'])) * 6.0
    bowlers['hp_bowling_strike_rate'] = np.where(bowlers['hp_wickets'] > 0, bowlers['hp_balls'] / bowlers['hp_wickets'], 999.0)
    bowlers['hp_bowling_average'] = np.where(bowlers['hp_wickets'] > 0, bowlers['hp_runs_conceded'] / bowlers['hp_wickets'], 999.0)
    bowlers['hp_dot_pct'] = (bowlers['hp_dots'] / np.maximum(1, bowlers['hp_balls'])) * 100.0
    bowlers['hp_boundary_pct'] = ((bowlers['hp_fours_conceded'] + bowlers['hp_sixes_conceded']) / np.maximum(1, bowlers['hp_balls'])) * 100.0
    
    bowlers['ov_economy'] = (bowlers['ov_runs_conceded'] / np.maximum(1, bowlers['ov_balls'])) * 6.0
    bowlers['ov_bowling_average'] = np.where(bowlers['ov_wickets'] > 0, bowlers['ov_runs_conceded'] / bowlers['ov_wickets'], 999.0)
    bowlers['economy_delta'] = bowlers['ov_economy'] - bowlers['hp_economy']
    
    # Eligibility threshold: 150 HP balls
    eligible_bowlers = bowlers[bowlers['hp_balls'] >= 150].copy()
    
    # Composite Pressure Bowling Score
    econ_score = np.clip((11.0 - eligible_bowlers['hp_economy']) / (11.0 - 7.0), 0, 1) * 100
    sr_bowl_score = np.clip((28.0 - eligible_bowlers['hp_bowling_strike_rate']) / (28.0 - 11.0), 0, 1) * 100
    dot_score = np.clip((eligible_bowlers['hp_dot_pct'] - 28.0) / (48.0 - 28.0), 0, 1) * 100
    bnd_conceded_score = np.clip((24.0 - eligible_bowlers['hp_boundary_pct']) / (24.0 - 11.0), 0, 1) * 100
    vol_bowl_weight = np.clip(np.log(eligible_bowlers['hp_balls']) / np.log(600), 0.75, 1.05)
    
    eligible_bowlers['pressure_bowling_score'] = (
        0.35 * econ_score + 
        0.30 * sr_bowl_score + 
        0.20 * dot_score + 
        0.15 * bnd_conceded_score
    ) * vol_bowl_weight
    
    eligible_bowlers['pressure_bowling_score'] = eligible_bowlers['pressure_bowling_score'].round(2)
    eligible_bowlers = eligible_bowlers.sort_values(by='pressure_bowling_score', ascending=False).reset_index(drop=True)
    eligible_bowlers['pressure_rank'] = eligible_bowlers.index + 1

    # 5. Export Datasets
    print("\n5. Exporting updated alternate datasets...")
    for col in ['hp_average', 'hp_strike_rate', 'hp_boundary_pct', 'hp_dot_pct', 
                'ov_average', 'ov_strike_rate', 'sr_delta', 'avg_delta']:
        eligible_batters[col] = eligible_batters[col].round(2)
        
    for col in ['hp_economy', 'hp_bowling_strike_rate', 'hp_bowling_average', 
                'hp_dot_pct', 'hp_boundary_pct', 'ov_economy', 'ov_bowling_average', 'economy_delta']:
        eligible_bowlers[col] = eligible_bowlers[col].round(2)

    bat_cols = [
        'pressure_rank', 'striker_id', 'striker_name', 'pressure_batting_score',
        'hp_balls', 'hp_runs', 'hp_outs', 'hp_average', 'hp_strike_rate',
        'hp_boundary_pct', 'hp_dot_pct', 'ov_balls', 'ov_runs', 'ov_average', 
        'ov_strike_rate', 'sr_delta', 'avg_delta'
    ]
    top_batters_30 = eligible_batters[bat_cols].head(30)
    top_batters_30.to_csv('top_batters_under_pressure.csv', index=False)

    bowl_cols = [
        'pressure_rank', 'bowler_id', 'bowler_name', 'pressure_bowling_score',
        'hp_balls', 'hp_wickets', 'hp_runs_conceded', 'hp_economy', 
        'hp_bowling_strike_rate', 'hp_bowling_average', 'hp_dot_pct', 
        'hp_boundary_pct', 'ov_balls', 'ov_wickets', 'ov_economy', 
        'ov_bowling_average', 'economy_delta'
    ]
    top_bowlers_30 = eligible_bowlers[bowl_cols].head(30)
    top_bowlers_30.to_csv('top_bowlers_under_pressure.csv', index=False)

    # Master summary
    tb = top_batters_30[['pressure_rank', 'striker_id', 'striker_name', 'pressure_batting_score']].copy()
    tb['discipline'] = 'Batter'
    tb.rename(columns={'striker_id': 'player_id', 'striker_name': 'player_name', 'pressure_batting_score': 'pressure_rating'}, inplace=True)

    tbo = top_bowlers_30[['pressure_rank', 'bowler_id', 'bowler_name', 'pressure_bowling_score']].copy()
    tbo['discipline'] = 'Bowler'
    tbo.rename(columns={'bowler_id': 'player_id', 'bowler_name': 'player_name', 'pressure_bowling_score': 'pressure_rating'}, inplace=True)

    comb = pd.concat([tb, tbo], ignore_index=True)
    comb.to_csv('pressure_player_rankings.csv', index=False)

    print(f"Saved: top_batters_under_pressure.csv ({len(top_batters_30)} players)")
    print(f"Saved: top_bowlers_under_pressure.csv ({len(top_bowlers_30)} players)")
    print(f"Saved: pressure_player_rankings.csv")

    print("\n--- FINAL TOP 10 BATTERS UNDER PRESSURE ---")
    print(top_batters_30[['pressure_rank', 'striker_name', 'pressure_batting_score', 'hp_balls', 'hp_average', 'hp_strike_rate', 'sr_delta']].head(10).to_string(index=False))

    print("\n--- FINAL TOP 10 BOWLERS UNDER PRESSURE ---")
    print(top_bowlers_30[['pressure_rank', 'bowler_name', 'pressure_bowling_score', 'hp_balls', 'hp_wickets', 'hp_economy', 'hp_bowling_strike_rate']].head(10).to_string(index=False))

if __name__ == '__main__':
    build_pressure_model()
