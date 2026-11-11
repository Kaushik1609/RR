import pandas as pd
import numpy as np

def run_analysis():
    print("Loading data...")
    bbb = pd.read_csv('ball_by_ball.csv')
    matches = pd.read_csv('matches.csv')
    comps = pd.read_csv('competitions.csv')

    # Merge match context
    m_sub = matches[['match_id', 'stage', 'knock_out', 'revised_target', 'first_inn_revised_overs', 'second_inn_revised_overs', 'comp_id']]
    df = bbb.merge(m_sub, on='match_id', how='left')
    df = df.merge(comps[['comp_id', 'match_format', 'season']], on='comp_id', how='left')

    # Classify legal ball and runs
    df['is_legal'] = ~df['extras_type'].isin(['wide', 'Overstep'])
    df['batter_runs'] = df['runs']
    df['extra_runs'] = np.where(df['extras_type'].isin(['wide', 'Overstep', 'byes', 'legbyes', 'other']), 1, 0) + df['extras_2_runs']
    df['bowler_runs'] = df['runs'] + np.where(df['extras_type'].isin(['wide', 'Overstep']), 1, 0) + df['extras_2_runs']
    df['team_ball_runs'] = df['batter_runs'] + df['extra_runs']

    df.sort_values(by=['match_id', 'innings', 'overid', 'over_ball_num'], inplace=True)

    # Cumulative states before ball
    df['cum_runs_after'] = df.groupby(['match_id', 'innings'])['team_ball_runs'].cumsum()
    df['cum_wickets_after'] = df.groupby(['match_id', 'innings'])['is_wicket'].cumsum()
    df['cum_legal_after'] = df.groupby(['match_id', 'innings'])['is_legal'].cumsum()

    df['cum_runs_before'] = df.groupby(['match_id', 'innings'])['cum_runs_after'].shift(1).fillna(0)
    df['cum_wickets_before'] = df.groupby(['match_id', 'innings'])['cum_wickets_after'].shift(1).fillna(0)
    df['cum_legal_before'] = df.groupby(['match_id', 'innings'])['cum_legal_after'].shift(1).fillna(0)

    # Innings 1 total
    inn1_totals = df[df['innings'] == 1].groupby('match_id')['team_ball_runs'].sum().rename('inn1_total')
    df = df.merge(inn1_totals, on='match_id', how='left')

    # Max legal balls in innings
    df['total_legal_balls'] = np.where(df['innings'] == 1, 
                                       df['first_inn_revised_overs'].fillna(120),
                                       df['second_inn_revised_overs'].fillna(120))

    df['target'] = np.where(df['revised_target'].notna(), df['revised_target'], df['inn1_total'] + 1)
    df['runs_needed'] = df['target'] - df['cum_runs_before']
    df['balls_rem'] = np.maximum(0, df['total_legal_balls'] - df['cum_legal_before'])
    df['wickets_down'] = df['cum_wickets_before']
    df['wickets_in_hand'] = 10 - df['wickets_down']

    df['rrr'] = np.where(df['balls_rem'] > 0, (df['runs_needed'] / df['balls_rem']) * 6, 0.0)

    # Innings 2 pressure logic
    cond_chase_tight = (df['innings'] == 2) & (df['overid'] >= 15) & (df['runs_needed'] > 0) & (df['runs_needed'] <= 55) & (df['wickets_in_hand'] >= 1)
    cond_chase_rrr = (df['innings'] == 2) & (df['runs_needed'] > 0) & (df['rrr'] >= 9.5) & (df['wickets_in_hand'] <= 7)
    cond_chase_collapse = (df['innings'] == 2) & (df['runs_needed'] > 0) & (((df['overid'] <= 6) & (df['wickets_down'] >= 3)) | ((df['overid'] <= 11) & (df['wickets_down'] >= 5)))

    # Innings 1 pressure logic
    cond_inn1_collapse = (df['innings'] == 1) & (((df['overid'] <= 6) & (df['wickets_down'] >= 3)) | ((df['overid'] <= 10) & (df['wickets_down'] >= 4)))
    cond_inn1_death = (df['innings'] == 1) & (df['overid'] >= 16)

    # Knockout flag
    df['is_knockout'] = (df['knock_out'] == 1) | df['stage'].isin(['Final', 'Qualifier', 'Eliminator', 'Semi Final', 'Super 4'])

    df['is_high_pressure'] = cond_chase_tight | cond_chase_rrr | cond_chase_collapse | cond_inn1_collapse | cond_inn1_death

    total_balls = len(df)
    hp_balls = df['is_high_pressure'].sum()
    pct = (hp_balls / total_balls) * 100
    print(f"High pressure balls: {hp_balls} / {total_balls} ({pct:.2f}%)")

    # Let's inspect batter stats under pressure vs overall
    # Batters:
    # Pressure:
    b_hp = df[df['is_high_pressure']]
    b_all = df

    # Aggregations for Batters:
    print("\nComputing batter metrics...")
    batter_hp = b_hp.groupby(['striker_id', 'striker_name']).agg(
        hp_balls=('is_legal', 'sum'),
        hp_runs=('batter_runs', 'sum'),
        hp_dismissals=('dismissed_player_id', lambda x: (x == x.name[0]).sum() if hasattr(x, 'name') else 0),
        hp_fours=('is_four', 'sum'),
        hp_sixes=('is_six', 'sum'),
        hp_dots=('batter_runs', lambda x: (x == 0).sum())
    ).reset_index()

    # Actual dismissals for striker:
    # A batter is dismissed when is_wicket is True and dismissed_player_id == striker_id
    b_hp_w = b_hp[b_hp['is_wicket'] & (b_hp['dismissed_player_id'] == b_hp['striker_id'])].groupby('striker_id').size().rename('hp_outs')
    batter_hp = batter_hp.merge(b_hp_w, on='striker_id', how='left')
    batter_hp['hp_outs'] = batter_hp['hp_outs'].fillna(0)

    # Overall batter stats:
    batter_ov = b_all.groupby('striker_id').agg(
        ov_balls=('is_legal', 'sum'),
        ov_runs=('batter_runs', 'sum'),
        ov_fours=('is_four', 'sum'),
        ov_sixes=('is_six', 'sum')
    ).reset_index()
    b_ov_w = b_all[b_all['is_wicket'] & (b_all['dismissed_player_id'] == b_all['striker_id'])].groupby('striker_id').size().rename('ov_outs')
    batter_ov = batter_ov.merge(b_ov_w, on='striker_id', how='left')
    batter_ov['ov_outs'] = batter_ov['ov_outs'].fillna(0)

    bat_merged = batter_hp.merge(batter_ov, on='striker_id', how='inner')
    
    # Calculate rates
    bat_merged['hp_strike_rate'] = (bat_merged['hp_runs'] / np.maximum(1, bat_merged['hp_balls'])) * 100
    bat_merged['hp_average'] = bat_merged['hp_runs'] / np.maximum(1, bat_merged['hp_outs'])
    bat_merged['hp_boundary_pct'] = ((bat_merged['hp_fours'] + bat_merged['hp_sixes']) / np.maximum(1, bat_merged['hp_balls'])) * 100
    bat_merged['hp_dot_pct'] = (bat_merged['hp_dots'] / np.maximum(1, bat_merged['hp_balls'])) * 100

    bat_merged['ov_strike_rate'] = (bat_merged['ov_runs'] / np.maximum(1, bat_merged['ov_balls'])) * 100
    bat_merged['ov_average'] = bat_merged['ov_runs'] / np.maximum(1, bat_merged['ov_outs'])

    # Filter by minimum sample size under pressure (e.g. at least 150 pressure balls)
    eligible_batters = bat_merged[bat_merged['hp_balls'] >= 150].copy()
    print(f"Eligible batters with >= 150 HP balls: {len(eligible_batters)}")
    
    # Bowlers:
    print("\nComputing bowler metrics...")
    bowler_hp = b_hp.groupby(['bowler_id', 'bowler_name']).agg(
        hp_balls=('is_legal', 'sum'),
        hp_runs_conceded=('bowler_runs', 'sum'),
        hp_dots=('bowler_runs', lambda x: (x == 0).sum()),
        hp_fours_conceded=('is_four', 'sum'),
        hp_sixes_conceded=('is_six', 'sum')
    ).reset_index()
    
    # Bowler wickets (dismissals credited to bowler, excluding run-outs)
    b_hp_bw = b_hp[b_hp['is_wicket'] & (~b_hp['dismissal_type'].isin(['RunOut', 'Retire Out']))].groupby('bowler_id').size().rename('hp_wickets')
    bowler_hp = bowler_hp.merge(b_hp_bw, on='bowler_id', how='left')
    bowler_hp['hp_wickets'] = bowler_hp['hp_wickets'].fillna(0)

    # Overall bowler stats:
    bowler_ov = b_all.groupby('bowler_id').agg(
        ov_balls=('is_legal', 'sum'),
        ov_runs_conceded=('bowler_runs', 'sum')
    ).reset_index()
    b_ov_bw = b_all[b_all['is_wicket'] & (~b_all['dismissal_type'].isin(['RunOut', 'Retire Out']))].groupby('bowler_id').size().rename('ov_wickets')
    bowler_ov = bowler_ov.merge(b_ov_bw, on='bowler_id', how='left')
    bowler_ov['ov_wickets'] = bowler_ov['ov_wickets'].fillna(0)

    bowl_merged = bowler_hp.merge(bowler_ov, on='bowler_id', how='inner')
    
    bowl_merged['hp_economy'] = (bowl_merged['hp_runs_conceded'] / np.maximum(1, bowl_merged['hp_balls'])) * 6
    bowl_merged['hp_strike_rate'] = np.where(bowl_merged['hp_wickets'] > 0, bowl_merged['hp_balls'] / bowl_merged['hp_wickets'], np.nan)
    bowl_merged['hp_average'] = np.where(bowl_merged['hp_wickets'] > 0, bowl_merged['hp_runs_conceded'] / bowl_merged['hp_wickets'], np.nan)
    bowl_merged['hp_dot_pct'] = (bowl_merged['hp_dots'] / np.maximum(1, bowl_merged['hp_balls'])) * 100
    bowl_merged['hp_boundary_pct'] = ((bowl_merged['hp_fours_conceded'] + bowl_merged['hp_sixes_conceded']) / np.maximum(1, bowl_merged['hp_balls'])) * 100

    bowl_merged['ov_economy'] = (bowl_merged['ov_runs_conceded'] / np.maximum(1, bowl_merged['ov_balls'])) * 6
    bowl_merged['ov_average'] = np.where(bowl_merged['ov_wickets'] > 0, bowl_merged['ov_runs_conceded'] / bowl_merged['ov_wickets'], np.nan)

    eligible_bowlers = bowl_merged[bowl_merged['hp_balls'] >= 150].copy()
    print(f"Eligible bowlers with >= 150 HP balls: {len(eligible_bowlers)}")

    print("\nTop 10 Batters by raw HP Strike Rate (min 150 balls):")
    print(eligible_batters.sort_values(by='hp_strike_rate', ascending=False)[['striker_name', 'hp_balls', 'hp_runs', 'hp_average', 'hp_strike_rate', 'ov_strike_rate']].head(10))

    print("\nTop 10 Bowlers by lowest HP Economy (min 150 balls):")
    print(eligible_bowlers.sort_values(by='hp_economy', ascending=True)[['bowler_name', 'hp_balls', 'hp_wickets', 'hp_economy', 'ov_economy', 'hp_dot_pct']].head(10))

if __name__ == '__main__':
    run_analysis()

