"""
final_checks.py
Comprehensive validation and final verification checks for SupeRR Selector Season 2.
Inputs: scored_balls_v4.parquet, ratings_*_v4.csv
Outputs: ratings_batters_final.csv, ratings_bowlers_final.csv
"""

import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr

def main():
    print("=" * 80)
    print("FINAL CHECKS & VERIFICATION SUITE")
    print("=" * 80)

    # Load data
    df = pd.read_parquet('scored_balls_v4.parquet')
    df['season_int'] = df['season'].astype(int)
    
    bat_v4 = pd.read_csv('ratings_batters_v4.csv')
    bowl_v4 = pd.read_csv('ratings_bowlers_v4.csv')
    teams_df = pd.read_csv('teams.csv')
    team_map = dict(zip(teams_df['team_id'], teams_df['team_name']))
    df['batting_team_name'] = df['batting_team_id'].map(team_map)
    df['bowling_team_name'] = df['bowling_team_id'].map(team_map)

    inn2 = df[(df['innings'] == 2) & df['in_scope_clean_v4']].copy()
    inn2 = inn2.sort_values(['match_id', 'innings_ball_num'])
    inn2['wp_next'] = inn2.groupby('match_id')['wp_v4'].shift(-1).fillna(inn2['batting_team_won'].astype(float))
    inn2['delta_wp'] = inn2['wp_next'] - inn2['wp_v4']
    inn2['neg_delta_wp'] = -inn2['delta_wp']

    # Batters deliveries
    bat_balls = inn2[inn2['is_legal'] | (inn2['batter_runs'] > 0)].copy()
    bat_balls['net_val'] = bat_balls['bat_rae_v4'] - 15.0 * bat_balls['bat_out_ae_v4']
    bat_balls['w'] = bat_balls['pressure_score_w_v4']
    bat_balls['w2'] = bat_balls['w'] ** 2
    bat_balls['wnet'] = bat_balls['w'] * bat_balls['net_val']
    bat_balls['is_ord'] = (bat_balls['pressure_pct_v4'] < 0.5).astype(float)
    bat_balls['ord_net'] = bat_balls['is_ord'] * bat_balls['net_val']
    bat_balls['is_hp'] = (bat_balls['is_high_pressure_chase_v4'] == True).astype(float)

    # Bowlers deliveries
    bowl_balls = inn2.copy()
    bowl_balls['net_val'] = bowl_balls['bowl_rae_v4'] + 15.0 * bowl_balls['bowl_wae_v4']
    bowl_balls['w'] = bowl_balls['pressure_score_w_v4']
    bowl_balls['w2'] = bowl_balls['w'] ** 2
    bowl_balls['wnet'] = bowl_balls['w'] * bowl_balls['net_val']
    bowl_balls['is_ord'] = (bowl_balls['pressure_pct_v4'] < 0.5).astype(float)
    bowl_balls['ord_net'] = bowl_balls['is_ord'] * bowl_balls['net_val']
    bowl_balls['is_hp'] = (bowl_balls['is_high_pressure_chase_v4'] == True).astype(float)

    # =========================================================================
    # 1. MISSING NAMES CHECK
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. MISSING NAMES AUDIT")
    print("=" * 80)

    target_batters = ['David Miller', 'Nicholas Pooran', 'Heinrich Klaasen', 'Virat Kohli', 'Phil Salt', 'Kamindu Mendis']
    target_bowlers = ['Rashid Khan', 'Axar Patel', 'Jasprit Bumrah', 'Sunil Narine', 'Matheesha Pathirana']

    print("BATTERS:")
    for name in target_batters:
        match = bat_v4[bat_v4['name'].str.contains(name, case=False, na=False)]
        if len(match) > 0:
            row = match.iloc[0]
            print(f"  {row['name']} (ID {row['player_id']}): Rank={int(row['rank_v4'])}, Rating={row['rating']:+.3f}, "
                  f"90% CI=[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}], ShrunkLevel={row['shrunk_level']:+.3f}, "
                  f"ShrunkDelta={row['shrunk_delta']:+.3f}, HP_Balls={int(row['hp_balls'])}, HP_Matches={int(row['hp_matches'])}")
        else:
            sub = bat_balls[bat_balls['striker_name'].str.contains(name, case=False, na=False)]
            if len(sub) > 0:
                hp_sub = sub[sub['is_hp'] == 1]
                hp_b = len(hp_sub)
                hp_m = hp_sub['match_id'].nunique()
                p_name = sub['striker_name'].iloc[0]
                p_id = sub['striker_id'].iloc[0]
                print(f"  {p_name} (ID {p_id}): INELIGIBLE (hp_balls={hp_b}, hp_matches={hp_m} < 8)")
            else:
                print(f"  {name}: NOT FOUND")

    print("\nBOWLERS:")
    for name in target_bowlers:
        match = bowl_v4[bowl_v4['name'].str.contains(name, case=False, na=False)]
        if len(match) > 0:
            row = match.iloc[0]
            print(f"  {row['name']} (ID {row['player_id']}): Rank={int(row['rank_v4'])}, Rating={row['rating']:+.3f}, "
                  f"90% CI=[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}], ShrunkLevel={row['shrunk_level']:+.3f}, "
                  f"ShrunkDelta={row['shrunk_delta']:+.3f}, HP_Balls={int(row['hp_balls'])}, HP_Matches={int(row['hp_matches'])}")
        else:
            sub = bowl_balls[bowl_balls['bowler_name'].str.contains(name, case=False, na=False)]
            if len(sub) > 0:
                hp_sub = sub[sub['is_hp'] == 1]
                hp_b = len(hp_sub)
                hp_m = hp_sub['match_id'].nunique()
                p_name = sub['bowler_name'].iloc[0]
                p_id = sub['bowler_id'].iloc[0]
                print(f"  {p_name} (ID {p_id}): INELIGIBLE (hp_balls={hp_b}, hp_matches={hp_m} < 8)")
            else:
                print(f"  {name}: NOT FOUND")

    # =========================================================================
    # 2. RELIABILITY TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. SPLIT-HALF RELIABILITY BENCHMARK (Odd/Even & 2022-24 vs 2025-26)")
    print("=" * 80)

    def compute_shrunk(balls_df, id_col, min_hp=30, min_matches=1):
        m_grp = balls_df.groupby([id_col, 'match_id']).agg(
            w=('w', 'sum'), w2=('w2', 'sum'), wnet=('wnet', 'sum'),
            ord_cnt=('is_ord', 'sum'), ord_net=('ord_net', 'sum'),
            hp_balls=('is_hp', 'sum')
        ).reset_index()

        p_tot = m_grp.groupby(id_col).agg(
            tot_w=('w', 'sum'), tot_w2=('w2', 'sum'), tot_wnet=('wnet', 'sum'),
            tot_ord_cnt=('ord_cnt', 'sum'), tot_ord_net=('ord_net', 'sum'),
            hp_balls=('hp_balls', 'sum'), hp_matches=('hp_balls', lambda x: (x > 0).sum())
        ).reset_index()

        p_tot['n_eff_p'] = (p_tot['tot_w'] ** 2) / p_tot['tot_w2']
        p_tot['n_o'] = p_tot['tot_ord_cnt']
        p_tot['pressure_level'] = p_tot['tot_wnet'] / p_tot['tot_w']
        p_tot['ordinary_level'] = np.where(p_tot['n_o'] > 0, p_tot['tot_ord_net'] / p_tot['n_o'], np.nan)
        p_tot['delta'] = p_tot['pressure_level'] - p_tot['ordinary_level']

        qual = p_tot[(p_tot['hp_balls'] >= min_hp) & (p_tot['hp_matches'] >= min_matches) & p_tot['delta'].notna() & (p_tot['n_o'] >= 10)].copy()

        sigma2_p = balls_df['net_val'].var()
        sigma2_o = balls_df.loc[balls_df['is_ord'] == 1, 'net_val'].var()

        var_lev = qual['pressure_level'].var()
        mean_v_lev = (sigma2_p / qual['n_eff_p']).mean()
        tau2_lev = max(0.0, var_lev - mean_v_lev)
        k_lev = sigma2_p / tau2_lev if tau2_lev > 0 else 1e6
        lam_lev = qual['n_eff_p'] / (qual['n_eff_p'] + k_lev)
        qual['shrunk_level'] = lam_lev * qual['pressure_level'] + (1 - lam_lev) * qual['pressure_level'].mean()

        qual['v_delta'] = (sigma2_p / qual['n_eff_p']) + (sigma2_o / qual['n_o'])
        var_del = qual['delta'].var()
        mean_v_del = qual['v_delta'].mean()
        tau2_del = max(0.0, var_del - mean_v_del)
        lam_del = tau2_del / (tau2_del + qual['v_delta'])
        qual['shrunk_delta'] = lam_del * qual['delta'] + (1 - lam_del) * qual['delta'].mean()

        for lam in [0.0, 0.5, 1.0]:
            qual[f'rating_lam_{lam}'] = qual['shrunk_level'] + lam * qual['shrunk_delta']
        return qual

    def run_split(balls_df, id_col, role):
        rows = []
        # Parity split
        ha_par = balls_df[balls_df['match_id'] % 2 == 1].copy()
        hb_par = balls_df[balls_df['match_id'] % 2 == 0].copy()
        hp_a = ha_par[ha_par['is_hp'] == 1].groupby(id_col)['w'].count()
        hp_b = hb_par[hb_par['is_hp'] == 1].groupby(id_col)['w'].count()
        com_par = sorted(list(set(hp_a[hp_a >= 30].index).intersection(set(hp_b[hp_b >= 30].index))))
        
        qa_par = compute_shrunk(ha_par[ha_par[id_col].isin(com_par)], id_col, min_hp=30, min_matches=1)
        qb_par = compute_shrunk(hb_par[hb_par[id_col].isin(com_par)], id_col, min_hp=30, min_matches=1)
        m_par = pd.merge(qa_par, qb_par, on=id_col, suffixes=('_A', '_B'))
        
        # Time split
        ha_tim = balls_df[balls_df['season_int'].isin([2022, 2023, 2024])].copy()
        hb_tim = balls_df[balls_df['season_int'].isin([2025, 2026])].copy()
        hp_at = ha_tim[ha_tim['is_hp'] == 1].groupby(id_col)['w'].count()
        hp_bt = hb_tim[hb_tim['is_hp'] == 1].groupby(id_col)['w'].count()
        com_tim = sorted(list(set(hp_at[hp_at >= 30].index).intersection(set(hp_bt[hp_bt >= 30].index))))
        
        qa_tim = compute_shrunk(ha_tim[ha_tim[id_col].isin(com_tim)], id_col, min_hp=30, min_matches=1)
        qb_tim = compute_shrunk(hb_tim[hb_tim[id_col].isin(com_tim)], id_col, min_hp=30, min_matches=1)
        m_tim = pd.merge(qa_tim, qb_tim, on=id_col, suffixes=('_A', '_B'))

        for lam in [0.0, 0.5, 1.0]:
            col_a = f'rating_lam_{lam}_A'
            col_b = f'rating_lam_{lam}_B'
            
            rho_p, _ = spearmanr(m_par[col_a], m_par[col_b])
            t10_ap = set(m_par.sort_values(col_a, ascending=False).head(10)[id_col])
            t10_bp = set(m_par.sort_values(col_b, ascending=False).head(10)[id_col])
            t20_ap = set(m_par.sort_values(col_a, ascending=False).head(20)[id_col])
            t20_bp = set(m_par.sort_values(col_b, ascending=False).head(20)[id_col])
            
            rho_t, _ = spearmanr(m_tim[col_a], m_tim[col_b])
            t10_at = set(m_tim.sort_values(col_a, ascending=False).head(10)[id_col])
            t10_bt = set(m_tim.sort_values(col_b, ascending=False).head(10)[id_col])
            t20_at = set(m_tim.sort_values(col_a, ascending=False).head(20)[id_col])
            t20_bt = set(m_tim.sort_values(col_b, ascending=False).head(20)[id_col])
            
            rows.append({
                'Role': role,
                'lambda': lam,
                'n_parity': len(m_par),
                'Spearman_parity': round(rho_p, 4),
                'Top10_parity': len(t10_ap.intersection(t10_bp)),
                'Top20_parity': len(t20_ap.intersection(t20_bp)),
                'n_time': len(m_tim),
                'Spearman_time': round(rho_t, 4),
                'Top10_time': len(t10_at.intersection(t10_bt)),
                'Top20_time': len(t20_at.intersection(t20_bt)),
            })
        return rows

    rel_rows = run_split(bat_balls, 'striker_id', 'Batter') + run_split(bowl_balls, 'bowler_id', 'Bowler')
    rel_table = pd.DataFrame(rel_rows)
    print(rel_table.to_string(index=False))

    # =========================================================================
    # 3. DROP-ONE-MATCH ROBUSTNESS
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. DROP-ONE-MATCH SENSITIVITY (Impact of Removing Single Best Match)")
    print("=" * 80)

    def evaluate_drop_one(balls_df, ratings_df, id_col, role_name):
        print(f"\n--- {role_name} Top 10 Drop-One Match ---")
        m_grp = balls_df.groupby([id_col, 'match_id']).agg(
            w=('w', 'sum'), w2=('w2', 'sum'), wnet=('wnet', 'sum'),
            ord_cnt=('is_ord', 'sum'), ord_net=('ord_net', 'sum'),
            hp_balls=('is_hp', 'sum')
        ).reset_index()

        p_stats = m_grp.groupby(id_col).agg(
            tot_w=('w', 'sum'), tot_w2=('w2', 'sum'), tot_wnet=('wnet', 'sum'),
            tot_ord_cnt=('ord_cnt', 'sum'), tot_ord_net=('ord_net', 'sum')
        ).reset_index()
        p_stats['n_eff_p'] = (p_stats['tot_w'] ** 2) / p_stats['tot_w2']
        p_stats['n_o'] = p_stats['tot_ord_cnt']
        p_stats_map = dict(zip(p_stats[id_col], p_stats['n_eff_p']))
        p_no_map = dict(zip(p_stats[id_col], p_stats['n_o']))

        sigma2_p = balls_df['net_val'].var()
        sigma2_o = balls_df.loc[balls_df['is_ord'] == 1, 'net_val'].var()

        # Baseline parameters from full eligible
        var_lev = ratings_df['pressure_level'].var()
        n_eff_arr = ratings_df['player_id'].map(p_stats_map).values
        mean_v_lev = (sigma2_p / n_eff_arr).mean()
        tau2_lev = max(0.0, var_lev - mean_v_lev)
        k_lev = sigma2_p / tau2_lev if tau2_lev > 0 else 1e6
        mu_lev = ratings_df['pressure_level'].mean()

        var_del = ratings_df['delta'].var()
        n_o_arr = ratings_df['player_id'].map(p_no_map).values
        v_del_arr = (sigma2_p / n_eff_arr) + (sigma2_o / n_o_arr)
        mean_v_del = v_del_arr.mean()
        tau2_del = max(0.0, var_del - mean_v_del)
        mu_del = ratings_df['delta'].mean()

        for _, row in ratings_df.head(10).iterrows():
            pid = row['player_id']
            p_matches = m_grp[m_grp[id_col] == pid].copy()
            
            tot_w = p_matches['w'].sum()
            tot_w2 = p_matches['w2'].sum()
            tot_wnet = p_matches['wnet'].sum()
            tot_ord_cnt = p_matches['ord_cnt'].sum()
            tot_ord_net = p_matches['ord_net'].sum()
            cur_lev = tot_wnet / tot_w if tot_w > 0 else 0

            # Find match that contributed most to pressure_level
            p_matches['w_rem'] = tot_w - p_matches['w']
            p_matches['wnet_rem'] = tot_wnet - p_matches['wnet']
            p_matches['lev_rem'] = np.where(p_matches['w_rem'] > 0, p_matches['wnet_rem'] / p_matches['w_rem'], np.nan)
            p_matches['lev_drop'] = cur_lev - p_matches['lev_rem']
            
            best_m = p_matches.sort_values('lev_drop', ascending=False).iloc[0]
            drop_mid = int(best_m['match_id'])

            # Recompute player metrics without best_m
            w_rem = tot_w - best_m['w']
            w2_rem = tot_w2 - best_m['w2']
            wnet_rem = tot_wnet - best_m['wnet']
            ord_cnt_rem = tot_ord_cnt - best_m['ord_cnt']
            ord_net_rem = tot_ord_net - best_m['ord_net']

            n_eff_rem = (w_rem ** 2) / w2_rem if w2_rem > 0 else 1.0
            n_o_rem = ord_cnt_rem
            lev_rem = wnet_rem / w_rem if w_rem > 0 else 0.0
            ord_rem = ord_net_rem / n_o_rem if n_o_rem > 0 else 0.0
            del_rem = lev_rem - ord_rem

            lam_l = n_eff_rem / (n_eff_rem + k_lev)
            shrunk_l = lam_l * lev_rem + (1 - lam_l) * mu_lev

            v_del = (sigma2_p / n_eff_rem) + (sigma2_o / n_o_rem) if n_o_rem > 0 else 1.0
            lam_d = tau2_del / (tau2_del + v_del)
            shrunk_d = lam_d * del_rem + (1 - lam_d) * mu_del

            new_rating = shrunk_l + 0.5 * shrunk_d

            # New rank among all eligible players
            other_ratings = ratings_df[ratings_df['player_id'] != pid]['rating'].values
            all_new = np.append(other_ratings, new_rating)
            new_rank = int((all_new > new_rating).sum() + 1)

            print(f"  #{int(row['rank_v4'])} {row['name']}: Orig Rating={row['rating']:+.3f} -> New Rating={new_rating:+.3f} | "
                  f"Best Match ID={drop_mid} | Lev Drop={best_m['lev_drop']:.3f} | New Rank: #{new_rank} (of {len(ratings_df)})")

    evaluate_drop_one(bat_balls, bat_v4, 'striker_id', 'Batters')
    evaluate_drop_one(bowl_balls, bowl_v4, 'bowler_id', 'Bowlers')

    # =========================================================================
    # 4. SENSITIVITY GRID (18 Configurations)
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. SENSITIVITY GRID (W in {10,15,20} x Weighting {continuous, binary>=0.8, binary>=0.7} x lam in {0.5,1.0})")
    print("=" * 80)

    def run_sensitivity_grid(balls_df, ratings_df, id_col, role_name):
        eligible_ids = set(ratings_df['player_id'])
        id_to_name = dict(zip(ratings_df['player_id'], ratings_df['name']))
        top10_counts = defaultdict(int)

        weights_opts = ['continuous', 'binary_08', 'binary_07']
        W_opts = [10.0, 15.0, 20.0]
        lam_opts = [0.5, 1.0]

        for W in W_opts:
            if role_name == 'Batters':
                net_val = balls_df['bat_rae_v4'] - W * balls_df['bat_out_ae_v4']
            else:
                net_val = balls_df['bowl_rae_v4'] + W * balls_df['bowl_wae_v4']

            for w_type in weights_opts:
                if w_type == 'continuous':
                    w = balls_df['pressure_score_w_v4'].values
                elif w_type == 'binary_08':
                    w = (balls_df['pressure_pct_v4'].values >= 0.8).astype(float)
                else: # binary_07
                    w = (balls_df['pressure_pct_v4'].values >= 0.7).astype(float)

                w2 = w ** 2
                wnet = w * net_val.values
                is_ord = (balls_df['pressure_pct_v4'].values < 0.5).astype(float)
                ord_net = is_ord * net_val.values

                agg_df = pd.DataFrame({
                    id_col: balls_df[id_col],
                    'w': w, 'w2': w2, 'wnet': wnet,
                    'is_ord': is_ord, 'ord_net': ord_net
                }).groupby(id_col).sum().reset_index()

                agg_df = agg_df[agg_df[id_col].isin(eligible_ids)].copy()
                agg_df['n_eff_p'] = (agg_df['w'] ** 2) / agg_df['w2']
                agg_df['n_o'] = agg_df['is_ord']
                agg_df['pressure_level'] = agg_df['wnet'] / agg_df['w']
                agg_df['ordinary_level'] = agg_df['ord_net'] / agg_df['n_o']
                agg_df['delta'] = agg_df['pressure_level'] - agg_df['ordinary_level']

                sigma2_p = net_val.var()
                sigma2_o = net_val[balls_df['pressure_pct_v4'] < 0.5].var()

                var_lev = agg_df['pressure_level'].var()
                mean_v_lev = (sigma2_p / agg_df['n_eff_p']).mean()
                tau2_lev = max(0.0, var_lev - mean_v_lev)
                k_lev = sigma2_p / tau2_lev if tau2_lev > 0 else 1e6
                lam_lev = agg_df['n_eff_p'] / (agg_df['n_eff_p'] + k_lev)
                agg_df['shrunk_level'] = lam_lev * agg_df['pressure_level'] + (1 - lam_lev) * agg_df['pressure_level'].mean()

                agg_df['v_delta'] = (sigma2_p / agg_df['n_eff_p']) + (sigma2_o / agg_df['n_o'])
                var_del = agg_df['delta'].var()
                mean_v_del = agg_df['v_delta'].mean()
                tau2_del = max(0.0, var_del - mean_v_del)
                lam_del = tau2_del / (tau2_del + agg_df['v_delta'])
                agg_df['shrunk_delta'] = lam_del * agg_df['delta'] + (1 - lam_del) * agg_df['delta'].mean()

                for lam in lam_opts:
                    agg_df['rating'] = agg_df['shrunk_level'] + lam * agg_df['shrunk_delta']
                    top10 = agg_df.sort_values('rating', ascending=False).head(10)[id_col].tolist()
                    for pid in top10:
                        top10_counts[pid] += 1

        print(f"\n--- {role_name} Robustness Across 18 Configs (Top 10 Appearances) ---")
        robust_list = []
        for pid in eligible_ids:
            cnt = top10_counts.get(pid, 0)
            robust_list.append({
                'player_id': pid,
                'name': id_to_name.get(pid, str(pid)),
                'configs_top10': cnt,
                'pct_configs': cnt / 18.0
            })
        robust_df = pd.DataFrame(robust_list).sort_values('configs_top10', ascending=False)
        
        print(f"Players in >= 50% of configurations (>= 9 of 18):")
        print(robust_df[robust_df['configs_top10'] >= 9].to_string(index=False))
        return robust_df

    bat_robust = run_sensitivity_grid(bat_balls, bat_v4, 'striker_id', 'Batters')
    bowl_robust = run_sensitivity_grid(bowl_balls, bowl_v4, 'bowler_id', 'Bowlers')

    # =========================================================================
    # 5. OPPOSITION STRENGTH & BATTING ORDER / BOWLING TYPE BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. OPPOSITION QUALITY & ROLE-SPECIFIC BENCHMARKS")
    print("=" * 80)

    major_teams = {'India', 'Pakistan', 'West Indies', 'New Zealand', 'Bangladesh', 
                   'Sri Lanka', 'England', 'Australia', 'Zimbabwe', 'Afghanistan', 
                   'South Africa', 'Ireland'}

    hp_all = inn2[inn2['is_high_pressure_chase_v4'] == True].copy()
    mean_bo_league = hp_all['batting_order'].mean()
    share_tail_league = (hp_all['batting_order'] >= 8).mean()

    bat_hp_all = hp_all[hp_all['is_legal'] | (hp_all['batter_runs'] > 0)].copy()
    pace_share_league = (bat_hp_all['pace_or_spin'] == 'Pace').mean()
    spin_share_league = (bat_hp_all['pace_or_spin'] == 'Spin').mean()

    print(f"League Benchmarks on High-Pressure Chase Deliveries:")
    print(f"  Bowlers: Mean Batting Order faced = {mean_bo_league:.2f} | Share Tailenders (BO >= 8) = {share_tail_league*100:.1f}%")
    print(f"  Batters: Pace Faced = {pace_share_league*100:.1f}% | Spin Faced = {spin_share_league*100:.1f}%\n")

    print("BATTERS (Top 10):")
    for _, r in bat_v4.head(10).iterrows():
        p_b = bat_hp_all[bat_hp_all['striker_id'] == r['player_id']]
        ipl_sh = (p_b['comp_format'] == 'IPL').mean()
        t20i_sh = (p_b['comp_format'] == 'T20I').mean()
        non_major = ((p_b['comp_format'] == 'T20I') & (~p_b['bowling_team_name'].isin(major_teams))).mean()
        pace_sh = (p_b['pace_or_spin'] == 'Pace').mean()
        spin_sh = (p_b['pace_or_spin'] == 'Spin').mean()
        print(f"  #{int(r['rank_v4'])} {r['name']:<22}: IPL={ipl_sh*100:5.1f}% | T20I={t20i_sh*100:5.1f}% | Non-Major={non_major*100:4.1f}% | "
              f"Pace={pace_sh*100:5.1f}% (vs {pace_share_league*100:.1f}%) | Spin={spin_sh*100:5.1f}% (vs {spin_share_league*100:.1f}%)")

    print("\nBOWLERS (Top 10):")
    for _, r in bowl_v4.head(10).iterrows():
        p_b = hp_all[hp_all['bowler_id'] == r['player_id']]
        ipl_sh = (p_b['comp_format'] == 'IPL').mean()
        t20i_sh = (p_b['comp_format'] == 'T20I').mean()
        non_major = ((p_b['comp_format'] == 'T20I') & (~p_b['batting_team_name'].isin(major_teams))).mean()
        mean_bo = p_b['batting_order'].mean()
        tail_sh = (p_b['batting_order'] >= 8).mean()
        print(f"  #{int(r['rank_v4'])} {r['name']:<22}: IPL={ipl_sh*100:5.1f}% | T20I={t20i_sh*100:5.1f}% | Non-Major={non_major*100:4.1f}% | "
              f"Mean BO={mean_bo:4.2f} (vs {mean_bo_league:.2f}) | Tail (>=8)={tail_sh*100:4.1f}% (vs {share_tail_league*100:.1f}%)")

    # Name-by-name overlap with restricted benchmark
    # Restricted major data
    restr_mask = (df['comp_format'] == 'IPL') | ((df['comp_format'] == 'T20I') & df['batting_team_name'].isin(major_teams) & df['bowling_team_name'].isin(major_teams))
    restr_inn2 = df[restr_mask & (df['innings'] == 2) & df['in_scope_clean_v4']].copy()
    restr_hp = restr_inn2[restr_inn2['is_high_pressure_chase_v4'] == True]

    # Quick compute restricted ratings for overlap check
    def get_restr_top10(balls_df, id_col, role):
        hp_counts = balls_df[balls_df['is_hp'] == 1].groupby(id_col)['w'].count()
        m_counts = balls_df[balls_df['is_hp'] == 1].groupby(id_col)['match_id'].nunique()
        qual_ids = list(set(hp_counts[hp_counts >= 40].index).intersection(set(m_counts[m_counts >= 5].index)))
        sub = balls_df[balls_df[id_col].isin(qual_ids)].copy()
        q = compute_shrunk(sub, id_col, min_hp=40, min_matches=5)
        return q.sort_values('rating_lam_0.5', ascending=False).head(10)[id_col].tolist()

    bat_balls_restr = bat_balls[bat_balls['match_id'].isin(restr_inn2['match_id'])].copy()
    bowl_balls_restr = bowl_balls[bowl_balls['match_id'].isin(restr_inn2['match_id'])].copy()

    restr_bat_ids = get_restr_top10(bat_balls_restr, 'striker_id', 'Batters')
    restr_bowl_ids = get_restr_top10(bowl_balls_restr, 'bowler_id', 'Bowlers')

    id_to_name_all = dict(zip(df['striker_id'], df['striker_name']))
    id_to_name_all.update(dict(zip(df['bowler_id'], df['bowler_name'])))
    id_to_name_all.update(dict(zip(bat_v4['player_id'], bat_v4['name'])))
    id_to_name_all.update(dict(zip(bowl_v4['player_id'], bowl_v4['name'])))

    main_bat_top10 = bat_v4.head(10)['player_id'].tolist()
    main_bowl_top10 = bowl_v4.head(10)['player_id'].tolist()

    bat_overlap = set(main_bat_top10).intersection(set(restr_bat_ids))
    bowl_overlap = set(main_bowl_top10).intersection(set(restr_bowl_ids))

    print(f"\nName-by-Name Overlap with Restricted Benchmark (IPL + Major T20Is):")
    print(f"  Batters Overlap: {len(bat_overlap)}/10")
    print(f"    Main V4 Top 10:       {[id_to_name_all.get(x) for x in main_bat_top10]}")
    print(f"    Restricted Top 10:    {[id_to_name_all.get(x) for x in restr_bat_ids]}")
    print(f"  Bowlers Overlap: {len(bowl_overlap)}/10")
    print(f"    Main V4 Top 10:       {[id_to_name_all.get(x) for x in main_bowl_top10]}")
    print(f"    Restricted Top 10:    {[id_to_name_all.get(x) for x in restr_bowl_ids]}")

    # =========================================================================
    # 6. WIN PROBABILITY ADDED (WPA) AUDIT
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. WIN PROBABILITY ADDED (WPA) AUDIT")
    print("=" * 80)

    bat_med_wpa = bat_v4['wpa_per_hp_ball'].median()
    bowl_med_wpa = bowl_v4['wpa_per_hp_ball'].median()

    rho_bat_wpa, _ = spearmanr(bat_v4['rating'], bat_v4['wpa_per_hp_ball'])
    rho_bowl_wpa, _ = spearmanr(bowl_v4['rating'], bowl_v4['wpa_per_hp_ball'])

    print(f"Batters: Spearman(Rating, WPA/HP Ball) = {rho_bat_wpa:+.4f} | League Median = {bat_med_wpa:+.4f}")
    print(f"Bowlers: Spearman(Rating, WPA/HP Ball) = {rho_bowl_wpa:+.4f} | League Median = {bowl_med_wpa:+.4f}\n")

    print("BATTERS WPA (Top 10 and 11-20):")
    below_med_bat = []
    for _, r in bat_v4.head(20).iterrows():
        rk = int(r['rank_v4'])
        flag = " [BELOW MEDIAN]" if r['wpa_per_hp_ball'] < bat_med_wpa else ""
        if rk <= 10 and r['wpa_per_hp_ball'] < bat_med_wpa:
            below_med_bat.append(r['name'])
        prefix = "Top 10" if rk <= 10 else "11-20 "
        print(f"  [{prefix}] #{rk:2d} {r['name']:<22}: HP_Balls={int(r['hp_balls']):3d} | Total WPA={r['wpa_total']:+6.3f} | WPA/Ball={r['wpa_per_hp_ball']:+6.4f}{flag}")

    print(f"\nTop 10 Batters Below League Median WPA: {below_med_bat if len(below_med_bat) > 0 else 'None (0/10)'}")

    print("\nBOWLERS WPA (Top 10 and 11-20):")
    below_med_bowl = []
    for _, r in bowl_v4.head(20).iterrows():
        rk = int(r['rank_v4'])
        flag = " [BELOW MEDIAN]" if r['wpa_per_hp_ball'] < bowl_med_wpa else ""
        if rk <= 10 and r['wpa_per_hp_ball'] < bowl_med_wpa:
            below_med_bowl.append(r['name'])
        prefix = "Top 10" if rk <= 10 else "11-20 "
        print(f"  [{prefix}] #{rk:2d} {r['name']:<22}: HP_Balls={int(r['hp_balls']):3d} | Total WPA={r['wpa_total']:+6.3f} | WPA/Ball={r['wpa_per_hp_ball']:+6.4f}{flag}")

    print(f"\nTop 10 Bowlers Below League Median WPA: {below_med_bowl if len(below_med_bowl) > 0 else 'None (0/10)'}")

    # =========================================================================
    # 7. FINAL ORDER GENERATION & EXPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. FINAL ORDER GENERATION (Rank by Bootstrap Median Rank, Tie-Break: Top10 Prob, CI Low)")
    print("=" * 80)

    # Attach sensitivity grid counts
    bat_v4['sensitivity_top10'] = bat_v4['player_id'].map(dict(zip(bat_robust['player_id'], bat_robust['configs_top10']))).fillna(0).astype(int)
    bowl_v4['sensitivity_top10'] = bowl_v4['player_id'].map(dict(zip(bowl_robust['player_id'], bowl_robust['configs_top10']))).fillna(0).astype(int)

    def attach_flags(df, med_wpa):
        flags = []
        for _, r in df.iterrows():
            fl = []
            if r['ci_low'] <= 0:
                fl.append("CI includes 0")
            if r['hp_matches'] < 12:
                fl.append("Matches < 12")
            if r['wpa_per_hp_ball'] < med_wpa:
                fl.append("WPA < Median")
            flags.append("; ".join(fl) if len(fl) > 0 else "None")
        df['flags'] = flags
        return df

    bat_v4 = attach_flags(bat_v4, bat_med_wpa)
    bowl_v4 = attach_flags(bowl_v4, bowl_med_wpa)

    # Final sorting
    bat_final = bat_v4.sort_values(by=['median_rank', 'top10_prob', 'ci_low'], ascending=[True, False, False]).reset_index(drop=True)
    bat_final['rank_final'] = bat_final.index + 1

    bowl_final = bowl_v4.sort_values(by=['median_rank', 'top10_prob', 'ci_low'], ascending=[True, False, False]).reset_index(drop=True)
    bowl_final['rank_final'] = bowl_final.index + 1

    # Save outputs
    bat_final.to_csv('ratings_batters_final.csv', index=False)
    bowl_final.to_csv('ratings_bowlers_final.csv', index=False)
    print("Saved ratings_batters_final.csv and ratings_bowlers_final.csv.")

    print("\nPROPOSED FINAL TOP 10 BATTERS:")
    print(f"{'Final':<6} {'V4':<5} {'Player Name':<22} {'Rating':<8} {'90% CI':<18} {'Top10%':<8} {'Grid/18':<8} {'Flags'}")
    print("-" * 90)
    for i in range(10):
        r = bat_final.iloc[i]
        ci_str = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        print(f"#{r['rank_final']:<5} #{r['rank_v4']:<4} {r['name']:<22} {r['rating']:+6.3f}   {ci_str:<18} {r['top10_prob']*100:5.1f}%   {r['sensitivity_top10']:2d}/18     {r['flags']}")

    print("\nPROPOSED FINAL TOP 10 BOWLERS:")
    print(f"{'Final':<6} {'V4':<5} {'Player Name':<22} {'Rating':<8} {'90% CI':<18} {'Top10%':<8} {'Grid/18':<8} {'Flags'}")
    print("-" * 90)
    for i in range(10):
        r = bowl_final.iloc[i]
        ci_str = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        print(f"#{r['rank_final']:<5} #{r['rank_v4']:<4} {r['name']:<22} {r['rating']:+6.3f}   {ci_str:<18} {r['top10_prob']*100:5.1f}%   {r['sensitivity_top10']:2d}/18     {r['flags']}")

    # =========================================================================
    # 8. REPORT AUDIT
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. METHODOLOGY REPORT AUDIT & CONSISTENCY CHECK")
    print("=" * 80)

    if os.path.exists('METHODOLOGY_REPORT.md'):
        with open('METHODOLOGY_REPORT.md', 'r', encoding='utf-8') as f:
            md_text = f.read()

        mismatches = []
        # Audit top 10 batters numbers
        for _, r in bat_v4.head(10).iterrows():
            pid = str(r['player_id'])
            if pid not in md_text:
                mismatches.append(f"Batter ID {pid} ({r['name']}) missing in report")
            rate_str = f"{r['rating']:.3f}"
            if rate_str not in md_text:
                mismatches.append(f"Batter rating {rate_str} for {r['name']} not in report")

        # Audit top 10 bowlers numbers
        for _, r in bowl_v4.head(10).iterrows():
            pid = str(r['player_id'])
            if pid not in md_text:
                mismatches.append(f"Bowler ID {pid} ({r['name']}) missing in report")
            rate_str = f"{r['rating']:.3f}"
            if rate_str not in md_text:
                mismatches.append(f"Bowler rating {rate_str} for {r['name']} not in report")

        if len(mismatches) == 0:
            print("[✓] Zero mismatches detected between V4 top-10 numbers in CSV and METHODOLOGY_REPORT.md!")
        else:
            print(f"[!] Detected {len(mismatches)} mismatches:")
            for m in mismatches:
                print(f"    - {m}")
    else:
        print("METHODOLOGY_REPORT.md not found.")

    print("\n" + "=" * 80)
    print("FINAL CHECKS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
