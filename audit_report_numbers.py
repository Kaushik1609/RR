import pandas as pd
import re

with open('METHODOLOGY_REPORT.md', 'r', encoding='utf-8') as f:
    text = f.read()

bat = pd.read_csv('ratings_batters_final3.csv')
bowl = pd.read_csv('ratings_bowlers_final3.csv')
cand_bat = pd.read_csv('candidates_batters_final3.csv')
cand_bowl = pd.read_csv('candidates_bowlers_final3.csv')

print('=== 1. CHECKING TOP 10 BATTERS IN REPORT ===')
for _, r in bat.iterrows():
    name = r['name']
    pid = str(r['player_id'])
    rating_str = f"{r['rating']:+.3f}"
    ci_str = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
    assert name in text, f'Missing {name}'
    assert pid in text, f'Missing ID {pid} for {name}'
    assert rating_str in text, f'Missing rating {rating_str} for {name}'
    assert ci_str in text, f'Missing CI {ci_str} for {name}'
    print(f"  [OK] {name} ({pid}): rating={rating_str}, CI={ci_str}")

print('\n=== 2. CHECKING TOP 10 BOWLERS IN REPORT ===')
for _, r in bowl.iterrows():
    name = r['name']
    pid = str(r['player_id'])
    rating_str = f"{r['rating']:+.3f}"
    ci_str = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
    assert name in text, f'Missing {name}'
    assert pid in text, f'Missing ID {pid} for {name}'
    assert rating_str in text, f'Missing rating {rating_str} for {name}'
    assert ci_str in text, f'Missing CI {ci_str} for {name}'
    print(f"  [OK] {name} ({pid}): rating={rating_str}, CI={ci_str}")

print('\n=== 3. CHECKING SPLIT-HALF AND RESTRICTED NUMBERS ===')
expected_numbers = [
    ("-0.0608", "Batter time-split raw level rho"),
    ("+0.0914", "Batter time-split raw delta rho"),
    ("+0.433", "Restricted Kamindu Mendis rating"),
    ("+0.405", "Restricted Ashutosh Sharma rating"),
    ("+0.404", "Restricted Vaibhav Sooryavanshi rating"),
    ("+0.396", "Restricted Suryakumar Yadav rating"),
    ("+0.389", "Restricted Shreyas Iyer rating"),
    ("+0.337", "Restricted Shubman Gill rating"),
    ("+0.336", "Restricted Tilak Varma rating"),
    ("+0.328", "Restricted Tim David rating"),
    ("+0.324", "Restricted Mitchell Marsh rating"),
    ("+0.320", "Restricted Phil Salt rating"),
    ("+0.487", "Restricted Taskin Ahmed rating"),
    ("+0.476", "Restricted Obed McCoy rating"),
    ("+0.386", "Restricted Josh Hazlewood rating"),
    ("+0.329", "Restricted Harpreet Brar rating"),
    ("+0.325", "Restricted Jacob Duffy rating"),
    ("+0.318", "Restricted Shakib Al Hasan rating"),
    ("+0.300", "Restricted Harshit Rana rating"),
    ("+0.269", "Restricted Gudakesh Motie rating"),
    ("+0.267", "Restricted Dasun Shanaka rating"),
    ("7 / 10", "Batter overlap count"),
    ("8 / 10", "Bowler overlap count")
]

for val, desc in expected_numbers:
    assert val in text, f"Missing {desc} ({val})"
    print(f"  [OK] {desc}: {val}")

print('\n[✓] ZERO MISMATCHES FOUND ACROSS ALL AUDITED METRICS.')
