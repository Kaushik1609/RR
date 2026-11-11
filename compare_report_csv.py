import pandas as pd

bat_csv = pd.read_csv('ratings_batters_final3.csv')
bowl_csv = pd.read_csv('ratings_bowlers_final3.csv')

with open('METHODOLOGY_REPORT.md', 'r', encoding='utf-8') as f:
    text = f.read()

print("=== BATTERS: CSV vs METHODOLOGY_REPORT.md ===")
print(f"{'Rank':<5} {'Player Name':<22} {'Player ID':<10} {'CSV Rating':<12} {'In Report?'}")
print("-" * 65)
for idx, r in bat_csv.iterrows():
    rk = r['final_rank']
    name = r['name']
    pid = r['player_id']
    rate = f"{r['rating']:+.3f}"
    in_rep = (str(pid) in text) and (name in text)
    print(f"#{rk:<4} {name:<22} {pid:<10} {rate:<12} {'YES' if in_rep else 'NO'}")

print("\n=== BOWLERS: CSV vs METHODOLOGY_REPORT.md ===")
print(f"{'Rank':<5} {'Player Name':<22} {'Player ID':<10} {'CSV Rating':<12} {'In Report?'}")
print("-" * 65)
for idx, r in bowl_csv.iterrows():
    rk = r['final_rank']
    name = r['name']
    pid = r['player_id']
    rate = f"{r['rating']:+.3f}"
    in_rep = (str(pid) in text) and (name in text)
    print(f"#{rk:<4} {name:<22} {pid:<10} {rate:<12} {'YES' if in_rep else 'NO'}")

