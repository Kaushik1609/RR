# SupeRR Selector Season 2 — Situational Pressure Analytics 🏏

> **Rajasthan Royals Round 1 Analytics Challenge**  
> Data-driven system measuring batter and bowler performance under extreme chase pressure (IPL & T20I, 2022–2026).

---

## 📌 Executive Summary

This repository hosts the complete modeling pipeline, methodology documentation, interactive intelligence dashboard, and final stability-screened player rankings for **SupeRR Selector Season 2**.

- **Dataset**: 777 regulation T20 matches (IPL & T20I, 2022–2026), 200,999 deliveries, 37,171 high-pressure balls ($P \ge 80\%$).
- **Methodology**: In-game chase win probability modeling (LightGBM), dynamic leverage weighting, Empirical Bayes shrinkage ($\lambda = 0.5$), opponent batting order adjustments, and a **4-Point Stability Screen** (cluster bootstrap CI, 18-config sensitivity grid, restricted tournament benchmark, and drop-one match resilience).

---

## 🏆 Final Official Top 10 Rosters (Unstop Ready)

### 🏏 Top 10 Batters Under Pressure (Auction Realistic)
| Rank | Player Name | Player ID | Team Representation | Rating (r/b) | 90% Cluster CI | Pass Count | Median Rank |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Ashutosh Sharma** | `8101` | Punjab Kings / Delhi Capitals | **+0.384** | `[+0.134, +0.624]` | 3 / 4 | #6 |
| **2** | **Pathum Nissanka** | `6931` | Sri Lanka / Delhi Capitals | **+0.341** | `[+0.141, +0.568]` | 3 / 4 | #10 |
| **3** | **Tim David** | `6396` | Mumbai Indians / Australia | **+0.304** | `[+0.054, +0.561]` | 2 / 4 | #13 |
| **4** | **Mitchell Marsh** | `818` | Delhi Capitals / Australia | **+0.302** | `[+0.121, +0.481]` | 2 / 4 | #13 |
| **5** | **Evin Lewis** | `2573` | Lucknow Super Giants / West Indies | **+0.294** | `[+0.051, +0.487]` | 2 / 4 | #14 |
| **6** | **David Miller** | `1077` | Gujarat Titans / South Africa | **+0.286** | `[+0.028, +0.530]` | 2 / 4 | #15 |
| **7** | **Phil Salt** | `5553` | England / Delhi Capitals | **+0.248** | `[+0.007, +0.459]` | 2 / 4 | #20 |
| **8** | **Reeza Hendricks** | `768` | South Africa | **+0.267** | `[+0.075, +0.421]` | 1 / 4 | #18 |
| **9** | **Prabhsimran Singh** | `7216` | Punjab Kings / India A | **+0.252** | `[+0.087, +0.442]` | 1 / 4 | #18 |
| **10** | **Washington Sundar** | `4823` | India / Sunrisers Hyderabad | **+0.228** | `[+0.082, +0.382]` | 1 / 4 | #23 |

### 🎯 Top 10 Bowlers Under Pressure (Auction Realistic)
| Rank | Bowler Name | Player ID | Team Representation | Rating (r/b) | 90% Cluster CI | Pass Count | Median Rank |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Taskin Ahmed** | `2681` | Bangladesh | **+0.359** | `[+0.128, +0.590]` | 4 / 4 | #6 |
| **2** | **Obed McCoy** | `4837` | Rajasthan Royals / West Indies | **+0.367** | `[+0.018, +0.638]` | 4 / 4 | #7 |
| **3** | **Jacob Duffy** | `22920` | New Zealand / Royal Challengers Bengaluru | **+0.288** | `[+0.036, +0.562]` | 4 / 4 | #10 |
| **4** | **Harpreet Brar** | `7325` | Punjab Kings | **+0.279** | `[+0.005, +0.594]` | 3 / 4 | #12 |
| **5** | **Gudakesh Motie** | `3276` | West Indies | **+0.235** | `[+0.010, +0.461]` | 3 / 4 | #17 |
| **6** | **Dasun Shanaka** | `2436` | Sri Lanka / Gujarat Titans | **+0.242** | `[-0.011, +0.471]` | 1 / 4 | #14 |
| **7** | **Mitchell Santner** | `4101` | Chennai Super Kings / New Zealand | **+0.167** | `[+0.042, +0.322]` | 1 / 4 | #23 |
| **8** | **Mustafizur Rahman** | `4373` | Delhi Capitals / Bangladesh | **+0.263** | `[-0.063, +0.503]` | 0 / 4 | #12 |
| **9** | **Yuzvendra Chahal** | `1612` | India / Rajasthan Royals | **+0.255** | `[-0.096, +0.580]` | 0 / 4 | #15 |
| **10** | **Lockie Ferguson** | `4692` | Gujarat Titans / New Zealand | **+0.222** | `[-0.110, +0.575]` | 0 / 4 | #18 |

---

## 💎 Strategic Auction Intelligence

High-leverage targets specifically tailored for purse-efficient IPL auction bidding:

1. **Uncapped Indian Gems**:
   - **Ashutosh Sharma** (`ID: 8101`): #1 overall batter (+0.384 r/b, $+0.0290$ WPA/ball, 18/18 grid consensus).
   - **Harpreet Brar** (`ID: 7325`): Premier orthodox run-constrictor (+0.279 r/b, $+0.0120$ WPA/ball, 18/18 grid consensus).
   - **Mohsin Khan** (`ID: 6669`): Left-arm pace and yorker specialist (+0.396 r/b rate leader).
   - **Shashank Singh** (`ID: 2756`): Elite boundary striker in steep required run-rate turnarounds.
   - **Rahul Tewatia** (`ID: 3926`): Proven chase finisher across 18 high-pressure matches.

2. **Overseas Value Specialists**:
   - **Taskin Ahmed** (`ID: 2681`) & **Jacob Duffy** (`ID: 22920`): 4/4 screen pass pace spearheads.
   - **Obed McCoy** (`ID: 4837`): Rajasthan Royals death cutter specialist (4/4 pass).
   - **Kamindu Mendis** (`ID: 4904`): Ambidextrous spin/batter, ranked #1 in restricted benchmark.
   - **Mitchell Santner** (`ID: 4101`): Elite finger-spin economy with highest WPA/ball ($+0.0202$).

---

## 📂 Repository Structure

- `ratings_batters_final3.csv` & `ratings_bowlers_final3.csv`: Final ranked rosters with player IDs.
- `dashboard.html`: Interactive situational pressure performance dashboard.
- `METHODOLOGY_REPORT.pdf`: Publication-ready methodology documentation.
- `METHODOLOGY_REPORT.md` & `METHODOLOGY_REPORT.html`: Source report markdown and HTML versions.
- `pipeline_v4.py`, `final_selection.py`, `final_verify.py`: Modeling and validation pipeline.
