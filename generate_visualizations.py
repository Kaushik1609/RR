import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_charts():
    print("Generating charts for V4 ratings...")
    bat = pd.read_csv('ratings_batters_v4.csv')
    bowl = pd.read_csv('ratings_bowlers_v4.csv')
    
    # 1. Batters Chart V4
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f9fbfd')
    
    scatter = ax.scatter(
        bat['shrunk_level'], 
        bat['shrunk_delta'], 
        s=bat['hp_balls'] * 0.7, 
        c=bat['rating'], 
        cmap='coolwarm', 
        alpha=0.85, 
        edgecolors='#1e293b', 
        linewidth=1.2
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Composite Pressure Rating (Runs / Ball Above Par)', fontsize=11, fontweight='bold', labelpad=10)
    
    top10_bat = bat.head(10)
    for _, row in top10_bat.iterrows():
        ax.annotate(
            f"#{int(row['rank_v4'])} {row['name']}\n(+{row['rating']:.3f} r/b | {int(row['hp_balls'])}b)",
            (row['shrunk_level'], row['shrunk_delta']),
            xytext=(0, 9),
            textcoords='offset points',
            fontsize=8.5,
            fontweight='bold',
            color='#0f172a',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#334155', alpha=0.92, lw=0.8)
        )
        
    ax.set_title('Top T20 Batters Under Pressure (IPL & T20I: 2022–2026: V4)\nBubble Size = HP Chase Balls Faced | Color = Composite Rating (λ = 0.5 in Runs/Ball)', fontsize=12.5, fontweight='bold', pad=15)
    ax.set_xlabel('Shrunk High-Pressure Net Level (Runs Above Par per Ball)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Shrunk Pressure Delta / Clutch Lift (Runs Above Par per Ball)', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
    ax.axhline(0, color='#94a3b8', linestyle=':', lw=1.2)
    ax.axvline(0, color='#94a3b8', linestyle=':', lw=1.2)
    
    plt.tight_layout()
    bat_img = 'top_batters_pressure_matrix.png'
    plt.savefig(bat_img, dpi=300)
    plt.close()
    print(f"Saved: {bat_img}")

    # 2. Bowlers Chart V4
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f9fbfd')
    
    scatter = ax.scatter(
        bowl['shrunk_level'], 
        bowl['shrunk_delta'], 
        s=bowl['hp_balls'] * 0.7, 
        c=bowl['rating'], 
        cmap='viridis', 
        alpha=0.85, 
        edgecolors='#1e293b', 
        linewidth=1.2
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Composite Pressure Rating (Runs / Ball Saved Below Par)', fontsize=11, fontweight='bold', labelpad=10)
    
    top10_bowl = bowl.head(10)
    for _, row in top10_bowl.iterrows():
        ax.annotate(
            f"#{int(row['rank_v4'])} {row['name']}\n(+{row['rating']:.3f} r/b saved | {int(row['hp_balls'])}b)",
            (row['shrunk_level'], row['shrunk_delta']),
            xytext=(0, 9),
            textcoords='offset points',
            fontsize=8.5,
            fontweight='bold',
            color='#0f172a',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffff', edgecolor='#334155', alpha=0.92, lw=0.8)
        )
        
    ax.set_title('Top T20 Bowlers Under Pressure (IPL & T20I: 2022–2026: V4)\nBubble Size = HP Chase Balls Bowled | Color = Composite Rating (λ = 0.5 in Runs/Ball Saved)', fontsize=12.5, fontweight='bold', pad=15)
    ax.set_xlabel('Shrunk High-Pressure Net Level (Runs Saved Below Par per Ball)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Shrunk Pressure Delta / Clutch Lift (Runs Saved Below Par per Ball)', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
    ax.axhline(0, color='#94a3b8', linestyle=':', lw=1.2)
    ax.axvline(0, color='#94a3b8', linestyle=':', lw=1.2)
    
    plt.tight_layout()
    bowl_img = 'top_bowlers_pressure_matrix.png'
    plt.savefig(bowl_img, dpi=300)
    plt.close()
    print(f"Saved: {bowl_img}")

if __name__ == '__main__':
    generate_charts()
