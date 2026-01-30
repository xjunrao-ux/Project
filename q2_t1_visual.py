import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'

INPUT_FILE = 'mechanism_comparison_results.csv'

def load_data():
    if not os.path.exists(INPUT_FILE): return None
    return pd.read_csv(INPUT_FILE)

# ==========================================
# 1. 争议人物生存路径 (Trajectory)
# ==========================================
def plot_controversy_path(df, contestant_name, season_id):
    """
    展示某位选手的排名/得分在不同机制下的表现
    """
    sub = df[(df['Contestant'] == contestant_name) & (df['Season'] == season_id)].copy()
    if sub.empty: return

    plt.figure(figsize=(10, 6))
    
    # 绘制评委分 (左轴)
    ax1 = plt.gca()
    ax1.plot(sub['Week'], sub['Judge_Score'], 'b-o', label='Judge Score', alpha=0.5)
    ax1.set_ylabel('Judge Score', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 标记不同机制下的淘汰点
    # 如果在该机制下被淘汰，画一个 X
    for mech, color, marker, y_pos in [
        ('Sim_Status_Rank', 'orange', 'x', 20), 
        ('Sim_Status_Percent', 'green', 'x', 22), 
        ('Sim_Status_Save', 'red', 'X', 24)
    ]:
        elim_weeks = sub[sub[mech] == 'Eliminated']
        if not elim_weeks.empty:
            ax1.scatter(elim_weeks['Week'], [y_pos]*len(elim_weeks), 
                        color=color, s=200, marker=marker, label=f'Eliminated in {mech.split("_")[-1]}')

    plt.title(f'Figure 5: Counterfactual Survival Analysis - {contestant_name} (S{season_id})', fontsize=14, fontweight='bold')
    plt.xlabel('Week')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'fig5_{contestant_name.replace(" ", "_")}_survival.png', dpi=300)
    plt.close()

# ==========================================
# 2. 评委权力指数 (Power Index)
# ==========================================
def plot_power_index(df):
    """
    计算并展示评委分数与最终排名的相关性
    """
    # 简化：计算 "被淘汰者" 的平均评委分排名
    # 如果机制更公平，被淘汰者的评委分应该更低（Rank数值更大）
    
    # 筛选出被淘汰的行
    rank_elim = df[df['Sim_Status_Rank'] == 'Eliminated']
    pct_elim = df[df['Sim_Status_Percent'] == 'Eliminated']
    save_elim = df[df['Sim_Status_Save'] == 'Eliminated']
    
    # 计算这些被淘汰者的 评委得分均值 (越低说明越准)
    avg_score_rank = rank_elim['Judge_Score'].mean()
    avg_score_pct = pct_elim['Judge_Score'].mean()
    avg_score_save = save_elim['Judge_Score'].mean()
    
    # 画图
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Rank Rule', 'Percent Rule', "Judges' Save"], 
                   [avg_score_rank, avg_score_pct, avg_score_save],
                   color=['#f1c40f', '#3498db', '#e74c3c'])
    
    plt.title('Figure 6: Average Judge Score of Eliminated Contestants', fontsize=14, fontweight='bold')
    plt.ylabel('Average Judge Score (Lower is Better for Alignment)', fontsize=12)
    plt.ylim(15, 25) # 调整视窗以便观察差异
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('fig6_judge_power.png', dpi=300)
    plt.close()

# ==========================================
# 主运行
# ==========================================
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print("--- 开始生成 Q2 可视化 ---")
        
        # 1. 争议人物分析
        plot_controversy_path(df, 'Bobby Bones', 27)
        plot_controversy_path(df, 'Jerry Rice', 2)
        
        # 2. 权力指数
        plot_power_index(df)
        
        print("🎉 可视化完成！")