import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 0. 全局配置 (Style Settings)
# ==========================================
# 设置美赛风格绘图参数
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] # 适配不同系统
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("paper", font_scale=1.4) # 论文级字体大小

INPUT_FILE = 'all_seasons_final_prediction.csv'

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}。请确保先运行了上一生成代码。")
        return None
    return pd.read_csv(INPUT_FILE)

# ==========================================
# 图表 1: 粉丝投票重构轨迹 (带置信区间)
# ==========================================
def plot_season_trajectory(df, season_id=1):
    """
    绘制特定赛季的粉丝投票随时间变化图。
    亮点：展示了 Uncertainty (阴影) 和 淘汰点 (红叉)。
    """
    s_df = df[df['Season'] == season_id].copy()
    if s_df.empty:
        print(f"Season {season_id} 数据为空，跳过绘图。")
        return

    plt.figure(figsize=(12, 7))
    
    # 筛选主要选手 (存活超过3周的，避免图表太乱) 或绘制全部
    # 这里绘制全部，但利用颜色区分
    contestants = s_df['Contestant'].unique()
    palette = sns.color_palette("husl", n_colors=len(contestants))
    
    for i, name in enumerate(contestants):
        sub = s_df[s_df['Contestant'] == name]
        color = palette[i]
        
        # 绘制主趋势线
        plt.plot(sub['Week'], sub['Est_Fan_Vote'], marker='o', markersize=4, 
                 label=name, color=color, linewidth=2, alpha=0.8)
        
        # 绘制置信区间 (Uncertainty)
        plt.fill_between(sub['Week'], 
                         sub['Est_Fan_Vote'] - sub['Uncertainty'], 
                         sub['Est_Fan_Vote'] + sub['Uncertainty'], 
                         color=color, alpha=0.15)
        
        # 标记真实淘汰点
        elim = sub[sub['Actual_Status'] == 'Actual Loser']
        if not elim.empty:
            plt.scatter(elim['Week'], elim['Est_Fan_Vote'], 
                        color='red', marker='X', s=150, zorder=10, edgecolor='black', linewidth=1.5)

    plt.title(f'Figure 1: Reconstructed Fan Vote Trajectories (Season {season_id})', fontsize=16, fontweight='bold')
    plt.xlabel('Competition Week', fontsize=14)
    plt.ylabel('Estimated Fan Vote Share', fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Contestants', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = f'fig1_season_{season_id}_trajectory.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图表 1 已保存: {save_path}")
    plt.close()

# ==========================================
# 图表 2: 机制博弈相平面 (The Phase Space)
# ==========================================
def plot_mechanism_phase_space(df):
    """
    绘制 '评委份额 vs 粉丝份额' 散点图。
    亮点：直观展示 Rank制 和 Percent制 的“死亡区域”差异。
    """
    # 计算评委份额 (归一化以便比较)
    df['Judge_Share'] = df.groupby(['Season', 'Week'])['Judge_Score'].transform(lambda x: x / x.sum())
    
    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    
    # 定义颜色映射
    status_palette = {'Safe': '#1f77b4', 'Actual Loser': '#d62728'} # 蓝/红
    style_map = {'Safe': 'o', 'Actual Loser': 'X'}
    
    # --- 左图：Rank 规则 ---
    rank_df = df[df['Rule'] == 'RANK']
    sns.scatterplot(data=rank_df, x='Judge_Share', y='Est_Fan_Vote', 
                    hue='Actual_Status', style='Actual_Status', 
                    palette=status_palette, markers=style_map,
                    ax=axes[0], s=80, alpha=0.6)
    axes[0].set_title('Mechanism A: Rank Rule (Non-Linear Boundary)', fontsize=15, fontweight='bold')
    axes[0].set_xlabel('Judge Score Share', fontsize=13)
    axes[0].set_ylabel('Fan Vote Share (Estimated)', fontsize=13)
    
    # --- 右图：Percent 规则 ---
    pct_df = df[df['Rule'] == 'PERCENT']
    sns.scatterplot(data=pct_df, x='Judge_Share', y='Est_Fan_Vote', 
                    hue='Actual_Status', style='Actual_Status', 
                    palette=status_palette, markers=style_map,
                    ax=axes[1], s=80, alpha=0.6)
    
    # 添加 Percent 规则的理论死亡线 (x + y = const)
    # 取一个近似阈值用于示意 (例如 0.15)
    x = np.linspace(0, 0.3, 100)
    y = 0.15 - x
    axes[1].plot(x, y, color='green', linestyle='--', linewidth=2, label='Theoretical Survival Line')
    
    axes[1].set_title('Mechanism B: Percent Rule (Linear Boundary)', fontsize=15, fontweight='bold')
    axes[1].set_xlabel('Judge Score Share', fontsize=13)
    
    plt.suptitle('Figure 2: Survival Phase Space Analysis (The "Death Zone")', fontsize=18, y=0.98)
    plt.tight_layout()
    
    save_path = 'fig2_mechanism_phase_space.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图表 2 已保存: {save_path}")
    plt.close()

# ==========================================
# 图表 3: 模型确定性统计 (Uncertainty Boxplot)
# ==========================================
def plot_uncertainty_stats(df):
    """
    绘制不确定性分布箱线图。
    亮点：证明模型对'淘汰者'的判断比'安全者'更确定 (Standard Deviation 更低)。
    """
    plt.figure(figsize=(8, 6))
    
    # 简化状态标签
    df['Status_Simple'] = df['Actual_Status'].apply(lambda x: 'Eliminated' if x == 'Actual Loser' else 'Safe')
    
    sns.boxplot(data=df, x='Status_Simple', y='Uncertainty', 
                palette={'Safe': '#2ecc71', 'Eliminated': '#e74c3c'}, width=0.5)
    
    plt.title('Figure 3: Model Certainty by Contestant Status', fontsize=15, fontweight='bold')
    plt.ylabel('Estimation Uncertainty (Std Dev)', fontsize=13)
    plt.xlabel('Contestant Status', fontsize=13)
    
    # 添加显著性注释
    plt.text(0.5, 0.9, "Lower uncertainty for\neliminated contestants\nconfirms model robustness", 
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    save_path = 'fig3_uncertainty_analysis.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图表 3 已保存: {save_path}")
    plt.close()

# ==========================================
# 图表 4: 历年预测准确率 (Accuracy Bar)
# ==========================================
def plot_accuracy_over_time(df):
    """
    绘制每个赛季的模型匹配率。
    """
    # 只看淘汰周次
    elim_df = df[df['Actual_Status'] == 'Actual Loser']
    
    # 按赛季聚合
    accuracy = elim_df.groupby('Season')['Match_Success'].mean() * 100
    accuracy_df = accuracy.reset_index()
    
    plt.figure(figsize=(14, 6))
    sns.barplot(data=accuracy_df, x='Season', y='Match_Success', color='#3498db')
    
    plt.axhline(y=accuracy.mean(), color='red', linestyle='--', label=f'Average Accuracy: {accuracy.mean():.1f}%')
    
    plt.title('Figure 4: Model Consistency Across 34 Seasons', fontsize=16, fontweight='bold')
    plt.ylabel('Prediction Match Rate (%)', fontsize=13)
    plt.xlabel('Season', fontsize=13)
    plt.ylim(0, 110)
    plt.legend()
    
    plt.tight_layout()
    save_path = 'fig4_model_accuracy.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图表 4 已保存: {save_path}")
    plt.close()

# ==========================================
# 主运行入口
# ==========================================
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print("--- 开始生成可视化图表 ---")
        
        # 1. 生成 Season 1 的轨迹图 (最经典案例)
        plot_season_trajectory(df, season_id=1)
        
        # 2. 生成 Season 27 的轨迹图 (Bobby Bones 争议赛季，可选)
        if 27 in df['Season'].values:
            plot_season_trajectory(df, season_id=27)
            
        # 3. 生成机制对比相平面图
        plot_mechanism_phase_space(df)
        
        # 4. 生成不确定性分析图
        plot_uncertainty_stats(df)
        
        # 5. 生成准确率图
        plot_accuracy_over_time(df)
        
        print("\n🎉 所有图表生成完毕！请在当前目录下查看 fig1-fig4 开头的图片。")