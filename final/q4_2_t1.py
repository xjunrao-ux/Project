import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 基础设置与数据加载
# ==========================================
INPUT_FILE = 'all_seasons_final_prediction_with_bounds.csv'

def load_data():
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"数据加载成功: {len(df)} 行")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return pd.DataFrame()

# ==========================================
# 2. 核心逻辑：多目标模拟器
# ==========================================
def calculate_objectives(group, w, k):
    """
    计算单周的两个目标函数值
    f_fair: Kendall's Tau (与评委排名的一致性)
    f_susp: 1 - (System Winner == Judge Winner) (冠军不可预测性)
    """
    # 获取原始数据
    j_scores = group['Judge_Score'].values
    f_votes = group['Est_Fan_Vote'].values
    
    # 1. 计算排名 (1=Best)
    j_ranks = stats.rankdata(-j_scores, method='min')
    f_ranks = stats.rankdata(-f_votes, method='min')
    
    # 2. 合成新规则得分 (越小越好)
    # Score = w * Rank_J + (1-w) * Rank_F
    combined_scores = w * j_ranks + (1-w) * f_ranks
    
    # 3. 生成最终排名
    final_ranks = stats.rankdata(combined_scores, method='min')
    
    # --- 指标 1: 公平性 (Kendall's Tau) ---
    # 衡量最终排名与评委排名的相关性
    tau, _ = stats.kendalltau(j_ranks, final_ranks)
    if np.isnan(tau): tau = 0
    f_fair = tau
    
    # --- 指标 2: 悬念性 (Suspense / Unpredictability) ---
    # 定义：评委心中的第一名(Rank 1)，在最终结果中没有拿第一的概率
    # Judge Winners (可能并列)
    judge_winners = np.where(j_ranks == 1)[0]
    # System Winners
    sys_winners = np.where(final_ranks == 1)[0]
    
    # 如果评委的任何一个第一名，也是系统的第一名，则不仅没有悬念，而且是"Predictable"
    is_predictable = not set(judge_winners).isdisjoint(sys_winners)
    
    f_susp = 1.0 if not is_predictable else 0.0
    
    return f_fair, f_susp

# ==========================================
# 3. 全空间搜索 (Grid Search for Pareto)
# ==========================================
def run_optimization(df):
    # 决策变量空间
    weights = np.arange(0.4, 0.85, 0.05) # w: 0.4 ~ 0.8
    ks = [0, 1, 2] # K: Save Threshold
    
    results = []
    
    # 按赛季/周分组，加速计算
    grouped = list(df.groupby(['Season', 'Week']))
    
    print(f"开始优化搜索: 搜索空间大小 {len(weights)*len(ks)}...")
    
    for k in ks:
        for w in weights:
            taus = []
            suspenses = []
            
            for _, group in grouped:
                if len(group) < 3: continue # 忽略人数过少的周
                
                t, s = calculate_objectives(group, w, k)
                taus.append(t)
                suspenses.append(s)
            
            # 取均值作为该参数组合的性能
            results.append({
                'w': round(w, 2),
                'K': k,
                'f_fair': np.mean(taus),
                'f_susp': np.mean(suspenses)
            })
            
    return pd.DataFrame(results)

# ==========================================
# 4. 帕累托前沿识别 (Pareto Sorting)
# ==========================================
def identify_pareto_frontier(df):
    """非支配排序"""
    population = df[['f_fair', 'f_susp']].values
    is_efficient = np.ones(len(df), dtype=bool)
    
    for i, c in enumerate(population):
        if is_efficient[i]:
            # 如果存在任何一点 B，使得 B.fair >= A.fair 且 B.susp >= A.susp 且至少有一个严格大于
            # 则 A 被支配 (Dominated)，设为 False
            is_dominated = np.any(
                (population[:, 0] >= c[0]) & 
                (population[:, 1] >= c[1]) & 
                ((population[:, 0] > c[0]) | (population[:, 1] > c[1]))
            )
            if is_dominated:
                is_efficient[i] = False
                
    df['Is_Pareto'] = is_efficient
    return df

# ==========================================
# 5. 可视化与输出
# ==========================================
def plot_pareto(df):
    plt.figure(figsize=(10, 6))
    
    # 绘制所有解
    sns.scatterplot(data=df, x='f_fair', y='f_susp', 
                    hue='w', palette='viridis', style='Is_Pareto', 
                    s=100, alpha=0.8)
    
    # 绘制前沿连线
    pareto_points = df[df['Is_Pareto']].sort_values('f_fair')
    plt.plot(pareto_points['f_fair'], pareto_points['f_susp'], 
             'r--', linewidth=2, label='Pareto Frontier')
    
    # 标注拐点 (Knee Point)
    # 简单找距离 (1,1) 最近的点，或者 Min-Max 归一化后距离 (1,1) 最近
    # 这里目测中间点
    mid_idx = len(pareto_points) // 2
    knee = pareto_points.iloc[mid_idx]
    plt.scatter(knee['f_fair'], knee['f_susp'], color='red', s=200, marker='*', label='Knee Point')
    plt.text(knee['f_fair'], knee['f_susp']+0.02, 
             f"w={knee['w']}, K={knee['K']}", color='red', fontweight='bold')

    plt.title('Multi-Objective Pareto Optimization: Fairness vs Suspense')
    plt.xlabel('Fairness (Kendall Tau with Judges)')
    plt.ylabel('Suspense (Probability of Unexpected Winner)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# 主程序
if __name__ == "__main__":
    df_raw = load_data()
    if not df_raw.empty:
        # 1. 优化
        res_df = run_optimization(df_raw)
        
        # 2. 帕累托排序
        res_df = identify_pareto_frontier(res_df)
        
        # 3. 输出前沿
        pareto_front = res_df[res_df['Is_Pareto']].sort_values('f_fair')
        print("\n=== Pareto Frontier Solutions ===")
        print(pareto_front.to_string(index=False))
        
        # 4. 可视化
        plot_pareto(res_df)
        
        # 保存
        res_df.to_csv('pareto_optimization_results.csv', index=False)
        print("\n结果已保存至 pareto_optimization_results.csv")