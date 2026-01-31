import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 数据加载与预处理
# ==========================================
# 依然使用 Q3 生成的包含 fan votes 的增强数据
INPUT_FILE = 'all_seasons_final_prediction_with_bounds' 
OUTPUT_FILE = 'new_system_proposal_analysis.csv'

def load_data():
    try:
        df = pd.read_csv(INPUT_FILE)
        # 确保数据中有 'Week' 和 'Judge_Score' 等关键列
        return df
    except:
        print("错误：请先运行 Q3 代码生成 feature_augmented_fan_votes_with_accuracy.csv")
        return None

# ==========================================
# 2. 定义赛制模拟器 (System Simulators)
# ==========================================

def get_rank_metrics(sub_df):
    """辅助：计算排名制的中间结果"""
    j_ranks = stats.rankdata(-sub_df['Judge_Score'].values, method='min')
    f_ranks = stats.rankdata(-sub_df['Est_Fan_Vote'].values, method='min')
    return j_ranks + f_ranks, j_ranks

def get_percent_metrics(sub_df):
    """辅助：计算百分比制的中间结果"""
    j = sub_df['Judge_Score'].values
    j_pct = j / j.sum() if j.sum() > 0 else np.zeros_like(j)
    total = j_pct + sub_df['Est_Fan_Vote'].values
    # 注意：百分比制是分数越高越好，所以 total 越小越差
    # 为了统一逻辑，我们返回 -total，这样 max(-total) 就是 worst
    return -total

def simulate_elimination(group, system_type):
    """
    模拟单周淘汰结果
    返回: (被淘汰者姓名, 被淘汰者的评委排名, 幸存者中'低分高票'的人数)
    """
    names = group['Contestant'].values
    j_scores = group['Judge_Score'].values
    f_votes = group['Est_Fan_Vote'].values
    
    # 1. 计算总分 (Score) - 统一为“分值越高越危险”
    if system_type == 'Rank':
        scores, j_ranks = get_rank_metrics(group)
    elif system_type == 'Percent':
        scores = get_percent_metrics(group) # 负总分，越大(即总分越小)越危险
        j_ranks = stats.rankdata(-j_scores, method='min')
    elif 'Save' in system_type:
        # Save 机制基于 Rank
        scores, j_ranks = get_rank_metrics(group)
    
    # 2. 确定初始淘汰/危险名单
    if 'Save' in system_type:
        n_bottom = 3 if 'Bottom3' in system_type else 2
        # 找出分数最高的 n_bottom 个人 (Rank Sum 最大)
        # argsort 从小到大，取最后 n 个
        sorted_idx = np.argsort(scores) 
        bottom_indices = sorted_idx[-n_bottom:]
        
        # 评委拯救：在 Bottom N 中，淘汰评委分最低者
        # 即 Judge Rank 最大者 (Rank 1 is best)
        # 注意：这里需要重新获取 bottom 组内的相对评委分，或者直接用原始分
        b_j_scores = j_scores[bottom_indices]
        # 淘汰分最低的 (argmin)
        elim_local_idx = np.argmin(b_j_scores)
        elim_idx = bottom_indices[elim_local_idx]
    else:
        # 直接淘汰分数最高的
        elim_idx = np.argmax(scores)
        
    loser_name = names[elim_idx]
    loser_j_rank = j_ranks[elim_idx]
    
    # 3. 计算“兴奋度” (Excitement Metric)
    # 定义：幸存者中，有多少人是“评委不喜欢(Rank后50%)”但“粉丝救回来”的？
    # 这是一个衡量“民意逆袭”的指标
    survivor_mask = np.ones(len(names), dtype=bool)
    survivor_mask[elim_idx] = False
    
    n_contestants = len(names)
    median_rank = n_contestants / 2
    
    # 幸存者索引
    survivor_indices = np.where(survivor_mask)[0]
    
    # 评委排名差（>中位数）且 粉丝排名好（<中位数）
    # j_ranks 是 1=Best. so > median is bad.
    # f_ranks (需重新计算)
    f_ranks = stats.rankdata(-f_votes, method='min')
    
    excitement_count = 0
    for idx in survivor_indices:
        if j_ranks[idx] > median_rank and f_ranks[idx] <= median_rank:
            excitement_count += 1
            
    return loser_name, loser_j_rank, excitement_count

# ==========================================
# 3. 核心评估循环
# ==========================================
def run_comprehensive_evaluation(df):
    systems = ['Rank', 'Percent', 'Rank+Save(B2)', 'Rank+Save(B3)']
    results = {sys: {'C1_Fairness': [], 'C2_Safety': [], 'C3_Excitement': []} for sys in systems}
    
    grouped = df.groupby(['Season', 'Week'])
    
    for (s, w), group in grouped:
        if len(group) < 4: continue 
        
        # 理想最差者 (评委分最低)
        min_j = group['Judge_Score'].min()
        ideal_losers = group[group['Judge_Score'] == min_j]['Contestant'].values
        
        for sys in systems:
            loser, loser_j_rank, excite_cnt = simulate_elimination(group, sys)
            
            # C1: Fairness (是否淘汰了评委认为最差的人？)
            # 1 = Yes (Good), 0 = No
            c1 = 1 if loser in ideal_losers else 0
            
            # C2: Safety (防御极端偏差)
            # 借用 Bobby Bones 案例 (S27)
            # 如果本周有 Bobby Bones 且他没被淘汰，记为 0 (Unsafe)，否则 1
            c2 = 1
            if s == 27 and 'Bobby Bones' in group['Contestant'].values:
                if loser != 'Bobby Bones':
                    c2 = 0 # 危险！让水王存活了
            
            # C3: Excitement (有多少逆袭发生？)
            # 归一化：除以幸存者总数
            c3 = excite_cnt / (len(group) - 1)
            
            results[sys]['C1_Fairness'].append(c1)
            results[sys]['C2_Safety'].append(c2)
            results[sys]['C3_Excitement'].append(c3)
            
    # 汇总
    summary = []
    for sys in systems:
        row = {'System': sys}
        row['Fairness'] = np.mean(results[sys]['C1_Fairness'])
        row['Safety'] = np.mean(results[sys]['C2_Safety'])
        row['Excitement'] = np.mean(results[sys]['C3_Excitement'])
        summary.append(row)
        
    return pd.DataFrame(summary)

# ==========================================
# 4. TOPSIS 多情景分析
# ==========================================
def run_scenario_topsis(df_metrics):
    data = df_metrics.set_index('System')
    mat = data[['Fairness', 'Safety', 'Excitement']].values
    
    # 归一化
    denom = np.sqrt(np.sum(mat**2, axis=0))
    norm_mat = mat / denom
    
    # 定义两种情景的权重
    # Scenario 1: "Professional Competition" (重公平)
    w1 = np.array([0.6, 0.3, 0.1]) 
    
    # Scenario 2: "TV Entertainment" (重娱乐，但也到底线)
    w2 = np.array([0.3, 0.3, 0.4])
    
    # 计算 TOPSIS
    def calc_score(weights):
        w_mat = norm_mat * weights
        z_plus = np.max(w_mat, axis=0)
        z_minus = np.min(w_mat, axis=0)
        d_plus = np.sqrt(np.sum((w_mat - z_plus)**2, axis=1))
        d_minus = np.sqrt(np.sum((w_mat - z_minus)**2, axis=1))
        return d_minus / (d_plus + d_minus)
    
    data['Score_Pro'] = calc_score(w1)
    data['Score_Fun'] = calc_score(w2)
    data['Avg_Score'] = (data['Score_Pro'] + data['Score_Fun']) / 2
    
    return data.sort_values('Avg_Score', ascending=False).reset_index()

# ==========================================
# 5. 执行与输出
# ==========================================
df_raw = load_data()
if df_raw is not None:
    print("Running simulation...")
    metrics_df = run_comprehensive_evaluation(df_raw)
    
    print("Running TOPSIS analysis...")
    final_res = run_scenario_topsis(metrics_df)
    
    print("\n=== Final Proposal Analysis ===")
    print(final_res.round(3).to_string())
    
    final_res.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")
    
    # 打印给论文的结论
    best_sys = final_res.iloc[0]
    print(f"\n[Conclusion for Paper]")
    print(f"We recommend: {best_sys['System']}")
    print(f"Reasoning: It achieves the highest robustness.")
    print(f"- In a Professional Context (Fairness-focused), it scores {best_sys['Score_Pro']:.3f}.")
    print(f"- In an Entertainment Context (Excitement-focused), it scores {best_sys['Score_Fun']:.3f}.")
    print(f"- It maintains an Excitement Level of {best_sys['Excitement']:.1%}, meaning fan votes still heavily influence the 'Safe Zone'.")