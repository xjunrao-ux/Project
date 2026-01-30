import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 配置与数据加载
# ==========================================
INPUT_FILE = 'all_seasons_final_prediction_with_bounds.csv'
OUTPUT_FILE = 'mechanism_detailed_comparison.csv'

def load_data():
    try:
        return pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"错误：未找到 {INPUT_FILE}，请确保已运行上一问代码。")
        return None

# ==========================================
# 核心算法：双轨制结算
# ==========================================
def calculate_rank_metrics(j_scores, f_votes):
    """
    Rank Algorithm:
    - Score: Rank(Judge) + Rank(Fan)
    - Ranking: Lower total is better.
    - Loser: Max total (Tie-breaker: Worst Fan Rank)
    """
    # rankdata 'min': smallest val gets rank 1. 
    # We want Highest Score -> Rank 1. So rank -scores.
    j_ranks = stats.rankdata(-j_scores, method='min')
    f_ranks = stats.rankdata(-f_votes, method='min')
    
    total_score = j_ranks + f_ranks
    
    # Who is worst? (Max value)
    max_val = np.max(total_score)
    candidates = np.where(total_score == max_val)[0]
    
    if len(candidates) > 1:
        # Tie-breaker: Person with worse Fan Rank (Larger number) loses
        sub_f = f_ranks[candidates]
        worst_f = np.max(sub_f)
        loser_idx = candidates[np.where(sub_f == worst_f)[0][0]]
    else:
        loser_idx = candidates[0]
        
    # Calculate Final Rank for this week (1 = Best, N = Worst)
    # total_score small is good.
    final_ranks = stats.rankdata(total_score, method='min')
    
    return total_score, final_ranks, loser_idx

def calculate_pct_metrics(j_scores, f_votes):
    """
    Percentage Algorithm:
    - Score: %Judge + %Fan
    - Ranking: Higher total is better.
    - Loser: Min total
    """
    j_sum = np.sum(j_scores)
    j_pct = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
    
    total_score = j_pct + f_votes
    
    # Who is worst? (Min value)
    loser_idx = np.argmin(total_score)
    
    # Calculate Final Rank (1 = Best, N = Worst)
    # total_score large is good.
    final_ranks = stats.rankdata(-total_score, method='min')
    
    return total_score, final_ranks, loser_idx, j_pct

# ==========================================
# 主程序
# ==========================================
def run_comparison_analysis():
    df = load_data()
    if df is None: return

    detailed_results = []
    flip_count = 0
    total_weeks = 0

    # 按赛季和周次分组
    grouped = df.groupby(['Season', 'Week'])

    for (s, w), group in grouped:
        if len(group) < 2: continue # 跳过决赛或单人周

        names = group['Contestant'].values
        j_scores = group['Judge_Score'].values
        f_votes = group['Est_Fan_Vote'].values
        
        # 1. 计算 Rank 算法指标
        r_scores, r_ranks, r_loser_idx = calculate_rank_metrics(j_scores, f_votes)
        r_loser_name = names[r_loser_idx]
        
        # 2. 计算 Percentage 算法指标
        p_scores, p_ranks, p_loser_idx, j_pcts = calculate_pct_metrics(j_scores, f_votes)
        p_loser_name = names[p_loser_idx]
        
        # 3. 检测反转
        is_flip = (r_loser_name != p_loser_name)
        if is_flip: flip_count += 1
        total_weeks += 1
        
        # 4. 记录详细数据
        for i, name in enumerate(names):
            detailed_results.append({
                'Season': s,
                'Week': w,
                'Contestant': name,
                'Judge_Score': j_scores[i],
                'Est_Fan_Vote': f_votes[i],
                
                # --- Rank Algorithm Output ---
                'Rank_Algo_Score': r_scores[i],       # Lower is better
                'Rank_Algo_Rank': r_ranks[i],         # 1 is Best
                'Rank_Sim_Status': 'Eliminated' if i == r_loser_idx else 'Safe',
                
                # --- Percent Algorithm Output ---
                'Pct_Algo_Score': round(p_scores[i], 4), # Higher is better
                'Pct_Algo_Rank': p_ranks[i],             # 1 is Best
                'Pct_Sim_Status': 'Eliminated' if i == p_loser_idx else 'Safe',
                
                # --- Comparison ---
                'Is_Result_Flip': is_flip
            })

    # 保存表格
    res_df = pd.DataFrame(detailed_results)
    res_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ 分析完成！详细对比表已保存至: {OUTPUT_FILE}")
    print(f"\n=== 结论反转分析 ===")
    print(f"总模拟周数: {total_weeks}")
    print(f"结论反转次数 (Flip Count): {flip_count}")
    print(f"反转率 (Sensitivity): {flip_count/total_weeks*100:.1f}%")
    
    # 打印一个具体的反转案例
    flip_example = res_df[res_df['Is_Result_Flip'] == True].iloc[0]
    s_ex, w_ex = flip_example['Season'], flip_example['Week']
    print(f"\n[反转案例] Season {s_ex} Week {w_ex}:")
    example_df = res_df[(res_df['Season'] == s_ex) & (res_df['Week'] == w_ex)]
    print(example_df[['Contestant', 'Judge_Score', 'Est_Fan_Vote', 'Rank_Algo_Rank', 'Pct_Algo_Rank', 'Rank_Sim_Status', 'Pct_Sim_Status']].to_string(index=False))

if __name__ == "__main__":
    run_comparison_analysis()