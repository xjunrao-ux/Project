import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 1. Load Data
INPUT_FILE = 'all_seasons_final_prediction_with_bounds.csv'
try:
    df = pd.read_csv(INPUT_FILE)
except:
    print("错误：无法加载输入文件。")
    df = None

# 2. 模拟函数
def get_rank_loser(sub_df):
    """Rank Rule: Max Total Rank"""
    j_ranks = stats.rankdata(-sub_df['Judge_Score'].values, method='min')
    f_ranks = stats.rankdata(-sub_df['Est_Fan_Vote'].values, method='min')
    total = j_ranks + f_ranks
    
    max_val = np.max(total)
    candidates = np.where(total == max_val)[0]
    
    if len(candidates) > 1:
        f_ranks_sub = stats.rankdata(-sub_df['Est_Fan_Vote'].values, method='min')
        worst_f = np.max(f_ranks_sub[candidates])
        idx = candidates[np.where(f_ranks_sub[candidates] == worst_f)[0][0]]
    else:
        idx = candidates[0]
    return sub_df.iloc[idx]['Contestant']

def get_percent_loser(sub_df):
    """Percent Rule: Min Total Percent"""
    j_scores = sub_df['Judge_Score'].values
    j_sum = np.sum(j_scores)
    j_pct = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
    total = j_pct + sub_df['Est_Fan_Vote'].values
    idx = np.argmin(total)
    return sub_df.iloc[idx]['Contestant']

def get_save_loser(sub_df, rule):
    """Judges' Save Logic"""
    # 1. Identify Bottom 2
    if rule == 'RANK':
        j_ranks = stats.rankdata(-sub_df['Judge_Score'].values, method='min')
        f_ranks = stats.rankdata(-sub_df['Est_Fan_Vote'].values, method='min')
        total = j_ranks + f_ranks
        sorted_idx = np.argsort(-total) # Descending (Worst first)
    else:
        j_scores = sub_df['Judge_Score'].values
        j_sum = np.sum(j_scores)
        j_pct = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
        total = j_pct + sub_df['Est_Fan_Vote'].values
        sorted_idx = np.argsort(total) # Ascending (Worst first)
        
    bottom2_idx = sorted_idx[:2]
    
    # 2. Eliminate Lower Judge Score
    idx1, idx2 = bottom2_idx[0], bottom2_idx[1]
    s1 = sub_df.iloc[idx1]['Judge_Score']
    s2 = sub_df.iloc[idx2]['Judge_Score']
    
    if s1 < s2: return sub_df.iloc[idx1]['Contestant']
    elif s1 > s2: return sub_df.iloc[idx2]['Contestant']
    else: return sub_df.iloc[idx1]['Contestant'] # Tie -> Default

# 3. 主分析循环
if df is not None:
    results = []
    
    grouped = df.groupby(['Season', 'Week'])
    
    for (s, w), group in grouped:
        if len(group) < 2: continue
        
        # 理想淘汰者（评委眼中最差）
        min_j = group['Judge_Score'].min()
        ideal_losers = group[group['Judge_Score'] == min_j]['Contestant'].values
        
        # 模拟四种情况
        l_rank = get_rank_loser(group)
        l_pct = get_percent_loser(group)
        l_rank_save = get_save_loser(group, 'RANK')
        l_pct_save = get_save_loser(group, 'PERCENT')
        
        # 获取淘汰者的评委分
        s_rank = group[group['Contestant'] == l_rank]['Judge_Score'].values[0]
        s_pct = group[group['Contestant'] == l_pct]['Judge_Score'].values[0]
        s_rank_save = group[group['Contestant'] == l_rank_save]['Judge_Score'].values[0]
        s_pct_save = group[group['Contestant'] == l_pct_save]['Judge_Score'].values[0]
        
        results.append({
            'Season': s, 'Week': w,
            'Match_Rank': l_rank in ideal_losers,
            'Match_Percent': l_pct in ideal_losers,
            'Match_Rank_Save': l_rank_save in ideal_losers,
            'Match_Percent_Save': l_pct_save in ideal_losers,
            'Score_Rank': s_rank,
            'Score_Percent': s_pct,
            'Score_Rank_Save': s_rank_save,
            'Score_Percent_Save': s_pct_save
        })
        
    res_df = pd.DataFrame(results)
    
    # 汇总统计
    summary = pd.DataFrame({
        'Method': ['Rank Rule', 'Percent Rule', 'Rank + Save', 'Percent + Save'],
        'Judge Alignment (%)': [
            res_df['Match_Rank'].mean() * 100,
            res_df['Match_Percent'].mean() * 100,
            res_df['Match_Rank_Save'].mean() * 100,
            res_df['Match_Percent_Save'].mean() * 100
        ],
        'Avg Loser Score': [
            res_df['Score_Rank'].mean(),
            res_df['Score_Percent'].mean(),
            res_df['Score_Rank_Save'].mean(),
            res_df['Score_Percent_Save'].mean()
        ]
    })
    
    summary['Judge Alignment (%)'] = summary['Judge Alignment (%)'].round(1)
    summary['Avg Loser Score'] = summary['Avg Loser Score'].round(2)
    
    summary.to_csv('method_recommendation_analysis.csv', index=False)
    print("\n=== 最终推荐分析表 ===")
    print(summary.to_string(index=False))