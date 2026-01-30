import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 加载数据
# ==========================================
INPUT_FILE = 'all_seasons_final_prediction_with_bounds.csv'
OUTPUT_FILE = 'mechanism_comparison_results.csv'

def load_data():
    try:
        return pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("错误：未找到输入文件。请先运行上一问代码生成 csv。")
        return None

# ==========================================
# 2. 定义评分规则算法
# ==========================================
def calculate_rank_loser(df_week):
    """
    Rank Rule: Loser is the one with MAX Total Rank.
    Tie-breaker: One with worse Fan Rank (larger value, fewer votes) loses.
    """
    scores = df_week['Judge_Score'].values
    votes = df_week['Est_Fan_Vote'].values
    names = df_week['Contestant'].values
    
    # rankdata: 'min' method assigns 1 to lowest value. 
    # We want 1 for HIGHEST score/vote. So we rank negative values.
    j_ranks = stats.rankdata(-scores, method='min')
    f_ranks = stats.rankdata(-votes, method='min')
    
    total = j_ranks + f_ranks
    
    # Identify worst
    max_val = np.max(total)
    candidates = np.where(total == max_val)[0]
    
    if len(candidates) == 1:
        return names[candidates[0]], candidates[0]
    else:
        # Tie-breaker: Worse Fan Rank (Larger number) loses
        sub_f = f_ranks[candidates]
        worst_f = np.max(sub_f)
        # Check if tie again
        final_cands = candidates[np.where(sub_f == worst_f)[0]]
        # If still tie, return first (rare)
        return names[final_cands[0]], final_cands[0]

def calculate_percent_loser(df_week):
    """
    Percent Rule: Loser is the one with MIN Total Percent.
    Tie-breaker: Assume Judge Score prevails (or strictly min).
    """
    scores = df_week['Judge_Score'].values
    votes = df_week['Est_Fan_Vote'].values
    names = df_week['Contestant'].values
    
    j_sum = np.sum(scores)
    j_pct = scores / j_sum if j_sum > 0 else np.zeros_like(scores)
    # Fan votes are already shares (sum to 1 approx)
    
    total = j_pct + votes
    
    # Identify worst
    min_val = np.min(total)
    candidates = np.where(total == min_val)[0]
    
    # Tie-breaker: Lower Judge Score loses? Or random. 
    # Let's use Judge Score as tie breaker for Percent rule (Professionalism)
    if len(candidates) > 1:
        sub_j = scores[candidates]
        worst_j = np.min(sub_j)
        final_cands = candidates[np.where(sub_j == worst_j)[0]]
        return names[final_cands[0]], final_cands[0]
    else:
        return names[candidates[0]], candidates[0]

def calculate_judges_save_loser(df_week, current_rule):
    """
    Judges' Save: 
    1. Identify Bottom 2 based on Current Rule.
    2. Eliminate the one with LOWER Judge Score.
    """
    scores = df_week['Judge_Score'].values
    votes = df_week['Est_Fan_Vote'].values
    names = df_week['Contestant'].values
    
    # 1. Determine Bottom 2
    if current_rule == 'RANK':
        j_ranks = stats.rankdata(-scores, method='min')
        f_ranks = stats.rankdata(-votes, method='min')
        total = j_ranks + f_ranks
        # Sort descending (Higher rank num is worse)
        sorted_idx = np.argsort(-total) # Worst first
        bottom_2_idx = sorted_idx[:2]
    else:
        j_sum = np.sum(scores)
        j_pct = scores / j_sum if j_sum > 0 else np.zeros_like(scores)
        total = j_pct + votes
        # Sort ascending (Lower pct is worse)
        sorted_idx = np.argsort(total) # Worst first
        bottom_2_idx = sorted_idx[:2]
        
    # 2. Judges Decide
    idx1, idx2 = bottom_2_idx[0], bottom_2_idx[1]
    
    if scores[idx1] < scores[idx2]:
        return names[idx1] # idx1 eliminated (lower judge score)
    elif scores[idx1] > scores[idx2]:
        return names[idx2] # idx2 eliminated
    else:
        # If Judge Scores tied, revert to Fan Vote (Original Rule)
        # Who was the original loser? The one ranked worst (idx1)
        return names[idx1]

# ==========================================
# 3. 仿真主程序
# ==========================================
def run_simulation():
    df = load_data()
    if df is None: return

    results = []
    
    # Iterate over Season/Week
    # Only consider weeks where elimination is possible (skip weeks with < 2 people)
    grouped = df.groupby(['Season', 'Week'])
    
    for (s, w), group in grouped:
        if len(group) < 2: continue
        
        # Check actual status (Ground Truth)
        # Is there an 'Actual Loser' this week?
        # Note: In our previous step, we marked Actual Loser. 
        # If Week 1 had no elimination, 'Actual_Status' is all Safe.
        actual_loser_row = group[group['Actual_Status'].str.contains('Loser', case=False, na=False)]
        actual_loser = actual_loser_row['Contestant'].iloc[0] if not actual_loser_row.empty else 'None'
        
        current_rule = group['Rule'].iloc[0]
        
        # 1. Run Rank Sim
        rank_loser, _ = calculate_rank_loser(group)
        
        # 2. Run Percent Sim
        pct_loser, _ = calculate_percent_loser(group)
        
        # 3. Run Save Sim
        save_loser = calculate_judges_save_loser(group, current_rule)
        
        # 4. Record Results for every contestant
        for idx, row in group.iterrows():
            name = row['Contestant']
            
            # Determine simulated status
            sim_status_rank = 'Eliminated' if name == rank_loser else 'Safe'
            sim_status_pct = 'Eliminated' if name == pct_loser else 'Safe'
            sim_status_save = 'Eliminated' if name == save_loser else 'Safe'
            
            # Flip Check
            # Is Rank outcome different from Percent outcome?
            is_mech_flip = (rank_loser != pct_loser)
            
            # Save Check
            # Did Save change the outcome compared to the Current Rule?
            if current_rule == 'RANK':
                is_save_flip = (save_loser != rank_loser)
            else:
                is_save_flip = (save_loser != pct_loser)

            results.append({
                'Season': s,
                'Week': w,
                'Contestant': name,
                'Judge_Score': row['Judge_Score'],
                'Est_Fan_Vote': row['Est_Fan_Vote'],
                'Actual_Status': row['Actual_Status'],
                'Original_Rule': current_rule,
                'Sim_Status_Rank': sim_status_rank,
                'Sim_Status_Percent': sim_status_pct,
                'Sim_Status_Save': sim_status_save,
                'Diff_Rank_vs_Pct': is_mech_flip,
                'Diff_Save_vs_Orig': is_save_flip
            })

    # Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 机制对比完成！结果已保存至: {OUTPUT_FILE}")
    
    # --- 简要分析打印 ---
    # 1. 翻转率统计
    # Create a per-week summary (drop duplicates)
    week_summary = res_df[['Season', 'Week', 'Diff_Rank_vs_Pct', 'Diff_Save_vs_Orig']].drop_duplicates()
    total_weeks = len(week_summary)
    rank_pct_flips = week_summary['Diff_Rank_vs_Pct'].sum()
    save_flips = week_summary['Diff_Save_vs_Orig'].sum()
    
    print("\n=== 核心指标摘要 ===")
    print(f"总分析周数: {total_weeks}")
    print(f"机制翻转次数 (Rank vs Percent): {rank_pct_flips} ({rank_pct_flips/total_weeks*100:.1f}%)")
    print(f"评委拯救触发次数 (Save vs Original): {save_flips} ({save_flips/total_weeks*100:.1f}%)")
    
    # 2. 争议人物检查
    targets = ['Jerry Rice', 'Billy Ray Cyrus', 'Bristol Palin', 'Bobby Bones']
    print("\n=== 争议人物反事实分析 (是否被提前淘汰?) ===")
    for t in targets:
        # Check if they were ever eliminated in simulation but safe in reality
        # Filter for rows where they are the contestant
        t_df = res_df[res_df['Contestant'] == t]
        if t_df.empty: continue
        
        # Look for weeks where Actual=Safe but Sim=Eliminated
        danger_weeks = t_df[
            (t_df['Actual_Status'] == 'Safe') & 
            ((t_df['Sim_Status_Rank'] == 'Eliminated') | 
             (t_df['Sim_Status_Percent'] == 'Eliminated') | 
             (t_df['Sim_Status_Save'] == 'Eliminated'))
        ]
        
        if not danger_weeks.empty:
            print(f"\n>> {t} 险些被淘汰的周次:")
            print(danger_weeks[['Season', 'Week', 'Judge_Score', 'Original_Rule', 'Sim_Status_Rank', 'Sim_Status_Percent', 'Sim_Status_Save']].head(5).to_string(index=False))
        else:
            print(f"\n>> {t}: 极其坚挺，没有任何机制能淘汰他/她。")

if __name__ == "__main__":
    run_simulation()