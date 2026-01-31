import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 配置与加载
# ==========================================
INPUT_FILE = 'all_seasons_final_prediction_with_bounds.csv'
OUTPUT_FILE = 'judges_save_impact_analysis.csv'

def load_data():
    try:
        df = pd.read_csv(INPUT_FILE)
        return df
    except FileNotFoundError:
        print(f"错误：未找到 {INPUT_FILE}。")
        return None

# ==========================================
# 核心算法：评委拯救 (Judges' Save)
# ==========================================
def determine_bottom_two(j_scores, f_votes, rule):
    """根据当前规则找出倒数两名 (Bottom 2)"""
    n = len(j_scores)
    # Rank制: 数值越大越差; Percent制: 数值越小越差
    if rule == 'RANK':
        j_ranks = stats.rankdata(-j_scores, method='min')
        f_ranks = stats.rankdata(-f_votes, method='min')
        total = j_ranks + f_ranks
        sorted_idx = np.argsort(-total) # Descending (Worst first)
    else:
        j_sum = np.sum(j_scores)
        j_pct = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
        total = j_pct + f_votes
        sorted_idx = np.argsort(total) # Ascending (Worst first)
        
    return sorted_idx[:2]

def apply_judges_save(bottom2_idx, j_scores, names):
    """评委裁决: 淘汰 Bottom 2 中评委分更低者"""
    idx1, idx2 = bottom2_idx[0], bottom2_idx[1]
    
    # idx1 是原始最差，idx2 是原始倒数第二
    if j_scores[idx1] < j_scores[idx2]:
        return names[idx1] # idx1 真差，淘汰
    elif j_scores[idx1] > j_scores[idx2]:
        return names[idx2] # idx1 被救，idx2 淘汰 (翻转!)
    else:
        return names[idx1] # 平局，维持原判

def get_original_loser(j_scores, f_votes, names, rule):
    """获取原始规则淘汰者"""
    if rule == 'RANK':
        j_ranks = stats.rankdata(-j_scores, method='min')
        f_ranks = stats.rankdata(-f_votes, method='min')
        total = j_ranks + f_ranks
        max_val = np.max(total)
        candidates = np.where(total == max_val)[0]
        # Rank Tie-breaker: 粉丝Rank大者(票少)淘汰
        if len(candidates) > 1:
            f_ranks_sub = f_ranks[candidates]
            worst_f = np.max(f_ranks_sub)
            idx = candidates[np.where(f_ranks_sub == worst_f)[0][0]]
        else:
            idx = candidates[0]
    else:
        j_sum = np.sum(j_scores)
        j_pct = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
        total = j_pct + f_votes
        idx = np.argmin(total)
        
    return names[idx]

# ==========================================
# 主程序
# ==========================================
def run_save_impact_analysis():
    df = load_data()
    if df is None: return

    results = []
    
    print("开始模拟评委拯救机制...")
    
    grouped = df.groupby(['Season', 'Week'])
    count_weeks = 0
    count_flips = 0
    
    for (s, w), group in grouped:
        if len(group) < 2: continue
        
        j_scores = group['Judge_Score'].values
        f_votes = group['Est_Fan_Vote'].values
        names = group['Contestant'].values
        curr_rule = group['Rule'].iloc[0]
        
        # 1. 原始结果
        orig_loser = get_original_loser(j_scores, f_votes, names, curr_rule)
        
        # 2. 评委拯救结果
        b2_idx = determine_bottom_two(j_scores, f_votes, curr_rule)
        save_loser = apply_judges_save(b2_idx, j_scores, names)
        
        # 3. 对比
        is_flip = (orig_loser != save_loser)
        if is_flip: count_flips += 1
        count_weeks += 1
        
        # 记录
        b2_names = [names[i] for i in b2_idx]
        b2_scores = [j_scores[i] for i in b2_idx]
        
        results.append({
            'Season': s,
            'Week': w,
            'Rule': curr_rule,
            'Original_Loser': orig_loser,
            'Save_Loser': save_loser,
            'Is_Result_Flipped': is_flip,
            'Bottom_2_Contestants': ", ".join(b2_names),
            'Bottom_2_Scores': str(b2_scores),
            'Saved_Contestant': orig_loser if is_flip else None, # 被救者
            'Eliminated_Instead': save_loser if is_flip else None # 替死者
        })
        
    # 保存
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ 分析完成！结果已保存至: {OUTPUT_FILE}")
    print(f"\n=== 机制影响摘要 ===")
    print(f"总模拟周数: {count_weeks}")
    print(f"结果翻转次数: {count_flips}")
    print(f"机制修正率 (Impact Rate): {count_flips/count_weeks*100:.1f}%")
    
    # 展示 Jerry Rice 案例 (S2 W7)
    print("\n[案例验证: Jerry Rice (Season 2 Week 7)]")
    jerry_case = res_df[(res_df['Season']==2) & (res_df['Week']==7)]
    if not jerry_case.empty:
        print(jerry_case[['Original_Loser', 'Save_Loser', 'Is_Result_Flipped', 'Saved_Contestant', 'Eliminated_Instead']].to_string(index=False))

if __name__ == "__main__":
    run_save_impact_analysis()