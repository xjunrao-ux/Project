import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置与加载
# ==========================================
INPUT_FILE = 'all_seasons_final_prediction_with_bounds.csv'
OUTPUT_FILE = 'controversy_flip_analysis.csv'

def load_data():
    try:
        # 尝试读取上一问生成的数据
        return pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"错误：未找到 {INPUT_FILE}。请确保文件存在。")
        return None

# ==========================================
# 2. 核心逻辑：单周模拟与翻转判定
# ==========================================
def calculate_rank_loser(sub_df):
    """Rank Rule: Max Total Rank = Loser"""
    # rankdata 'min': smallest val gets rank 1. We want High Score -> Rank 1.
    j_ranks = stats.rankdata(-sub_df['Judge_Score'].values, method='min')
    f_ranks = stats.rankdata(-sub_df['Est_Fan_Vote'].values, method='min')
    total = j_ranks + f_ranks
    
    max_val = np.max(total)
    candidates = np.where(total == max_val)[0]
    
    # Tie-breaker: Worse Fan Rank (Larger number) loses
    if len(candidates) > 1:
        sub_f = f_ranks[candidates]
        worst_f = np.max(sub_f)
        final_idx = candidates[np.where(sub_f == worst_f)[0][0]]
    else:
        final_idx = candidates[0]
        
    return sub_df.iloc[final_idx]['Contestant']

def calculate_percent_loser(sub_df):
    """Percent Rule: Min Total Percent = Loser"""
    j_scores = sub_df['Judge_Score'].values
    j_sum = np.sum(j_scores)
    j_pct = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
    
    # Fan votes are already shares/pct
    total = j_pct + sub_df['Est_Fan_Vote'].values
    
    min_idx = np.argmin(total)
    return sub_df.iloc[min_idx]['Contestant']

def calculate_save_loser(sub_df, current_rule):
    """Judges' Save: Bottom 2 -> Lower Judge Score Eliminated"""
    j_scores = sub_df['Judge_Score'].values
    names = sub_df['Contestant'].values
    
    # Identify Bottom 2 based on Current Rule
    if current_rule == 'RANK':
        j_ranks = stats.rankdata(-sub_df['Judge_Score'].values, method='min')
        f_ranks = stats.rankdata(-sub_df['Est_Fan_Vote'].values, method='min')
        total = j_ranks + f_ranks
        sorted_idx = np.argsort(-total) # Descending (Worst/Largest Rank first)
    else:
        j_sum = np.sum(j_scores)
        j_pct = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
        total = j_pct + sub_df['Est_Fan_Vote'].values
        sorted_idx = np.argsort(total) # Ascending (Worst/Smallest Pct first)
        
    bottom2_idx = sorted_idx[:2]
    
    # Judge Decision: Eliminate lower score
    p1_idx, p2_idx = bottom2_idx[0], bottom2_idx[1]
    
    if j_scores[p1_idx] < j_scores[p2_idx]:
        return names[p1_idx]
    elif j_scores[p1_idx] > j_scores[p2_idx]:
        return names[p2_idx]
    else:
        return names[p1_idx] # Tie -> Original worst goes

def normalize_status(status_str):
    """将各种状态描述统一为 ELIMINATED 或 Safe"""
    s = str(status_str).upper()
    if 'LOSER' in s or 'ELIM' in s or 'OUT' in s:
        return 'ELIMINATED'
    return 'Safe'

# ==========================================
# 3. 主程序
# ==========================================
def run_controversy_analysis():
    df = load_data()
    if df is None: return

    # 定义目标争议人物
    targets = [
        (2, 'Jerry Rice'),
        (4, 'Billy Ray Cyrus'),
        (11, 'Bristol Palin'),
        (27, 'Bobby Bones')
    ]
    
    results = []
    
    print("开始争议人物反事实翻转分析...")
    
    for season_id, name in targets:
        # 获取该选手所在赛季的所有数据
        season_data = df[df['Season'] == season_id]
        if name not in season_data['Contestant'].values: continue
            
        # 遍历该选手存活的每一周
        weeks = sorted(season_data['Week'].unique())
        
        for w in weeks:
            week_data = season_data[season_data['Week'] == w]
            if len(week_data) < 2: continue # 跳过决赛或单人数据
            
            # 如果该选手本周不在（已淘汰），不再分析
            if name not in week_data['Contestant'].values: continue
                
            # 获取当前环境
            curr_rule = week_data['Rule'].iloc[0]
            contestant_row = week_data[week_data['Contestant'] == name].iloc[0]
            
            # 1. 获取真实状态 (Ground Truth)
            raw_status = contestant_row['Actual_Status']
            real_status_norm = normalize_status(raw_status)
            
            # 2. 运行三种机制模拟
            loser_rank = calculate_rank_loser(week_data)
            loser_pct = calculate_percent_loser(week_data)
            loser_save = calculate_save_loser(week_data, curr_rule)
            
            # 3. 确定模拟状态
            sim_status_rank = 'ELIMINATED' if loser_rank == name else 'Safe'
            sim_status_pct = 'ELIMINATED' if loser_pct == name else 'Safe'
            sim_status_save = 'ELIMINATED' if loser_save == name else 'Safe'
            
            # 4. 判断是否翻转 (Flip Check)
            # Flip = 模拟结果与真实历史不符
            # 对于争议人物，我们特别关注：原本 Safe -> 模拟 ELIMINATED
            flip_rank = (sim_status_rank != real_status_norm)
            flip_pct = (sim_status_pct != real_status_norm)
            flip_save = (sim_status_save != real_status_norm)
            
            results.append({
                'Season': season_id,
                'Contestant': name,
                'Week': w,
                'Judge_Score': contestant_row['Judge_Score'],
                'Original_Rule': curr_rule,
                'Actual_Status': real_status_norm,
                
                # Simulation Outcomes
                'Outcome_Rank': sim_status_rank,
                'Outcome_Pct': sim_status_pct,
                'Outcome_Save': sim_status_save,
                
                # Flip Flags (Did the result change?)
                'Flip_if_Rank': flip_rank,
                'Flip_if_Pct': flip_pct,
                'Flip_if_Save': flip_save
            })
            
    # 输出与保存
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 分析完成！结果已保存至: {OUTPUT_FILE}")
    
    # === 关键发现摘要 ===
    print("\n=== 关键翻转时刻 (Result Flips) ===")
    # 筛选出原本安全，但在某种新规则下被淘汰的时刻
    critical_flips = res_df[
        (res_df['Actual_Status'] == 'Safe') & 
        (res_df[['Flip_if_Rank', 'Flip_if_Pct', 'Flip_if_Save']].any(axis=1))
    ]
    
    if not critical_flips.empty:
        # 格式化打印
        cols_to_show = ['Season', 'Contestant', 'Week', 'Original_Rule', 'Outcome_Rank', 'Outcome_Pct', 'Outcome_Save']
        print(critical_flips[cols_to_show].to_string(index=False))
        print(f"\n共发现 {len(critical_flips)} 个潜在淘汰点。")
    else:
        print("未发现任何翻转点，说明这些选手的生存能力极强，不受规则影响。")

if __name__ == "__main__":
    run_controversy_analysis()