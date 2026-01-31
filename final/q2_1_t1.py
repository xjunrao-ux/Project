import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置与加载
# ==========================================
INPUT_FILE = 'all_seasons_final_prediction_with_bounds.csv'
OUTPUT_FILE = 'mechanism_flip_analysis.csv'

def load_data():
    try:
        # 尝试读取带边界的文件，如果不存在则读取普通文件
        df = pd.read_csv(INPUT_FILE)
        print(f"成功加载数据: {INPUT_FILE}")
        return df
    except FileNotFoundError:
        try:
            df = pd.read_csv('all_seasons_final_prediction.csv')
            print("加载备用数据: all_seasons_final_prediction.csv")
            return df
        except:
            print("错误：未找到任何输入数据文件。")
            return None

# ==========================================
# 2. 核心算法引擎
# ==========================================
def calculate_rank_metrics(j_scores, f_votes, names):
    """
    Rank 规则计算引擎
    返回: (总分列表, 排名列表, 淘汰者名字)
    逻辑: Rank(Judge) + Rank(Fan). 数值最大者淘汰.
    """
    # rankdata 'min': 最小的数得1. 我们希望分数高的得1. 所以对负数排名.
    j_ranks = stats.rankdata(-j_scores, method='min')
    f_ranks = stats.rankdata(-f_votes, method='min')
    
    total_score = j_ranks + f_ranks
    
    # 计算本周排名 (总分越小越好 -> 1 is Best)
    final_ranks = stats.rankdata(total_score, method='min')
    
    # 寻找淘汰者 (总分最大者)
    max_val = np.max(total_score)
    candidates = np.where(total_score == max_val)[0]
    
    if len(candidates) > 1:
        # Tie-breaker: 粉丝 Rank 差(数值大)者淘汰
        sub_f = f_ranks[candidates]
        worst_f = np.max(sub_f)
        # 如果粉丝Rank还一样(极罕见)，取第一个
        final_idx = candidates[np.where(sub_f == worst_f)[0][0]]
    else:
        final_idx = candidates[0]
        
    return total_score, final_ranks, names[final_idx]

def calculate_pct_metrics(j_scores, f_votes, names):
    """
    Percentage 规则计算引擎
    返回: (总分列表, 排名列表, 淘汰者名字)
    逻辑: %Judge + %Fan. 数值最小者淘汰.
    """
    j_sum = np.sum(j_scores)
    # 防止除以0
    j_pct = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
    
    total_score = j_pct + f_votes
    
    # 计算本周排名 (总分越大越好 -> 1 is Best)
    final_ranks = stats.rankdata(-total_score, method='min')
    
    # 寻找淘汰者 (总分最小者)
    min_idx = np.argmin(total_score)
    
    return total_score, final_ranks, names[min_idx], j_pct

# ==========================================
# 3. 主分析程序
# ==========================================
def run_flip_analysis():
    df = load_data()
    if df is None: return

    results = []
    
    # 统计计数器
    stats_counter = {
        'total_weeks': 0,
        'flip_weeks': 0
    }
    
    # 按赛季/周次分组
    grouped = df.groupby(['Season', 'Week'])
    
    print("开始执行机制翻转检测...")
    
    for (s, w), group in grouped:
        # 跳过非淘汰周 (如单人数据或决赛展示)
        if len(group) < 2: continue
        
        names = group['Contestant'].values
        j_scores = group['Judge_Score'].values
        f_votes = group['Est_Fan_Vote'].values
        
        # --- 运行 Rank 算法 ---
        r_scores, r_ranks, r_loser = calculate_rank_metrics(j_scores, f_votes, names)
        
        # --- 运行 Percent 算法 ---
        p_scores, p_ranks, p_loser, j_pcts = calculate_pct_metrics(j_scores, f_votes, names)
        
        # --- 判断翻转 ---
        is_flip = (r_loser != p_loser)
        
        if is_flip:
            stats_counter['flip_weeks'] += 1
        stats_counter['total_weeks'] += 1
        
        # --- 记录每位选手的数据 ---
        for i, name in enumerate(names):
            # 判断在各规则下是否安全
            status_rank = 'Eliminated' if name == r_loser else 'Safe'
            status_pct = 'Eliminated' if name == p_loser else 'Safe'
            
            # 判断该选手个人的命运是否因规则改变而不同
            # (例如：在 Rank 下死，在 Percent 下活，或者反之)
            personal_flip = (status_rank != status_pct)
            
            results.append({
                'Season': s,
                'Week': w,
                'Contestant': name,
                'Judge_Score': j_scores[i],
                'Est_Fan_Vote': round(f_votes[i], 4),
                'Actual_Status': group.iloc[i]['Actual_Status'],
                
                # Rank 算法详情
                'Rank_Algo_Score': r_scores[i],
                'Rank_Algo_Rank': r_ranks[i],
                'Sim_Status_Rank': status_rank,
                
                # Percent 算法详情
                'Pct_Algo_Score': round(p_scores[i], 4),
                'Pct_Algo_Rank': p_ranks[i],
                'Sim_Status_Percent': status_pct,
                
                # 核心结论
                'Is_Week_Flipped': is_flip,       # 这一周的结果是否不一样
                'Is_Personal_Fate_Flipped': personal_flip # 这个人的命运是否不一样
            })

    # 保存结果
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_FILE, index=False)
    
    # --- 输出统计摘要 ---
    flip_rate = stats_counter['flip_weeks'] / stats_counter['total_weeks'] * 100
    
    print(f"\n✅ 分析完成！结果已保存至: {OUTPUT_FILE}")
    print("="*40)
    print(f"总模拟周次: {stats_counter['total_weeks']}")
    print(f"结果翻转次数: {stats_counter['flip_weeks']}")
    print(f"机制敏感度 (Flip Rate): {flip_rate:.1f}%")
    print("="*40)
    
    # 打印一个具体的翻转案例供检查
    if stats_counter['flip_weeks'] > 0:
        flip_rows = res_df[res_df['Is_Week_Flipped'] == True]
        example = flip_rows.iloc[0]
        s_ex, w_ex = example['Season'], example['Week']
        
        print(f"\n[案例展示] Season {s_ex} Week {w_ex} 的翻转详情:")
        # 展示该周所有人的简表
        ex_df = res_df[(res_df['Season'] == s_ex) & (res_df['Week'] == w_ex)]
        print(ex_df[['Contestant', 'Judge_Score', 'Rank_Algo_Rank', 'Sim_Status_Rank', 'Pct_Algo_Rank', 'Sim_Status_Percent']].to_string(index=False))
        print("\n解读: 注意看 Sim_Status_Rank 和 Sim_Status_Percent 列的差异。")

if __name__ == "__main__":
    run_flip_analysis()