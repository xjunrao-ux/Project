import pandas as pd
import numpy as np
from scipy import stats
import warnings
import matplotlib.pyplot as plt

# 忽略警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置与数据加载 (Config & Data Load)
# ==========================================
DATA_FILENAME = 'cleaned_DWTS_data.csv'
TARGET_SEASON = 1
OUTPUT_FILENAME = 'season_1_full_forecast.csv'

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # 确保淘汰周次是数值型，填充空值为100 (代表未被淘汰)
        if 'eliminated_week' in df.columns:
            df['eliminated_week_filled'] = df['eliminated_week'].fillna(100)
        return df
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return None

def get_weekly_data(df, season_id, week_num):
    """获取特定周的活跃选手数据"""
    season_df = df[df['season'] == season_id].copy()
    
    # 筛选活跃选手：淘汰周次 >= 当前周
    active_df = season_df[season_df['eliminated_week_filled'] >= week_num].copy()
    
    # 自动查找评委分列 (适配 weekX_judgeY 格式)
    judge_cols = [c for c in df.columns if f'week{week_num}_judge' in c]
    
    if not judge_cols:
        return pd.DataFrame(), None
        
    # 计算评委总分
    active_df['Week_Judge_Total'] = active_df[judge_cols].sum(axis=1)
    # 过滤掉缺赛选手 (分数为0)
    active_df = active_df[active_df['Week_Judge_Total'] > 0]
    
    # 确定本周被淘汰者
    elim_name = None
    elim_candidates = active_df[active_df['eliminated_week_filled'] == week_num]
    if not elim_candidates.empty:
        elim_name = elim_candidates.iloc[0]['celebrity_name']
        
    return active_df[['celebrity_name', 'Week_Judge_Total']], elim_name

# ==========================================
# 2. 贝叶斯 MCMC 模型 (Bayesian Model)
# ==========================================
class BayesianFanVoteReconstructor:
    def __init__(self, n_samples=3000, burn_in=1000):
        self.n_samples = n_samples
        self.burn_in = burn_in

    def _calculate_rank(self, scores):
        # 分数越高 -> 排名越前 (数值越小, 1st)
        return stats.rankdata(-np.array(scores), method='min')

    def _likelihood(self, judge_scores, fan_votes, eliminated_idx):
        """
        Rank 规则似然函数：
        验证 fan_votes 是否会导致 eliminated_idx 选手获得最差的总排名。
        """
        if eliminated_idx is None: return 1.0
        
        j_ranks = self._calculate_rank(judge_scores)
        f_ranks = self._calculate_rank(fan_votes)
        total_ranks = j_ranks + f_ranks
        
        # 寻找最差排名 (数值最大)
        worst_indices = np.where(total_ranks == np.max(total_ranks))[0]
        
        if eliminated_idx in worst_indices:
            return 1.0 # 符合历史事实
        else:
            return 0.0 # 拒绝

    def sample_week(self, judge_scores, eliminated_idx):
        n_c = len(judge_scores)
        samples = []
        current_votes = np.ones(n_c) / n_c
        
        for i in range(self.n_samples + self.burn_in):
            # 随机扰动 (Random Walk)
            noise = np.random.normal(0, 0.05, n_c)
            proposal = np.abs(current_votes + noise)
            proposal /= proposal.sum()
            
            # 接受/拒绝
            if self._likelihood(judge_scores, proposal, eliminated_idx) > 0:
                current_votes = proposal
            elif self._likelihood(judge_scores, current_votes, eliminated_idx) == 0:
                current_votes = proposal # 强制跳出无效状态
            
            if i >= self.burn_in:
                samples.append(current_votes)
                
        return np.array(samples)

# ==========================================
# 3. 主程序：全赛季预测与验证
# ==========================================
def run_full_season_analysis():
    df = load_data(DATA_FILENAME)
    if df is None: return

    model = BayesianFanVoteReconstructor(n_samples=5000, burn_in=1000)
    all_results = []
    
    print(f"--- 开始 Season {TARGET_SEASON} 全赛季分析 ---")
    
    # Season 1 淘汰发生周次：Week 2, 3, 4, 5
    # 我们遍历这些周次
    elimination_weeks = sorted(df[(df['season'] == TARGET_SEASON) & (df['eliminated_week'].notna())]['eliminated_week'].unique())
    elimination_weeks = [int(w) for w in elimination_weeks if w > 1] # 跳过Week 1 (无淘汰)
    
    for w in elimination_weeks:
        df_week, elim_name = get_weekly_data(df, TARGET_SEASON, w)
        
        if df_week.empty: continue
            
        names = df_week['celebrity_name'].values
        j_scores = df_week['Week_Judge_Total'].values
        
        # 找到被淘汰者索引
        elim_idx = None
        if elim_name and elim_name in names:
            elim_idx = np.where(names == elim_name)[0][0]
            
        print(f"正在分析 Week {w}... 参赛: {len(names)} 人, 淘汰: {elim_name}")
        
        # --- 运行模型 ---
        posterior_samples = model.sample_week(j_scores, elim_idx)
        
        if len(posterior_samples) > 0:
            est_votes = posterior_samples.mean(axis=0)
            std_votes = posterior_samples.std(axis=0)
            
            # --- 一致性验证 (Consistency Check) ---
            # 使用估算的平均得票率，重新计算排名，看是否能“复现”淘汰结果
            j_ranks = stats.rankdata(-j_scores, method='min')
            f_ranks = stats.rankdata(-est_votes, method='min')
            total_ranks = j_ranks + f_ranks
            
            # 预测的最差者 (可能有多个并列)
            predicted_worst_indices = np.where(total_ranks == np.max(total_ranks))[0]
            predicted_worst_names = names[predicted_worst_indices]
            
            # 检查真实淘汰者是否在预测的最差名单中
            is_match = elim_name in predicted_worst_names
            
            for i, name in enumerate(names):
                all_results.append({
                    'Season': TARGET_SEASON,
                    'Week': w,
                    'Contestant': name,
                    'Judge_Score': j_scores[i],
                    'Judge_Rank': j_ranks[i],
                    'Est_Fan_Vote_Share': round(est_votes[i], 4),
                    'Uncertainty_Std': round(std_votes[i], 4), # 确定度度量
                    'Fan_Rank_Est': f_ranks[i],
                    'Total_Rank_Sim': total_ranks[i],
                    'Actual_Status': 'Eliminated' if name == elim_name else 'Safe',
                    'Prediction_Match': is_match # 预测结果是否匹配
                })
        else:
            print(f"  警告: Week {w} 无有效解")

    # --- 4. 生成结果表格 ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # 保存 CSV
        results_df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\n✅ 分析完成！结果已保存至: {OUTPUT_FILENAME}")
        
        # 打印预览
        print("\n=== 预测结果预览 (Week 2) ===")
        print(results_df[results_df['Week'] == 2][['Contestant', 'Judge_Score', 'Est_Fan_Vote_Share', 'Uncertainty_Std', 'Actual_Status', 'Prediction_Match']])
        
        # 打印验证总结
        match_count = results_df[results_df['Actual_Status'] == 'Eliminated']['Prediction_Match'].sum()
        total_elim = len(elimination_weeks)
        print(f"\n=== 模型验证总结 ===")
        print(f"在 {total_elim} 个淘汰周中，模型成功复现了 {match_count} 次淘汰结果。")
        print(f"模型一致性 (Consistency): {match_count/total_elim*100:.1f}%")

if __name__ == "__main__":
    run_full_season_analysis()