import pandas as pd
import numpy as np
from scipy import stats
import warnings
import time

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置
# ==========================================
DATA_FILENAME = 'cleaned_DWTS_data.csv'
OUTPUT_FILENAME = 'all_seasons_final_prediction.csv'

def get_scoring_rule(season_id):
    """Season 1-2 & 28+ use RANK; Season 3-27 use PERCENT"""
    if season_id in [1, 2] or season_id >= 28:
        return 'RANK'
    else:
        return 'PERCENT'

# ==========================================
# 2. 数据工具函数
# ==========================================
def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except:
        return None

def get_season_weeks(df, season_id):
    """自动扫描该赛季包含的所有周次"""
    week_cols = [c for c in df.columns if c.startswith('week') and '_judge' in c]
    weeks = set()
    for c in week_cols:
        try:
            part = c.split('_')[0].replace('week', '')
            weeks.add(int(part))
        except: pass
    return sorted(list(weeks))

def get_weekly_participants(df, season_id, week_num):
    """获取活跃选手、评委分及结果"""
    season_df = df[df['season'] == season_id].copy()
    judge_cols = [c for c in df.columns if f'week{week_num}_judge' in c]
    if not judge_cols: return pd.DataFrame()
    
    season_df['Week_Judge_Total'] = season_df[judge_cols].sum(axis=1)
    # 活跃判定: 分数>0
    return season_df[season_df['Week_Judge_Total'] > 0][['celebrity_name', 'Week_Judge_Total', 'placement', 'results']].copy()

def identify_actual_loser(df, season_id, current_week, max_week):
    """
    识别真实淘汰者 (Ground Truth)
    """
    curr_df = get_weekly_participants(df, season_id, current_week)
    if curr_df.empty: return None, curr_df
    
    curr_names = set(curr_df['celebrity_name'].values)
    loser_name = None
    
    if current_week < max_week:
        # 普通周：看下周谁消失了
        next_df = get_weekly_participants(df, season_id, current_week + 1)
        next_names = set(next_df['celebrity_name'].values)
        leavers = list(curr_names - next_names)
        
        # 排除退赛 (Withdrew)
        valid_leavers = []
        for name in leavers:
            res = curr_df[curr_df['celebrity_name'] == name]['results'].iloc[0]
            if not (isinstance(res, str) and "Withdrew" in res):
                valid_leavers.append(name)
        
        if valid_leavers:
            loser_name = valid_leavers[0] # 简单取第一个
    else:
        # 决赛周：看 Placement 最差者 (数值最大)
        non_winners = curr_df[curr_df['placement'] > 1]
        if not non_winners.empty:
            worst_p = non_winners['placement'].max()
            loser_name = non_winners[non_winners['placement'] == worst_p]['celebrity_name'].iloc[0]
            
    return loser_name, curr_df

# ==========================================
# 3. 核心算法: MCMC + 预测验证
# ==========================================
class SmartBayesianModel:
    def __init__(self, rule, n_samples=1500, burn_in=500):
        self.rule = rule
        self.n_samples = n_samples
        self.burn_in = burn_in
    
    def _rank_rule(self, j_scores, f_votes):
        """返回是否符合Rank规则 (Max Total = Eliminated)"""
        j_ranks = stats.rankdata(-np.array(j_scores), method='min')
        f_ranks = stats.rankdata(-np.array(f_votes), method='min')
        return j_ranks + f_ranks
    
    def _percent_rule(self, j_scores, f_votes):
        """返回是否符合Percent规则 (Min Total = Eliminated)"""
        j_sum = np.sum(j_scores)
        if j_sum == 0: return np.zeros_like(j_scores)
        return (np.array(j_scores)/j_sum) + f_votes

    def _likelihood(self, j_scores, f_votes, elim_idx):
        if elim_idx is None: return 1.0
        
        if self.rule == 'RANK':
            totals = self._rank_rule(j_scores, f_votes)
            # 验证: elim_idx 是否是最大值
            return 1.0 if elim_idx in np.where(totals == np.max(totals))[0] else 0.0
        else:
            totals = self._percent_rule(j_scores, f_votes)
            # 验证: elim_idx 是否是最小值
            return 1.0 if elim_idx in np.where(totals == np.min(totals))[0] else 0.0

    def sample(self, j_scores, elim_idx):
        n = len(j_scores)
        samples = []
        curr = np.ones(n)/n
        
        for i in range(self.n_samples + self.burn_in):
            # 扰动
            prop = np.abs(curr + np.random.normal(0, 0.03, n))
            prop /= prop.sum()
            
            # 校验
            if self._likelihood(j_scores, prop, elim_idx) > 0:
                curr = prop
            elif self._likelihood(j_scores, curr, elim_idx) == 0:
                curr = prop # 强制跳出
            
            if i >= self.burn_in:
                samples.append(curr)
        return np.array(samples)

    def predict_outcome(self, j_scores, est_votes, names):
        """基于估算的均值，预测谁会被淘汰"""
        if self.rule == 'RANK':
            totals = self._rank_rule(j_scores, est_votes)
            # 找最差 (Max)
            worst_idx = np.where(totals == np.max(totals))[0]
        else:
            totals = self._percent_rule(j_scores, est_votes)
            # 找最差 (Min)
            worst_idx = np.where(totals == np.min(totals))[0]
        
        return names[worst_idx] # 返回预测的淘汰者名单(数组)

# ==========================================
# 4. 主程序
# ==========================================
def run_final_prediction_task():
    df = load_data(DATA_FILENAME)
    if df is None: return

    all_results = []
    seasons = sorted(df['season'].unique())
    print(f"开始处理 {len(seasons)} 个赛季...")
    
    start_time = time.time()
    
    for s_id in seasons:
        rule = get_scoring_rule(s_id)
        weeks = get_season_weeks(df, s_id)
        if not weeks: continue
        max_week = max(weeks)
        
        model = SmartBayesianModel(rule)
        
        print(f"Season {s_id} ({rule}) processing...")
        
        for w in weeks:
            # 1. 识别真实淘汰者
            real_loser, df_week = identify_actual_loser(df, s_id, w, max_week)
            if df_week.empty: continue
            
            names = df_week['celebrity_name'].values
            j_scores = df_week['Week_Judge_Total'].values
            
            elim_idx = None
            if real_loser and real_loser in names:
                elim_idx = np.where(names == real_loser)[0][0]
            
            # 2. MCMC 估算
            try:
                posterior = model.sample(j_scores, elim_idx)
                if len(posterior) > 0:
                    est_votes = posterior.mean(axis=0)
                    std_votes = posterior.std(axis=0)
                    
                    # 3. 预测匹配验证
                    predicted_losers = model.predict_outcome(j_scores, est_votes, names)
                    match_success = (real_loser in predicted_losers) if real_loser else True
                    
                    # 记录
                    for i, name in enumerate(names):
                        status = 'Actual Loser' if name == real_loser else 'Safe'
                        
                        all_results.append({
                            'Season': s_id,
                            'Week': w,
                            'Rule': rule,
                            'Contestant': name,
                            'Judge_Score': j_scores[i],
                            'Est_Fan_Vote': round(est_votes[i], 4),
                            'Uncertainty': round(std_votes[i], 4), # 确定度
                            'Actual_Status': status,
                            'Is_Predicted_Loser': name in predicted_losers,
                            'Match_Success': match_success # 预测是否匹配
                        })
            except Exception as e:
                print(f"Err {s_id}-{w}: {e}")

    # ==========================================
    # 5. 保存
    # ==========================================
    if all_results:
        res_df = pd.DataFrame(all_results)
        res_df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\n✅ 成功生成文件: {OUTPUT_FILENAME}")
        
        # 验证统计
        total_elim = res_df[res_df['Actual_Status'] == 'Actual Loser'].shape[0]
        matches = res_df[(res_df['Actual_Status'] == 'Actual Loser') & (res_df['Match_Success'] == True)].shape[0]
        print(f"淘汰预测准确率: {matches}/{total_elim} ({matches/total_elim*100:.1f}%)")
        
        # 预览
        print("\n=== S1 Week 2 结果预览 ===")
        print(res_df[(res_df['Season']==1) & (res_df['Week']==2)][['Contestant', 'Est_Fan_Vote', 'Uncertainty', 'Actual_Status', 'Match_Success']])

if __name__ == "__main__":
    run_final_prediction_task()