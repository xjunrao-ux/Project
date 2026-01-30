import pandas as pd
import numpy as np
from scipy import stats
import warnings
import time

# 忽略警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置与规则定义
# ==========================================
DATA_FILENAME = 'cleaned_DWTS_data.csv'
OUTPUT_FILENAME = 'all_seasons_full_estimated_votes.csv'

def get_scoring_rule(season_id):
    """
    根据题目描述返回当季规则
    Rank Rule: Season 1-2, 28-34
    Percent Rule: Season 3-27
    """
    if season_id in [1, 2] or season_id >= 28:
        return 'RANK'
    else:
        return 'PERCENT'

# ==========================================
# 2. 数据处理与淘汰检测模块 (升级版)
# ==========================================
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return None

def get_season_weeks(df, season_id):
    """
    自动检测该赛季有多少周 (通过扫描列名)
    返回: [1, 2, ..., max_week]
    """
    # 查找所有形如 weekX_judge... 的列
    week_cols = [c for c in df.columns if c.startswith('week') and '_judge' in c]
    # 提取周数
    weeks = set()
    for c in week_cols:
        try:
            # 格式通常是 week1_judge... 或 week10_judge...
            # 提取 'week' 和 '_' 之间的数字
            part = c.split('_')[0].replace('week', '')
            weeks.add(int(part))
        except:
            pass
    return sorted(list(weeks))

def get_weekly_participants(df, season_id, week_num):
    """获取该周参赛选手名单和评委分"""
    season_df = df[df['season'] == season_id].copy()
    
    # 查找该周评委分列
    judge_cols = [c for c in df.columns if f'week{week_num}_judge' in c]
    if not judge_cols:
        return pd.DataFrame()
    
    # 计算总分
    season_df['Week_Judge_Total'] = season_df[judge_cols].sum(axis=1)
    
    # 筛选活跃选手 (分数 > 0)
    active_df = season_df[season_df['Week_Judge_Total'] > 0].copy()
    
    return active_df[['celebrity_name', 'Week_Judge_Total', 'placement', 'results']]

def identify_loser(df, season_id, current_week, max_week):
    """
    核心逻辑：识别本周谁被淘汰了 (Loser)
    """
    # 1. 获取本周参赛者
    curr_df = get_weekly_participants(df, season_id, current_week)
    if curr_df.empty: return None, curr_df
    
    curr_names = set(curr_df['celebrity_name'].values)
    
    target_loser = None
    
    # 2. 判断逻辑
    if current_week < max_week:
        # === 普通周 ===
        # 看谁下周不在了
        next_df = get_weekly_participants(df, season_id, current_week + 1)
        next_names = set(next_df['celebrity_name'].values)
        
        # 消失的人 (Leavers)
        leavers = list(curr_names - next_names)
        
        if leavers:
            # 通常只有一个淘汰者，如果有多个，取其中之一或都取
            # 这里简单取 placement 最差的那个作为主要目标 (应对退赛等情况)
            # 检查是否有 "Withdrew"
            valid_leavers = []
            for name in leavers:
                status = curr_df[curr_df['celebrity_name'] == name]['results'].iloc[0]
                if isinstance(status, str) and "Withdrew" in status:
                    continue # 退赛不算票选淘汰
                valid_leavers.append(name)
            
            if valid_leavers:
                target_loser = valid_leavers[0] # 取第一个检测到的
                
    else:
        # === 决赛周 (Last Week) ===
        # 没有下周了，直接看 Placement
        # 找出 Placement 最差的人 (数值最大，比如 2nd Place > 1st Place)
        # 注意：排除冠军
        # 假设冠军 placement = 1
        non_winners = curr_df[curr_df['placement'] > 1]
        if not non_winners.empty:
            # 取排名最靠后的 (数值最大)
            worst_placement = non_winners['placement'].max()
            target_loser = non_winners[non_winners['placement'] == worst_placement]['celebrity_name'].iloc[0]
            
    return target_loser, curr_df

# ==========================================
# 3. 智能贝叶斯 MCMC 模型 (保持不变)
# ==========================================
class SmartBayesianReconstructor:
    def __init__(self, rule_type='RANK', n_samples=2000, burn_in=500):
        self.rule_type = rule_type
        self.n_samples = n_samples
        self.burn_in = burn_in

    def _rank_likelihood(self, judge_scores, fan_votes, eliminated_idx):
        if eliminated_idx is None: return 1.0 # 无淘汰 (如Week 1)，无约束
        j_ranks = stats.rankdata(-np.array(judge_scores), method='min')
        f_ranks = stats.rankdata(-np.array(fan_votes), method='min')
        total = j_ranks + f_ranks
        worst_indices = np.where(total == np.max(total))[0]
        return 1.0 if eliminated_idx in worst_indices else 0.0

    def _percent_likelihood(self, judge_scores, fan_votes, eliminated_idx):
        if eliminated_idx is None: return 1.0
        j_sum = np.sum(judge_scores)
        if j_sum == 0: return 0.0
        j_pct = np.array(judge_scores) / j_sum
        total = j_pct + fan_votes
        worst_indices = np.where(total == np.min(total))[0]
        return 1.0 if eliminated_idx in worst_indices else 0.0

    def sample_week(self, judge_scores, eliminated_idx):
        n_c = len(judge_scores)
        samples = []
        current_votes = np.ones(n_c) / n_c
        
        for i in range(self.n_samples + self.burn_in):
            noise = np.random.normal(0, 0.03, n_c)
            proposal = np.abs(current_votes + noise)
            proposal /= proposal.sum()
            
            if self.rule_type == 'RANK':
                is_valid = self._rank_likelihood(judge_scores, proposal, eliminated_idx)
                curr_valid = self._rank_likelihood(judge_scores, current_votes, eliminated_idx)
            else:
                is_valid = self._percent_likelihood(judge_scores, proposal, eliminated_idx)
                curr_valid = self._percent_likelihood(judge_scores, current_votes, eliminated_idx)
            
            if is_valid > 0:
                current_votes = proposal
            elif curr_valid == 0:
                current_votes = proposal
            
            if i >= self.burn_in:
                samples.append(current_votes)
        return np.array(samples)

# ==========================================
# 4. 全赛季批处理主程序
# ==========================================
def run_full_analysis():
    df = load_data(DATA_FILENAME)
    if df is None: return

    all_results = []
    seasons = sorted(df['season'].unique())
    print(f"检测到 {len(seasons)} 个赛季。开始全量处理 (包含决赛周)...")
    
    start_time = time.time()
    
    for s_id in seasons:
        rule = get_scoring_rule(s_id)
        # 获取该赛季所有有效周次
        weeks = get_season_weeks(df, s_id)
        if not weeks: continue
        
        max_week = max(weeks)
        # 适当减少采样数以提升速度
        model = SmartBayesianReconstructor(rule_type=rule, n_samples=1500, burn_in=500)
        
        print(f"处理 Season {s_id} ({len(weeks)} 周) ...")
        
        for w in weeks:
            # 1. 识别本周情况
            loser_name, df_week = identify_loser(df, s_id, w, max_week)
            
            if df_week.empty: continue
            
            names = df_week['celebrity_name'].values
            j_scores = df_week['Week_Judge_Total'].values
            
            # 2. 找到 Loser 索引
            elim_idx = None
            if loser_name and loser_name in names:
                elim_idx = np.where(names == loser_name)[0][0]
            
            # 3. 运行 MCMC
            try:
                posterior = model.sample_week(j_scores, elim_idx)
                
                if len(posterior) > 0:
                    est_votes = posterior.mean(axis=0)
                    std_votes = posterior.std(axis=0)
                    
                    # 记录结果
                    for i, name in enumerate(names):
                        status = 'Safe'
                        if name == loser_name:
                            status = 'Eliminated' if w < max_week else 'Runner-up/Lost'
                        
                        all_results.append({
                            'Season': s_id,
                            'Week': w,
                            'Rule_Used': rule,
                            'Contestant': name,
                            'Judge_Score': j_scores[i],
                            'Est_Fan_Vote': round(est_votes[i], 4),
                            'Uncertainty': round(std_votes[i], 4),
                            'Status': status
                        })
            except Exception as e:
                print(f"  Error S{s_id} W{w}: {e}")

    # ==========================================
    # 5. 输出
    # ==========================================
    if all_results:
        res_df = pd.DataFrame(all_results)
        res_df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\n✅ 处理完成！已生成包含所有周次(含决赛)的文件: {OUTPUT_FILENAME}")
        
        # 验证一下 Season 1 是否有 Week 5 和 Week 6
        s1 = res_df[res_df['Season'] == 1]
        print(f"Season 1 包含周次: {sorted(s1['Week'].unique())}")
        print("预览 S1 最后一周结果:")
        print(s1[s1['Week'] == s1['Week'].max()][['Week', 'Contestant', 'Judge_Score', 'Est_Fan_Vote', 'Status']])
    else:
        print("未生成数据。")

if __name__ == "__main__":
    run_full_analysis()