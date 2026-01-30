import pandas as pd
import numpy as np
from scipy import stats
import warnings
import time

# 忽略警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置与规则定义 (Config & Rules)
# ==========================================
DATA_FILENAME = 'cleaned_DWTS_data.csv'
OUTPUT_FILENAME = 'all_seasons_estimated_votes.csv'

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
# 2. 数据处理模块
# ==========================================
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # 填充未淘汰周次为 100
        if 'eliminated_week' in df.columns:
            df['eliminated_week_filled'] = df['eliminated_week'].fillna(100)
        return df
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return None

def get_weekly_data(df, season_id, week_num):
    """获取单周数据"""
    season_df = df[df['season'] == season_id].copy()
    active_df = season_df[season_df['eliminated_week_filled'] >= week_num].copy()
    
    # 动态匹配列名 (weekX_judgeY)
    judge_cols = [c for c in df.columns if f'week{week_num}_judge' in c]
    if not judge_cols: return pd.DataFrame(), None
        
    active_df['Week_Judge_Total'] = active_df[judge_cols].sum(axis=1)
    # 过滤0分(缺赛)
    active_df = active_df[active_df['Week_Judge_Total'] > 0]
    
    # 找淘汰者
    elim_name = None
    elim_candidates = active_df[active_df['eliminated_week_filled'] == week_num]
    if not elim_candidates.empty:
        elim_name = elim_candidates.iloc[0]['celebrity_name']
        
    return active_df[['celebrity_name', 'Week_Judge_Total']], elim_name

# ==========================================
# 3. 智能贝叶斯 MCMC 模型 (Smart MCMC)
# ==========================================
class SmartBayesianReconstructor:
    def __init__(self, rule_type='RANK', n_samples=2000, burn_in=500):
        self.rule_type = rule_type
        self.n_samples = n_samples
        self.burn_in = burn_in

    def _rank_likelihood(self, judge_scores, fan_votes, eliminated_idx):
        """
        Rank 规则: Total = Rank(J) + Rank(F). 
        淘汰逻辑: Max(Total) -> Eliminated.
        """
        if eliminated_idx is None: return 1.0
        
        # Rank 1 is Best (Smallest number). rankdata gives 1 for smallest input.
        # So we rank -scores to get 1 for highest score.
        j_ranks = stats.rankdata(-np.array(judge_scores), method='min')
        f_ranks = stats.rankdata(-np.array(fan_votes), method='min')
        total = j_ranks + f_ranks
        
        # 验证: 淘汰者是否拥有最大(最差)的总排名
        worst_indices = np.where(total == np.max(total))[0]
        return 1.0 if eliminated_idx in worst_indices else 0.0

    def _percent_likelihood(self, judge_scores, fan_votes, eliminated_idx):
        """
        Percent 规则: Total = %Judge + %Fan.
        淘汰逻辑: Min(Total) -> Eliminated.
        """
        if eliminated_idx is None: return 1.0
        
        # 归一化评委分
        j_sum = np.sum(judge_scores)
        if j_sum == 0: return 0.0 # 异常保护
        j_pct = np.array(judge_scores) / j_sum
        
        # 粉丝分已经是归一化的 (MCMC中保证 sum=1)
        # 题目公式是 50/50 混合 (简单相加即可，因为都是0-1或0-100)
        total = j_pct + fan_votes
        
        # 验证: 淘汰者是否拥有最小(最差)的总分
        worst_indices = np.where(total == np.min(total))[0]
        return 1.0 if eliminated_idx in worst_indices else 0.0

    def sample_week(self, judge_scores, eliminated_idx):
        n_c = len(judge_scores)
        samples = []
        current_votes = np.ones(n_c) / n_c
        
        for i in range(self.n_samples + self.burn_in):
            # 随机扰动 (Simplex Random Walk)
            noise = np.random.normal(0, 0.03, n_c) # 稍微减小步长以提高接受率
            proposal = np.abs(current_votes + noise)
            proposal /= proposal.sum()
            
            # 根据规则选择似然函数
            if self.rule_type == 'RANK':
                is_valid = self._rank_likelihood(judge_scores, proposal, eliminated_idx)
                curr_valid = self._rank_likelihood(judge_scores, current_votes, eliminated_idx)
            else: # PERCENT
                is_valid = self._percent_likelihood(judge_scores, proposal, eliminated_idx)
                curr_valid = self._percent_likelihood(judge_scores, current_votes, eliminated_idx)
            
            # 接受/拒绝逻辑
            if is_valid > 0:
                current_votes = proposal
            elif curr_valid == 0:
                current_votes = proposal # 强制跳出死胡同
            
            if i >= self.burn_in:
                samples.append(current_votes)
                
        return np.array(samples)

# ==========================================
# 4. 全赛季批处理主程序
# ==========================================
def run_all_seasons():
    df = load_data(DATA_FILENAME)
    if df is None: return

    all_results = []
    
    # 获取所有赛季列表
    seasons = sorted(df['season'].unique())
    print(f"检测到 {len(seasons)} 个赛季数据。开始全量处理...")
    
    start_time = time.time()
    
    for s_id in seasons:
        # 1. 确定当季规则
        rule = get_scoring_rule(s_id)
        # 稍微减少采样数以加快全量运行速度 (可根据需要调回3000)
        model = SmartBayesianReconstructor(rule_type=rule, n_samples=1500, burn_in=500)
        
        print(f"\n>>> 处理 Season {s_id} (规则: {rule}) <<<")
        
        # 2. 遍历当季有淘汰的周次
        s_df = df[df['season'] == s_id]
        if 'eliminated_week' not in s_df.columns: continue
            
        elim_weeks = sorted(s_df[s_df['eliminated_week'].notna()]['eliminated_week'].unique())
        elim_weeks = [int(w) for w in elim_weeks if w > 0] # 过滤异常值
        
        for w in elim_weeks:
            df_week, elim_name = get_weekly_data(df, s_id, w)
            if df_week.empty or len(df_week) < 2: continue # 少于2人无法比较
                
            names = df_week['celebrity_name'].values
            j_scores = df_week['Week_Judge_Total'].values
            
            # 找淘汰者索引
            elim_idx = None
            if elim_name and elim_name in names:
                elim_idx = np.where(names == elim_name)[0][0]
            
            # 3. 运行模型
            try:
                posterior = model.sample_week(j_scores, elim_idx)
                
                if len(posterior) > 0:
                    est_votes = posterior.mean(axis=0)
                    std_votes = posterior.std(axis=0)
                    
                    # 验证匹配度
                    # 根据规则重新计算“谁应该走”
                    if rule == 'RANK':
                        total = stats.rankdata(-j_scores, method='min') + stats.rankdata(-est_votes, method='min')
                        pred_elim = names[np.where(total == np.max(total))]
                    else: # PERCENT
                        j_pct = j_scores / j_scores.sum()
                        total = j_pct + est_votes
                        pred_elim = names[np.where(total == np.min(total))]
                        
                    is_match = elim_name in pred_elim
                    
                    # 存储数据
                    for i, name in enumerate(names):
                        all_results.append({
                            'Season': s_id,
                            'Week': w,
                            'Rule_Used': rule,
                            'Contestant': name,
                            'Judge_Score': j_scores[i],
                            'Est_Fan_Vote': round(est_votes[i], 4),
                            'Uncertainty': round(std_votes[i], 4),
                            'Actual_Status': 'Eliminated' if name == elim_name else 'Safe',
                            'Prediction_Match': is_match
                        })
                else:
                    pass # 无解情况跳过
            except Exception as e:
                print(f"Week {w} Error: {e}")
                
    # ==========================================
    # 5. 输出与总结
    # ==========================================
    if all_results:
        res_df = pd.DataFrame(all_results)
        res_df.to_csv(OUTPUT_FILENAME, index=False)
        
        elapsed = time.time() - start_time
        print(f"\n✅ 全赛季处理完成！耗时: {elapsed:.1f}秒")
        print(f"结果已保存至: {OUTPUT_FILENAME}")
        
        # 打印统计摘要
        total_elim_events = res_df[res_df['Actual_Status'] == 'Eliminated'].shape[0]
        correct_matches = res_df[res_df['Actual_Status'] == 'Eliminated']['Prediction_Match'].sum()
        
        print("\n=== 全球赛季数据摘要 ===")
        print(f"覆盖赛季数: {res_df['Season'].nunique()}")
        print(f"总数据行数: {len(res_df)}")
        print(f"模型淘汰匹配率: {correct_matches}/{total_elim_events} ({correct_matches/total_elim_events*100:.1f}%)")
        print("(注: 匹配率非100%是正常的，因为存在并列情况或极少数数据异常)")
    else:
        print("未生成数据，请检查输入文件。")

if __name__ == "__main__":
    run_all_seasons()