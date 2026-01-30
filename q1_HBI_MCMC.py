import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# 忽略警告以保持输出整洁
warnings.filterwarnings('ignore')

# ==========================================
# 1. 数据准备 (Data Preparation)
# ==========================================
def load_and_prep_data(filepath):
    """读取并预处理数据"""
    df = pd.read_csv(filepath)
    # 将淘汰周次中的 NaN 填充为 100 (代表未被淘汰/进入决赛)
    df['eliminated_week_filled'] = df['eliminated_week'].fillna(100)
    return df

def get_weekly_data(df, season_id, week_num):
    """获取特定赛季、特定周次的活跃选手数据"""
    season_df = df[df['season'] == season_id].copy()
    
    # 筛选活跃选手：淘汰周次 >= 当前周
    active_df = season_df[season_df['eliminated_week_filled'] >= week_num].copy()
    
    # 计算本周评委总分
    # 查找形如 'week2_judge1_score' 的列
    judge_cols = [c for c in df.columns if f'week{week_num}_judge' in c]
    
    # 求和 (跳过NaN，处理缺席评委情况)
    active_df['Week_Judge_Total'] = active_df[judge_cols].sum(axis=1)
    
    # 过滤掉分数为0的行 (可能该周缺赛或退赛)
    active_df = active_df[active_df['Week_Judge_Total'] > 0]
    
    # 确定本周被淘汰者
    elim_name = None
    elim_candidates = active_df[active_df['eliminated_week_filled'] == week_num]
    if not elim_candidates.empty:
        elim_name = elim_candidates.iloc[0]['celebrity_name']
        
    return active_df[['celebrity_name', 'Week_Judge_Total']], elim_name

# ==========================================
# 2. 贝叶斯 MCMC 采样器 (The Model)
# ==========================================
class BayesianFanVoteReconstructor:
    def __init__(self, n_samples=3000, burn_in=1000):
        self.n_samples = n_samples # 采样次数
        self.burn_in = burn_in     # 预热期 (丢弃不稳定的初始样本)

    def _calculate_rank(self, scores):
        """
        计算排名。
        注意：分数越高(或票数越多)，排名数值越小(1st, 2nd...)。
        使用 rankdata(-scores) 实现降序排名。
        """
        return stats.rankdata(-np.array(scores), method='min')

    def _likelihood(self, judge_scores, fan_votes, eliminated_idx):
        """
        似然函数：如果生成的粉丝票数导致了正确的淘汰结果，返回 1.0，否则返回 0。
        针对 Season 1 的 Rank 规则：总排名数值最大者(最差)被淘汰。
        """
        if eliminated_idx is None:
            return 1.0 # 本周无人淘汰
            
        # 1. 计算排名
        j_ranks = self._calculate_rank(judge_scores)
        f_ranks = self._calculate_rank(fan_votes)
        
        # 2. 计算总排名 (Rank Sum)
        total_ranks = j_ranks + f_ranks
        
        # 3. 寻找最差者 (总排名数值最大)
        max_rank_val = np.max(total_ranks)
        worst_indices = np.where(total_ranks == max_rank_val)[0]
        
        # 4. 验证历史真实被淘汰者是否在“最差名单”中
        if eliminated_idx in worst_indices:
            return 1.0
        else:
            return 0.0 # 硬约束：不符合历史的样本直接拒绝

    def sample_week(self, judge_scores, eliminated_idx):
        """对单周进行采样"""
        n_contestants = len(judge_scores)
        samples = []
        
        # 初始化：假设大家票数均等
        current_votes = np.ones(n_contestants) / n_contestants
        
        for i in range(self.n_samples + self.burn_in):
            # 提议 (Proposal): 在当前票数上增加随机扰动
            noise = np.random.normal(0, 0.05, n_contestants)
            proposal = np.abs(current_votes + noise)
            proposal /= proposal.sum() # 归一化，保证总和为1
            
            # 计算接受率
            lik_curr = self._likelihood(judge_scores, current_votes, eliminated_idx)
            lik_prop = self._likelihood(judge_scores, proposal, eliminated_idx)
            
            # Metropolis-Hastings 接受准则
            if lik_prop > 0: 
                current_votes = proposal # 只要符合约束就接受
            elif lik_curr == 0:
                current_votes = proposal # 如果当前状态也不符合，尝试跳出
            
            # 记录样本 (跳过预热期)
            if i >= self.burn_in and self._likelihood(judge_scores, current_votes, eliminated_idx) > 0:
                samples.append(current_votes)
                
        return np.array(samples)

# ==========================================
# 3. 运行模型 (Execution)
# ==========================================
# 加载数据
filepath = 'cleaned_DWTS_data.csv' 
df = load_and_prep_data(filepath)
model = BayesianFanVoteReconstructor()

# 存储结果
all_results = []

# 以 Season 1 为例 (Rank 规则)
season_id = 1
print(f"--- Running Reconstruction for Season {season_id} ---")

# Season 1 通常有 6 周，淘汰主要发生在 Week 2-5
weeks_to_process = [2, 3, 4, 5]

for w in weeks_to_process:
    # 获取本周数据
    df_week, elim_name = get_weekly_data(df, season_id, w)
    
    if df_week.empty: continue
    
    names = df_week['celebrity_name'].values
    j_scores = df_week['Week_Judge_Total'].values
    
    # 找到被淘汰者的索引
    elim_idx = None
    if elim_name and elim_name in names:
        elim_idx = np.where(names == elim_name)[0][0]
        
    print(f"Week {w}: {len(names)} contestants. Eliminated: {elim_name}")
    
    # 运行采样
    posterior_samples = model.sample_week(j_scores, elim_idx)
    
    if len(posterior_samples) > 0:
        # 计算统计量
        est_votes = posterior_samples.mean(axis=0) # 粉丝得票率均值
        std_votes = posterior_samples.std(axis=0)  # 不确定性 (标准差)
        
        for i, name in enumerate(names):
            all_results.append({
                'Season': season_id,
                'Week': w,
                'Contestant': name,
                'Judge_Score': j_scores[i],
                'Est_Fan_Vote': est_votes[i],
                'Uncertainty': std_votes[i],
                'Status': 'Eliminated' if name == elim_name else 'Safe'
            })

# 转为 DataFrame 展示
results_df = pd.DataFrame(all_results)
print("\n=== Model Output Sample ===")
print(results_df.head(10))

# ==========================================
# 4. 可视化 (Visualization)
# ==========================================
def plot_results(res_df):
    plt.figure(figsize=(12, 6))
    
    # 筛选几位主要选手展示趋势
    top_names = res_df['Contestant'].unique()[:5]
    
    for name in top_names:
        subset = res_df[res_df['Contestant'] == name]
        plt.plot(subset['Week'], subset['Est_Fan_Vote'], marker='o', label=name)
        # 添加置信区间阴影
        plt.fill_between(subset['Week'], 
                         subset['Est_Fan_Vote'] - subset['Uncertainty'], 
                         subset['Est_Fan_Vote'] + subset['Uncertainty'], alpha=0.1)
    
    plt.title(f'Reconstructed Fan Vote Share (Season {season_id})')
    plt.xlabel('Week')
    plt.ylabel('Estimated Vote Share')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# 绘制结果
if not results_df.empty:
    plot_results(results_df)