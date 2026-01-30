import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

# ==========================================
# 配置区域 (只需修改这里)
# ==========================================
# 1. 告诉程序你的文件名叫什么
DATA_FILENAME = 'cleaned_DWTS_data.csv' 

# 2. 想要跑哪个赛季？(美赛建议先跑 Season 1 验证，再跑全量)
TARGET_SEASON = 20 

# ==========================================
# 核心逻辑 (无需修改)
# ==========================================
warnings.filterwarnings('ignore')

def load_and_prep_data(filepath):
    """读取数据并处理空值"""
    try:
        df = pd.read_csv(filepath)
        print(f"成功读取数据文件: {filepath}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}。请确保csv文件和代码在同一个文件夹内！")
        return None

    # 将淘汰周次中的 NaN 填充为 100 (代表未被淘汰/进入决赛)
    # 确保列名匹配你清洗后的数据 (eliminated_week)
    if 'eliminated_week' in df.columns:
        df['eliminated_week_filled'] = df['eliminated_week'].fillna(100)
    else:
        print("错误：数据中缺少 'eliminated_week' 列，请检查表头。")
        return None
        
    return df

def get_weekly_data(df, season_id, week_num):
    """获取特定周的数据"""
    season_df = df[df['season'] == season_id].copy()
    
    # 筛选活跃选手：淘汰周次 >= 当前周
    active_df = season_df[season_df['eliminated_week_filled'] >= week_num].copy()
    
    # 自动查找评委分列 (适配 weekX_judgeY 格式)
    judge_cols = [c for c in df.columns if f'week{week_num}_judge' in c]
    
    if not judge_cols:
        return pd.DataFrame(), None # 该周没有数据
        
    # 计算评委总分
    active_df['Week_Judge_Total'] = active_df[judge_cols].sum(axis=1)
    
    # 过滤掉分数为0的行 (缺赛)
    active_df = active_df[active_df['Week_Judge_Total'] > 0]
    
    # 确定本周被淘汰者
    elim_name = None
    # 只有 eliminated_week 正好等于当前周的人，才是本周被淘汰的
    elim_candidates = active_df[active_df['eliminated_week_filled'] == week_num]
    if not elim_candidates.empty:
        elim_name = elim_candidates.iloc[0]['celebrity_name']
        
    return active_df[['celebrity_name', 'Week_Judge_Total']], elim_name

class BayesianFanVoteReconstructor:
    """贝叶斯逆向推断模型"""
    def __init__(self, n_samples=3000, burn_in=1000):
        self.n_samples = n_samples
        self.burn_in = burn_in

    def _calculate_rank(self, scores):
        # 分数越高，排名越靠前(数值越小，如1st)
        return stats.rankdata(-np.array(scores), method='min')

    def _likelihood(self, judge_scores, fan_votes, eliminated_idx):
        if eliminated_idx is None: return 1.0
        
        # Rank 规则：总排名数值最大者(最差)被淘汰
        j_ranks = self._calculate_rank(judge_scores)
        f_ranks = self._calculate_rank(fan_votes)
        total_ranks = j_ranks + f_ranks
        
        worst_indices = np.where(total_ranks == np.max(total_ranks))[0]
        
        if eliminated_idx in worst_indices:
            return 1.0
        else:
            return 0.0 # 不符合历史事实，拒绝

    def sample_week(self, judge_scores, eliminated_idx):
        n_c = len(judge_scores)
        samples = []
        current_votes = np.ones(n_c) / n_c
        
        for i in range(self.n_samples + self.burn_in):
            # 随机扰动
            noise = np.random.normal(0, 0.05, n_c)
            proposal = np.abs(current_votes + noise)
            proposal /= proposal.sum()
            
            # 接受/拒绝
            if self._likelihood(judge_scores, proposal, eliminated_idx) > 0:
                current_votes = proposal
            elif self._likelihood(judge_scores, current_votes, eliminated_idx) == 0:
                current_votes = proposal # 强制跳出死胡同
            
            if i >= self.burn_in:
                samples.append(current_votes)
                
        return np.array(samples)

# ==========================================
# 主程序运行入口
# ==========================================
if __name__ == "__main__":
    # 1. 加载数据
    df = load_and_prep_data(DATA_FILENAME)
    
    if df is not None:
        model = BayesianFanVoteReconstructor()
        all_results = []
        
        print(f"--- 开始分析第 {TARGET_SEASON} 赛季 (基于Rank规则逆推) ---")
        
        # 遍历第2周到第10周 (通常第1周不淘汰)
        for w in range(2, 11):
            df_week, elim_name = get_weekly_data(df, TARGET_SEASON, w)
            
            if df_week.empty: 
                continue # 该周无数据，跳过
                
            names = df_week['celebrity_name'].values
            j_scores = df_week['Week_Judge_Total'].values
            
            # 找到被淘汰者的索引
            elim_idx = None
            if elim_name and elim_name in names:
                elim_idx = np.where(names == elim_name)[0][0]
                
            print(f"正在计算第 {w} 周... 参赛人数: {len(names)}, 淘汰者: {elim_name}")
            
            # 运行模型
            posterior_samples = model.sample_week(j_scores, elim_idx)
            
            if len(posterior_samples) > 0:
                est_votes = posterior_samples.mean(axis=0)
                std_votes = posterior_samples.std(axis=0)
                
                for i, name in enumerate(names):
                    all_results.append({
                        'Season': TARGET_SEASON,
                        'Week': w,
                        'Contestant': name,
                        'Judge_Score': j_scores[i],
                        'Est_Fan_Vote': est_votes[i],
                        'Uncertainty': std_votes[i],
                        'Status': 'Eliminated' if name == elim_name else 'Safe'
                    })
            else:
                print(f"  警告: 第 {w} 周未能找到符合约束的解 (可能数据有误或约束过紧)")

        # 结果展示
        if all_results:
            results_df = pd.DataFrame(all_results)
            print("\n=== 计算完成！结果预览 ===")
            print(results_df.head(10))
            
            # 保存结果，供后续小问使用
            output_filename = f'season_{TARGET_SEASON}_estimated_votes.csv'
            results_df.to_csv(output_filename, index=False)
            print(f"\n完整结果已保存至: {output_filename}")
            
            # 简单绘图
            plt.figure(figsize=(10, 6))
            for name in results_df['Contestant'].unique()[:5]: # 只画前5个人
                subset = results_df[results_df['Contestant'] == name]
                plt.errorbar(subset['Week'], subset['Est_Fan_Vote'], yerr=subset['Uncertainty'], label=name, marker='o', capsize=5)
            plt.title(f'Season {TARGET_SEASON} Estimated Fan Votes')
            plt.xlabel('Week')
            plt.ylabel('Vote Share')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("未生成任何结果，请检查赛季ID是否正确。")