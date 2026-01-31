import pandas as pd
import numpy as np
import scipy.optimize as opt
import warnings

# 忽略数值计算中的一些警告
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 工具函数
# ---------------------------------------------------------

def parse_elimination_week(res_str):
    """解析 results 列，返回被淘汰的周次。如果没有被淘汰则返回 100"""
    if pd.isna(res_str):
        return 100
    res_str = str(res_str).strip()
    
    # 处理 "Eliminated Week X"
    if 'Eliminated' in res_str and 'Week' in res_str:
        try:
            # 提取数字
            parts = res_str.split('Week')
            if len(parts) > 1:
                return int(parts[1].strip())
        except:
            pass
            
    # 处理冠军、亚军等情况
    if any(x in res_str.lower() for x in ['winner', 'runner', 'place', 'finalist']):
        return 100
        
    # 处理退赛 "Withdrew" - 通常视作该周离开，但这里为了排名预测，暂且视作非正常淘汰
    # 或者如果题目要求预测退赛，可以将退赛视作淘汰。这里设为100以避免干扰模型训练。
    if 'Withdrew' in res_str:
        return 100 
        
    return 100

def softmax_neg(x, temperature=0.1):
    """Softmax的负向变体：分数越低，概率越大（用于计算成为最后一名的概率）"""
    x_stable = x - np.min(x)
    exp_x = np.exp(-x_stable / temperature)
    sum_exp = np.sum(exp_x)
    if sum_exp == 0: return np.ones_like(x) / len(x)
    return exp_x / sum_exp

# ---------------------------------------------------------
# 核心模型类
# ---------------------------------------------------------

class DWTSEstimator:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.process_data()
        
    def process_data(self):
        # 提取淘汰周次
        self.df['Elim_Week_Num'] = self.df['results'].apply(parse_elimination_week)
        
        # 识别分数 列
        self.score_cols = {}
        for w in range(1, 15): # 假设最多15周
            col = f'total_score_w{w}'
            if col in self.df.columns:
                self.score_cols[w] = col
                
        self.seasons = sorted(self.df['season'].unique())
        
    def get_season_data(self, season):
        return self.df[self.df['season'] == season].copy().reset_index(drop=True)
        
    def run_all_seasons(self, output_file='all_seasons_final_prediction_with_bounds.csv'):
        all_results = []
        
        print(f">>> 开始处理 {len(self.seasons)} 个赛季的数据...")
        
        for season in self.seasons:
            season_df = self.get_season_data(season)
            n_contestants = len(season_df)
            if n_contestants < 3: continue
            
            print(f"    Processing Season {season}...")
            
            # 构建该赛季的周数据字典
            weeks_data = {}
            max_week = int(season_df['Elim_Week_Num'].max())
            if max_week > 15: max_week = 12 # 修正异常大值
            
            for w in range(1, max_week + 1):
                if w not in self.score_cols: continue
                
                # 本周在该场的选手 (淘汰周次 >= w)
                active_mask = season_df['Elim_Week_Num'] >= w
                active_idx = np.where(active_mask)[0]
                
                if len(active_idx) < 2: continue
                
                # 本周被淘汰的选手 (淘汰周次 == w)
                elim_mask = season_df['Elim_Week_Num'] == w
                elim_idx = np.where(elim_mask)[0]
                
                # 提取裁判分
                raw_scores = season_df.loc[active_idx, self.score_cols[w]].values
                # 填充NaN
                raw_scores = np.nan_to_num(raw_scores, nan=np.nanmean(raw_scores) if len(raw_scores)>0 else 0)
                
                weeks_data[w] = {
                    'active_idx': active_idx,
                    'elim_idx': elim_idx,
                    'judge_scores': raw_scores
                }
            
            if not weeks_data: continue

            # --- 优化目标函数 ---
            def objective(params, noise_scale=0.0):
                # params: log(popularity) for each contestant
                pops = np.exp(params)
                nll = 0.0
                
                for w, info in weeks_data.items():
                    if len(info['elim_idx']) == 0: continue
                    
                    # 1. Judge Share
                    j_scores = info['judge_scores']
                    if noise_scale > 0:
                        j_scores = j_scores + np.random.normal(0, noise_scale, size=len(j_scores))
                    
                    j_sum = np.sum(j_scores)
                    j_share = j_scores / j_sum if j_sum > 0 else np.ones_like(j_scores)/len(j_scores)
                    
                    # 2. Fan Share (from params)
                    current_pops = pops[info['active_idx']]
                    p_sum = np.sum(current_pops)
                    f_share = current_pops / p_sum if p_sum > 0 else np.ones_like(current_pops)/len(current_pops)
                    
                    # 3. Total Score
                    total = 0.5 * j_share + 0.5 * f_share
                    
                    # 4. Probability of being eliminated (Lowest Score)
                    # 使用 Softmax Neg 计算成为倒数第一的概率
                    prob_bottom = softmax_neg(total, temperature=0.05)
                    
                    # 5. Add NLL for actual eliminated contestants
                    # 找到被淘汰者在当前 active 列表中的索引
                    local_elim_indices = []
                    for glob_idx in info['elim_idx']:
                        # 查找 glob_idx 在 info['active_idx'] 中的位置
                        matches = np.where(info['active_idx'] == glob_idx)[0]
                        if len(matches) > 0:
                            local_elim_indices.append(matches[0])
                            
                    for loc_i in local_elim_indices:
                        p = prob_bottom[loc_i]
                        nll -= np.log(max(p, 1e-6))
                        
                # L2 Regularization
                nll += 0.01 * np.sum(params**2)
                return nll

            # --- 蒙特卡洛模拟 (Bootstrap) ---
            # 运行多次优化，每次加入随机扰动，以获得置信区间
            n_sims = 10 # 增加次数可提高精度，但耗时
            sim_results = []
            
            for i in range(n_sims):
                init_guess = np.random.normal(0, 0.5, n_contestants)
                try:
                    res = opt.minimize(objective, init_guess, args=(0.5 if i>0 else 0.0), method='L-BFGS-B')
                    est_pops = np.exp(res.x)
                    
                    # 记录该次模拟下每一周的结果
                    for w, info in weeks_data.items():
                        act_idx = info['active_idx']
                        j_scores = info['judge_scores']
                        
                        j_sum = np.sum(j_scores)
                        j_share = j_scores / j_sum if j_sum > 0 else np.ones_like(j_scores)/len(j_scores)
                        
                        curr_pops = est_pops[act_idx]
                        f_share = curr_pops / np.sum(curr_pops)
                        
                        total_score = 0.5 * j_share + 0.5 * f_share
                        
                        # 确定该次模拟的排名（分数从高到低）
                        # argsort 返回的是从小到大的索引，[::-1] 翻转
                        # 我们关心谁是最后一名 (min score)
                        
                        for k, original_idx in enumerate(act_idx):
                            sim_results.append({
                                'Season': season,
                                'Week': w,
                                'Contestant_Idx': original_idx,
                                'Contestant': season_df.loc[original_idx, 'celebrity_name'],
                                'Judge_Score': j_scores[k],
                                'Sim_Fan_Vote': f_share[k],
                                'Sim_Total_Score': total_score[k]
                            })
                except Exception as e:
                    continue

            # --- 汇总结果 ---
            if not sim_results: continue
            
            df_sim = pd.DataFrame(sim_results)
            
            # 按选手和周次聚合
            grouped = df_sim.groupby(['Season', 'Week', 'Contestant'])
            stats = grouped.agg(
                Judge_Score=('Judge_Score', 'mean'),
                Est_Fan_Vote=('Sim_Fan_Vote', 'mean'),
                Uncertainty=('Sim_Fan_Vote', 'std'),
                Vote_LB=('Sim_Fan_Vote', 'min'),
                Vote_UB=('Sim_Fan_Vote', 'max'),
                Mean_Total_Score=('Sim_Total_Score', 'mean')
            ).reset_index()
            
            # 填充 NaN std
            stats['Uncertainty'] = stats['Uncertainty'].fillna(0)
            
            # --- 计算排名与匹配度 ---
            # 既然我们要输出 Prediction Match，需要在每组内重新看谁分数最低
            
            def check_correctness(g):
                # 找出该周被淘汰的人
                week_num = g['Week'].iloc[0]
                eliminated_names = season_df[season_df['Elim_Week_Num'] == week_num]['celebrity_name'].values
                
                # 找出预测分数最低的人 (Predicted Loser)
                # 假设只有最后一名被淘汰 (Bottom 1)
                g = g.sort_values('Mean_Total_Score', ascending=True) # 分数低在前
                
                # 标记 Predicted Loser (分数最低的那个人)
                g['Is_Predicted_Loser'] = False
                g.iloc[0, g.columns.get_loc('Is_Predicted_Loser')] = True
                
                # 如果当周有多人淘汰，可能有多个 Predicted Loser? 
                # 这里简化处理：Bottom N 对应 N 个淘汰者
                num_elim = len(eliminated_names)
                if num_elim > 1:
                     g.iloc[:num_elim, g.columns.get_loc('Is_Predicted_Loser')] = True
                
                # 标记 Actual Status
                g['Actual_Status'] = g['Contestant'].apply(lambda x: 'Actual Loser' if x in eliminated_names else 'Safe')
                
                # 标记 Rule
                g['Rule'] = 'RANK'
                
                # 标记 Match Success
                # 逻辑：预测是输家 且 实际是输家 -> True
                #      预测是赢家 且 实际是赢家 -> True
                #      否则 -> False
                g['Match_Success'] = g['Is_Predicted_Loser'] == (g['Actual_Status'] == 'Actual Loser')
                
                return g

            season_final = stats.groupby('Week').apply(check_correctness)
            all_results.append(season_final)
            
        # 合并所有数据
        if len(all_results) > 0:
            final_df = pd.concat(all_results, ignore_index=True)
            
            # 整理列顺序
            cols = ['Season', 'Week', 'Rule', 'Contestant', 'Judge_Score', 
                    'Est_Fan_Vote', 'Uncertainty', 'Vote_LB', 'Vote_UB', 
                    'Actual_Status', 'Is_Predicted_Loser', 'Match_Success']
            
            final_df = final_df[cols]
            
            # 简单格式化
            final_df['Est_Fan_Vote'] = final_df['Est_Fan_Vote'].round(4)
            final_df['Uncertainty'] = final_df['Uncertainty'].round(4)
            final_df['Vote_LB'] = final_df['Vote_LB'].round(4)
            final_df['Vote_UB'] = final_df['Vote_UB'].round(4)
            
            final_df.to_csv(output_file, index=False)
            print(f">>> 完成！文件已保存: {output_file}")
            return final_df
        else:
            print("未能生成任何结果。")
            return None

# 运行程序
if __name__ == "__main__":
    model = DWTSEstimator('cleaned_DWTS_data.csv')
    df_out = model.run_all_seasons()
    if df_out is not None:
        print(df_out.head())