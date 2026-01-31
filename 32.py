import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')

def main():
    print("正在启动 DWTS 数据分析流程...")

    # ==========================================
    # 1. 数据读取与预处理 (Data Loading)
    # ==========================================
    try:
        # 请确保 'cleaned_DWTS_data.csv' 文件在同一目录下
        df = pd.read_csv('cleaned_DWTS_data.csv')
        print(f"成功读取数据，共 {len(df)} 行。")
    except FileNotFoundError:
        print("错误：未找到 'cleaned_DWTS_data.csv' 文件。请上传数据。")
        return

    # 动态检测列名
    cols = df.columns.tolist()
    state_col = next((c for c in cols if 'state' in c.lower()), None)
    age_col = next((c for c in cols if 'age' in c.lower()), 'celebrity_age_during_season')
    ind_col = next((c for c in cols if 'industry' in c.lower()), 'celebrity_industry')

    # ==========================================
    # 2. 数据重构 (Reshaping to Long Format)
    # ==========================================
    print("正在重构数据格式...")
    long_data = []
    for idx, row in df.iterrows():
        season = row['season']
        celeb = row['celebrity_name']
        age = row[age_col]
        industry = row[ind_col]
        partner = row['ballroom_partner']
        
        # 简单的地域映射逻辑
        region = "International/Other"
        if state_col and pd.notna(row[state_col]):
            s = row[state_col]
            if s in ['CA', 'OR', 'WA', 'AZ', 'NV', 'ID', 'UT', 'HI', 'AK']: region = 'West'
            elif s in ['NY', 'PA', 'NJ', 'MA', 'CT', 'VT', 'NH', 'ME', 'RI']: region = 'Northeast'
            elif s in ['FL', 'TX', 'GA', 'NC', 'VA', 'TN', 'LA', 'AL', 'SC', 'KY', 'OK', 'MS', 'AR', 'WV']: region = 'South'
            elif s in ['IL', 'OH', 'MI', 'IN', 'WI', 'MN', 'MO', 'IA', 'KS', 'NE', 'ND', 'SD']: region = 'Midwest'
            else: region = 'USA (Other)'
        
        elim_week = row['eliminated_week'] if pd.notna(row['eliminated_week']) else 99
        
        # 遍历每一周
        for w in range(1, 12):
            score_col = f'total_score_w{w}'
            if score_col not in df.columns: continue
            score = row[score_col]
            
            # 过滤无效数据
            if pd.isna(score) or score == 0: continue
            
            is_eliminated = 1 if w == elim_week else 0
            
            long_data.append({
                'Season': season,
                'Week': w,
                'Celebrity': celeb,
                'Age': age,
                'Industry': industry,
                'Region': region,
                'Partner': partner,
                'Judge_Score': score,
                'Is_Eliminated': is_eliminated,
                'Weeks_Survived': elim_week if elim_week < 99 else w
            })

    long_df = pd.DataFrame(long_data)

    # ==========================================
    # 3. Q1: 隐变量反演 (Inverse Estimation of Fan Votes)
    # ==========================================
    print("正在运行 Q1 模块：反演观众投票 (Monte Carlo Simulation)...")
    
    def estimate_fan_votes(data, n_simulations=50):
        estimated_votes = []
        grouped = data.groupby(['Season', 'Week'])
        
        for (season, week), group in grouped:
            contestants = group['Celebrity'].tolist()
            judge_scores = group['Judge_Score'].values
            elim_mask = group['Is_Eliminated'].values
            n_c = len(contestants)
            
            # 如果本周无淘汰，返回均匀分布
            if sum(elim_mask) == 0:
                avg_votes = np.ones(n_c)/n_c
            else:
                valid_votes = []
                # 假设 S1-2 和 S28+ 使用排名制，其余使用百分比制
                use_rank = (season <= 2) or (season >= 28)
                
                for _ in range(n_simulations):
                    votes = np.random.dirichlet(np.ones(n_c))
                    
                    if use_rank:
                        # 排名制：分数高->Rank小(1)。总Rank大者淘汰。
                        j_ranks = rankdata(-judge_scores, method='min') 
                        f_ranks = rankdata(-votes, method='min')
                        total = j_ranks + f_ranks
                        worst_idx = np.argmax(total)
                    else:
                        # 百分比制：总占比最小者淘汰
                        j_pct = judge_scores / sum(judge_scores)
                        f_pct = votes
                        total = j_pct + f_pct
                        worst_idx = np.argmin(total)
                    
                    # 校验模拟结果
                    if elim_mask[worst_idx] == 1:
                        valid_votes.append(votes)
                
                # 取有效模拟的均值
                avg_votes = np.mean(valid_votes, axis=0) if valid_votes else np.ones(n_c)/n_c
                
            for i, celeb in enumerate(contestants):
                estimated_votes.append({
                    'Season': season, 
                    'Week': week, 
                    'Celebrity': celeb, 
                    'Est_Fan_Vote': avg_votes[i]
                })
        return pd.DataFrame(estimated_votes)

    fan_votes = estimate_fan_votes(long_df)
    full_df = pd.merge(long_df, fan_votes, on=['Season', 'Week', 'Celebrity'])
    print("观众投票反演完成。")

    # ==========================================
    # 4. Q3: 建模与归因分析 (Advanced Modeling)
    # ==========================================
    print("正在运行 Q3 模块：LMM 与 生存分析...")

    # 4.1 数据标准化 (Standardization)
    full_df['Age_Std'] = (full_df['Age'] - full_df['Age'].mean()) / full_df['Age'].std()
    full_df['Judge_Score_Std'] = (full_df['Judge_Score'] - full_df['Judge_Score'].mean()) / full_df['Judge_Score'].std()
    full_df['Fan_Vote_Std'] = (full_df['Est_Fan_Vote'] - full_df['Est_Fan_Vote'].mean()) / full_df['Est_Fan_Vote'].std()

    # 筛选主要行业 (样本量 > 10) 以保证模型稳定
    top_inds = full_df['Industry'].value_counts()[full_df['Industry'].value_counts() > 10].index
    model_data = full_df[full_df['Industry'].isin(top_inds)].copy()

    # 4.2 线性混合效应模型 (LMM)
    # 目标：分离 评委偏好 vs 观众偏好
    # 随机效应：Partner (剔除舞伴影响)
    formula_judge = "Judge_Score_Std ~ Age_Std + C(Industry) + C(Region)"
    formula_fan = "Fan_Vote_Std ~ Age_Std + C(Industry) + C(Region)"

    print("  - 正在拟合评委模型 (LMM)...")
    model_judge = smf.mixedlm(formula_judge, model_data, groups=model_data['Partner']).fit()
    
    print("  - 正在拟合观众模型 (LMM)...")
    model_fan = smf.mixedlm(formula_fan, model_data, groups=model_data['Partner']).fit()

    # 4.3 生存周期分析 (Survival Analysis Proxy)
    # 使用 OLS 分析各特征对“最大存活周数”的影响
    print("  - 正在拟合生存分析模型...")
    survival_df = full_df.groupby('Celebrity').agg({
        'Weeks_Survived': 'max',
        'Age_Std': 'first',
        'Industry': 'first',
        'Region': 'first'
    }).reset_index()
    survival_df = survival_df[survival_df['Industry'].isin(top_inds)].copy()
    
    formula_surv = "Weeks_Survived ~ Age_Std + C(Industry) + C(Region)"
    model_surv = smf.ols(formula_surv, data=survival_df).fit()

    # ==========================================
    # 5. 结果汇总与导出 (Result Aggregation)
    # ==========================================
    results = []
    
    # 提取系数并进行对比检验
    params_j = model_judge.params
    params_f = model_fan.params
    bse_j = model_judge.bse
    bse_f = model_fan.bse
    params_s = model_surv.params
    pvals_s = model_surv.pvalues

    common_params = params_j.index.intersection(params_f.index)
    
    for p in common_params:
        if 'Group Var' in p: continue # 跳过随机效应方差项
        
        # 提取系数
        coef_j = params_j[p]
        coef_f = params_f[p]
        coef_s = params_s.get(p, np.nan) # 生存模型系数
        pval_s = pvals_s.get(p, np.nan)
        
        # Z-Test: 检验评委系数与观众系数是否有显著差异
        # Z = (b1 - b2) / sqrt(se1^2 + se2^2)
        se_j = bse_j[p]
        se_f = bse_f[p]
        z_score = (coef_j - coef_f) / np.sqrt(se_j**2 + se_f**2)
        p_diff = 2 * (1 - norm.cdf(abs(z_score)))
        
        # 简化变量名
        name = p.replace('C(Industry)[T.', '').replace('C(Region)[T.', '').replace(']', '')
        if name == 'Intercept': name = 'Baseline'
        
        results.append({
            'Factor': name,
            'Judge_Impact': coef_j,   # 评委偏好系数
            'Fan_Impact': coef_f,     # 观众偏好系数
            'Diff_P_Value': p_diff,   # 差异显著性 P值
            'Survival_Impact': coef_s,# 生存周数影响
            'Survival_P_Value': pval_s
        })

    res_df = pd.DataFrame(results)

    # 添加显著性星号
    def sig_star(p):
        if pd.isna(p): return ''
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        return ''

    res_df['Sig_Diff_Judge_Fan'] = res_df['Diff_P_Value'].apply(sig_star)
    res_df['Sig_Survival'] = res_df['Survival_P_Value'].apply(sig_star)

    # 整理列顺序
    cols = ['Factor', 'Judge_Impact', 'Fan_Impact', 'Diff_P_Value', 'Sig_Diff_Judge_Fan', 
            'Survival_Impact', 'Survival_P_Value', 'Sig_Survival']
    final_table = res_df[cols]

    # 保存文件
    output_filename = 'Q3_Factor_Analysis_Results.csv'
    final_table.to_csv(output_filename, index=False)
    
    print("\n" + "="*50)
    print(f"分析完成！结果已保存为: {output_filename}")
    print("="*50)
    print(final_table.to_string())

if __name__ == "__main__":
    main()