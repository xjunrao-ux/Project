import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置与数据加载
# ==========================================
INPUT_PRED = 'all_seasons_final_prediction_with_bounds.csv'
INPUT_META = 'cleaned_DWTS_data.csv'
OUTPUT_FILE = 'final_augmented_prediction_results.csv'

def load_and_merge_data():
    print("--- 正在加载数据 ---")
    try:
        df_pred = pd.read_csv(INPUT_PRED)
        df_meta = pd.read_csv(INPUT_META)
        
        # 统一列名
        if 'celebrity_name' in df_meta.columns:
            df_meta = df_meta.rename(columns={'celebrity_name': 'Contestant', 'season': 'Season'})
        
        # 标准化名字以匹配
        df_meta['Contestant_Norm'] = df_meta['Contestant'].astype(str).str.strip().str.lower()
        df_pred['Contestant_Norm'] = df_pred['Contestant'].astype(str).str.strip().str.lower()
        
        # 寻找特征列
        age_col = [c for c in df_meta.columns if 'age' in c.lower()][0]
        ind_col = [c for c in df_meta.columns if 'industry' in c.lower()][0]
        
        # 合并数据
        df_merged = df_pred.merge(
            df_meta[['Contestant_Norm', 'Season', age_col, ind_col]],
            on=['Contestant_Norm', 'Season'],
            how='left'
        )
        
        return df_merged, age_col, ind_col
        
    except Exception as e:
        print(f"错误: 数据加载失败 - {e}")
        return None, None, None

# ==========================================
# 2. 特征工程 (Feature Engineering)
# ==========================================
def preprocess_features(df, age_col, ind_col):
    print("--- 正在进行特征工程 ---")
    
    # 1. 处理缺失值
    df[age_col] = pd.to_numeric(df[age_col], errors='coerce').fillna(35) # 默认35岁
    df[ind_col] = df[ind_col].fillna('Other')
    
    # 2. 行业归类 (Top 5 + Other)
    top_n = df[ind_col].value_counts().nlargest(5).index.tolist()
    df['Industry_Clean'] = df[ind_col].apply(lambda x: x if x in top_n else 'Other')
    
    # 3. 独热编码 (One-Hot Encoding)
    df_encoded = pd.get_dummies(df, columns=['Industry_Clean'], drop_first=False)
    
    # 4. 定义预测用特征
    ind_features = [c for c in df_encoded.columns if 'Industry_Clean_' in c]
    features = ['Judge_Score', age_col] + ind_features
    
    # 填补特征中的空值
    for f in features:
        df_encoded[f] = df_encoded[f].fillna(0)
        
    return df_encoded, features

# ==========================================
# 3. 约束检查逻辑 (Constraint Logic)
# ==========================================
def check_constraint_rank(j_scores, f_votes, loser_idx):
    """Rank制约束检查: 淘汰者的总Rank必须是最大的"""
    j_ranks = stats.rankdata(-j_scores, method='min')
    f_ranks = stats.rankdata(-f_votes, method='min')
    total = j_ranks + f_ranks
    
    max_val = np.max(total)
    candidates = np.where(total == max_val)[0]
    
    if len(candidates) > 1:
        if loser_idx not in candidates: return False
        # Tie-breaker: 粉丝Rank大(票少)者淘汰
        f_ranks_sub = f_ranks[candidates]
        worst_f = np.max(f_ranks_sub)
        return (f_ranks[loser_idx] == worst_f)
    else:
        return (candidates[0] == loser_idx)

def check_constraint_percent(j_scores, f_votes, loser_idx):
    """Percent制约束检查: 淘汰者总分必须最小"""
    j_sum = np.sum(j_scores)
    j_pct = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
    total = j_pct + f_votes
    
    min_idx = np.argmin(total)
    return (min_idx == loser_idx)

# ==========================================
# 4. EM 算法核心 (Iterative Sampling)
# ==========================================
def run_feature_augmented_em(df, features, iterations=2, n_samples=100):
    print(f"--- 启动特征增强型 EM 算法 (Iter={iterations}) ---")
    
    # 初始化估算值 (使用Q1结果或均匀分布)
    if 'Est_Fan_Vote' in df.columns:
        df['Current_Est'] = df['Est_Fan_Vote']
    else:
        df['Current_Est'] = 1.0 / df.groupby(['Season', 'Week'])['Judge_Score'].transform('count')
        
    # 初始化统计列
    df['Uncertainty'] = 0.0
    df['Vote_LB'] = df['Current_Est']
    df['Vote_UB'] = df['Current_Est']
    
    for i in range(iterations):
        print(f"  > 迭代 {i+1}/{iterations}...")
        
        # --- M-Step: 训练回归模型 (利用特征预测投票倾向) ---
        rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        rf.fit(df[features], df['Current_Est'])
        
        # 得到先验预测 (Prior)
        df['Prior_Vote'] = rf.predict(df[features])
        # 归一化 (每周总和为1)
        df['Prior_Vote'] = df.groupby(['Season', 'Week'])['Prior_Vote'].transform(lambda x: x / x.sum())
        
        # --- E-Step: 约束采样 (Constrained Sampling) ---
        new_votes = []
        uncertainties = []
        lbs = []
        ubs = []
        
        grouped = df.groupby(['Season', 'Week'])
        
        for (s, w), group in grouped:
            n_contestants = len(group)
            
            # 如果人太少，跳过
            if n_contestants < 2:
                new_votes.extend(group['Prior_Vote'].values)
                uncertainties.extend(np.zeros(n_contestants))
                lbs.extend(group['Prior_Vote'].values)
                ubs.extend(group['Prior_Vote'].values)
                continue
            
            # 识别真实淘汰者
            loser_row = group[group['Actual_Status'].astype(str).str.contains('Loser', case=False, na=False)]
            
            # 如果本周无淘汰 (如Week 1)，直接信任Prior
            if loser_row.empty:
                new_votes.extend(group['Prior_Vote'].values)
                uncertainties.extend(np.zeros(n_contestants)) # 无淘汰=无硬约束=不确定性由先验决定(简化为0或Prior方差)
                lbs.extend(group['Prior_Vote'].values)
                ubs.extend(group['Prior_Vote'].values)
                continue
                
            loser_name = loser_row['Contestant'].iloc[0]
            try:
                loser_idx = np.where(group['Contestant'].values == loser_name)[0][0]
            except:
                # 名字匹配失败，回退
                new_votes.extend(group['Prior_Vote'].values)
                uncertainties.extend(np.zeros(n_contestants))
                lbs.extend(group['Prior_Vote'].values)
                ubs.extend(group['Prior_Vote'].values)
                continue
                
            rule = group['Rule'].iloc[0]
            priors = group['Prior_Vote'].values
            j_scores = group['Judge_Score'].values
            
            # 采样循环
            valid_samples = []
            # Alpha设置：Strength越大，越相信特征模型；越小，越随机
            alpha = priors * 50 + 1 
            
            for _ in range(n_samples * 5): # 尝试多次以满足约束
                sample = np.random.dirichlet(alpha)
                
                is_valid = False
                if rule == 'RANK':
                    is_valid = check_constraint_rank(j_scores, sample, loser_idx)
                else: # PERCENT
                    is_valid = check_constraint_percent(j_scores, sample, loser_idx)
                
                if is_valid:
                    valid_samples.append(sample)
                    if len(valid_samples) >= n_samples: break
            
            if not valid_samples:
                # 无法找到满足约束的解 (说明Prior偏差极大或数据异常)，回退到Prior
                mean_v = priors
                std_v = np.zeros(n_contestants)
                lb_v = priors
                ub_v = priors
            else:
                valid_samples = np.array(valid_samples)
                mean_v = np.mean(valid_samples, axis=0)
                std_v = np.std(valid_samples, axis=0)
                lb_v = np.percentile(valid_samples, 2.5, axis=0)
                ub_v = np.percentile(valid_samples, 97.5, axis=0)
                
            new_votes.extend(mean_v)
            uncertainties.extend(std_v)
            lbs.extend(lb_v)
            ubs.extend(ub_v)
            
        # 更新估计值
        df['Current_Est'] = new_votes
        
        # 如果是最后一次迭代，保存统计量
        if i == iterations - 1:
            df['Uncertainty'] = uncertainties
            df['Vote_LB'] = lbs
            df['Vote_UB'] = ubs
            
    return df

# ==========================================
# 5. 准确性验证 (Accuracy Check)
# ==========================================
def validate_predictions(df):
    print("--- 正在验证预测准确性 ---")
    results = []
    
    grouped = df.groupby(['Season', 'Week'])
    
    for (s, w), group in grouped:
        if len(group) < 2: 
            results.extend([False]*len(group))
            continue
            
        j_scores = group['Judge_Score'].values
        f_votes = group['Current_Est'].values
        names = group['Contestant'].values
        rule = group['Rule'].iloc[0]
        
        # 重新模拟比赛判定谁被淘汰
        pred_loser_name = None
        
        if rule == 'RANK':
            j_ranks = stats.rankdata(-j_scores, method='min')
            f_ranks = stats.rankdata(-f_votes, method='min')
            total = j_ranks + f_ranks
            max_val = np.max(total)
            candidates = np.where(total == max_val)[0]
            if len(candidates) > 1:
                f_ranks_sub = f_ranks[candidates]
                worst_f = np.max(f_ranks_sub)
                pred_idx = candidates[np.where(f_ranks_sub == worst_f)[0][0]]
            else:
                pred_idx = candidates[0]
            pred_loser_name = names[pred_idx]
        else:
            j_sum = np.sum(j_scores)
            j_pct = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
            total = j_pct + f_votes
            pred_idx = np.argmin(total)
            pred_loser_name = names[pred_idx]
            
        # 记录预测结果
        for _, row in group.iterrows():
            results.append(row['Contestant'] == pred_loser_name)
            
    df['Is_Predicted_Loser'] = results
    
    # 判断是否匹配真实历史
    match_list = []
    for _, row in df.iterrows():
        is_actual_loser = 'Loser' in str(row['Actual_Status']) or 'Elim' in str(row['Actual_Status'])
        # 只要预测的淘汰者 = 真实的淘汰者，即为True
        match_list.append(row['Is_Predicted_Loser'] == is_actual_loser)
        
    df['Match_Success'] = match_list
    return df

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 加载
    df_raw, age_col, ind_col = load_and_merge_data()
    
    if df_raw is not None:
        # 2. 预处理
        df_encoded, features = preprocess_features(df_raw, age_col, ind_col)
        print(f"使用特征进行辅助推断: {features}")
        
        # 3. 运行算法
        df_final = run_feature_augmented_em(df_encoded, features, iterations=2)
        
        # 4. 验证
        df_final = validate_predictions(df_final)
        
        # 5. 输出
        # 重命名列以符合要求
        df_final.rename(columns={'Current_Est': 'Est_Fan_Vote'}, inplace=True)
        
        out_cols = ['Season', 'Week', 'Rule', 'Contestant', 'Judge_Score', 'Est_Fan_Vote', 
                    'Uncertainty', 'Vote_LB', 'Vote_UB', 
                    'Actual_Status', 'Is_Predicted_Loser', 'Match_Success']
        
        df_final[out_cols].to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ 任务完成！文件已保存: {OUTPUT_FILE}")
        print("\n[前5行预览]")
        print(df_final[out_cols].head().to_string())
        
        # 打印准确率统计
        acc = df_final[df_final['Actual_Status'].astype(str).str.contains('Loser')]['Match_Success'].mean()
        print(f"\n[模型表现] 淘汰周次预测准确率: {acc*100:.2f}%")