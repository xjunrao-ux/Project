import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from statsmodels.duration.hazard_regression import PHReg

def safe_get_param(model_res, feature_name, index):
    """安全获取模型参数，兼容Series和Array格式"""
    if hasattr(model_res.params, 'get'):
        return model_res.params.get(feature_name, np.nan)
    else:
        try:
            return model_res.params[index]
        except:
            return np.nan

def safe_get_pvalue(model_res, feature_name, index):
    """安全获取P值"""
    if hasattr(model_res.pvalues, 'get'):
        return model_res.pvalues.get(feature_name, np.nan)
    else:
        try:
            return model_res.pvalues[index]
        except:
            return np.nan

try:
    print("--- 正在加载与预处理数据 ---")
    # 1. 加载数据
    df_meta = pd.read_csv('cleaned_DWTS_data.csv')
    df_pred = pd.read_csv('all_seasons_final_prediction_with_bounds.csv')

    # 2. 统一列名与格式
    if 'celebrity_name' in df_meta.columns:
        df_meta = df_meta.rename(columns={'celebrity_name': 'Contestant', 'season': 'Season'})
    
    # 标准化名字以确保匹配
    df_meta['Contestant_Norm'] = df_meta['Contestant'].astype(str).str.strip().str.lower()
    df_pred['Contestant_Norm'] = df_pred['Contestant'].astype(str).str.strip().str.lower()
    
    # 3. 聚合预测数据（每位选手的赛季综合表现）
    contestant_stats = df_pred.groupby(['Season', 'Contestant_Norm']).agg({
        'Week': 'max',          # 生存时长（最大周次）
        'Judge_Score': 'mean',  # 平均评委分
        'Est_Fan_Vote': 'mean'  # 平均粉丝得票率
    }).reset_index()
    
    # 4. 匹配特征（年龄、行业）
    # 动态查找列名
    age_col = [c for c in df_meta.columns if 'age' in c.lower()][0]
    ind_col = [c for c in df_meta.columns if 'industry' in c.lower()][0]
    
    df_analysis = contestant_stats.merge(
        df_meta[['Contestant_Norm', 'Season', age_col, ind_col]], 
        on=['Contestant_Norm', 'Season'], 
        how='left'
    )
    
    # 5. 特征工程与清洗
    # 填充缺失年龄
    df_analysis[age_col] = pd.to_numeric(df_analysis[age_col], errors='coerce')
    df_analysis[age_col] = df_analysis[age_col].fillna(df_analysis[age_col].mean())
    
    # 清洗行业：保留Top 5，其余归为Other
    df_analysis[ind_col] = df_analysis[ind_col].fillna('Other')
    top_n = df_analysis[ind_col].value_counts().nlargest(5).index.tolist()
    df_analysis['Industry_Clean'] = df_analysis[ind_col].apply(lambda x: x if x in top_n else 'Other')
    
    # 标准化年龄
    scaler = StandardScaler()
    df_analysis['Age_Std'] = scaler.fit_transform(df_analysis[[age_col]])
    
    # 独热编码 (One-Hot Encoding)
    df_encoded = pd.get_dummies(df_analysis, columns=['Industry_Clean'], drop_first=True)
    # 清理列名中的特殊字符，防止公式报错
    df_encoded.columns = [c.replace('/', '_').replace(' ', '_').replace('-', '_') for c in df_encoded.columns]
    
    # 定义事件：是否被淘汰 (1=Eliminated, 0=Censored/Winner)
    season_max_weeks = df_pred.groupby('Season')['Week'].max()
    df_encoded['Season_Max'] = df_encoded['Season'].map(season_max_weeks)
    df_encoded['Is_Eliminated'] = np.where(df_encoded['Week'] < df_encoded['Season_Max'], 1, 0)
    
    # 6. 准备建模数据
    predictors = ['Age_Std'] + [c for c in df_encoded.columns if 'Industry_Clean_' in c]
    # 强制转换为浮点数，解决 "Pandas data cast" 错误
    model_cols = ['Judge_Score', 'Est_Fan_Vote', 'Week', 'Is_Eliminated'] + predictors
    df_model = df_encoded[model_cols].copy().astype(float)
    df_model = df_model.dropna()
    df_model = df_model[df_model['Week'] > 0] # Cox模型要求时长>0

    print(f"建模数据准备完毕，样本数: {len(df_model)}")
    formula = " + ".join(predictors)

    # 7. 模型训练
    print("正在训练 OLS 回归模型 (评委分 & 粉丝票)...")
    model_judge = smf.ols(f'Judge_Score ~ {formula}', data=df_model).fit()
    model_fan = smf.ols(f'Est_Fan_Vote ~ {formula}', data=df_model).fit()
    
    print("正在训练 Cox 比例风险模型 (生存分析)...")
    # 使用 numpy 数组避免索引对齐问题
    cox_exog = df_model[predictors].values
    cox_endog = df_model['Week'].values
    cox_status = df_model['Is_Eliminated'].values
    cox_model = PHReg(cox_endog, cox_exog, status=cox_status).fit()
    
    print("正在训练 随机生存森林代理 (Random Forest)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(cox_exog, cox_endog)
    
    # 8. 结果提取
    results = []
    cox_params = cox_model.params
    cox_pvalues = cox_model.pvalues
    
    for idx, feat in enumerate(predictors):
        res = {
            'Feature': feat,
            'Judge_Coef': safe_get_param(model_judge, feat, idx),
            'Fan_Coef': safe_get_param(model_fan, feat, idx),
            # Cox HR > 1 表示风险增加
            'Cox_HR': np.exp(cox_params[idx]), 
            'Cox_PVal': cox_pvalues[idx],
            # 特征重要性
            'RF_Importance': rf.feature_importances_[idx]
        }
        results.append(res)
        
    final_df = pd.DataFrame(results).round(4)
    print("\n=== 特征影响分析结果 (Feature Impact Results) ===")
    print(final_df[['Feature', 'Judge_Coef', 'Fan_Coef', 'Cox_HR', 'RF_Importance']].to_string(index=False))
    
    final_df.to_csv('feature_impact_analysis_final.csv', index=False)
    print("\n结果已保存至: feature_impact_analysis_final.csv")

except Exception as e:
    print(f"运行出错: {e}")
    import traceback
    traceback.print_exc()