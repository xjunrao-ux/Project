import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from statsmodels.duration.hazard_regression import PHReg

def safe_get_param(model_res, feature_name):
    """Robust parameter extraction"""
    if hasattr(model_res, 'params'):
        params = model_res.params
        if hasattr(params, 'get'): # Series/Dict
            return params.get(feature_name, np.nan)
        elif hasattr(params, 'index'): # Series
            if feature_name in params.index:
                return params[feature_name]
    return np.nan

def safe_get_pvalue(model_res, feature_name):
    """Robust p-value extraction"""
    if hasattr(model_res, 'pvalues'):
        pvals = model_res.pvalues
        if hasattr(pvals, 'get'):
            return pvals.get(feature_name, np.nan)
        elif hasattr(pvals, 'index'):
             if feature_name in pvals.index:
                return pvals[feature_name]
    return np.nan

try:
    print("--- Loading & Preprocessing Data ---")
    df_meta = pd.read_csv('cleaned_DWTS_data.csv')
    df_pred = pd.read_csv('all_seasons_final_prediction_with_bounds.csv')

    if 'celebrity_name' in df_meta.columns:
        df_meta = df_meta.rename(columns={'celebrity_name': 'Contestant', 'season': 'Season'})
    
    df_meta['Contestant_Norm'] = df_meta['Contestant'].astype(str).str.strip().str.lower()
    df_pred['Contestant_Norm'] = df_pred['Contestant'].astype(str).str.strip().str.lower()
    
    contestant_stats = df_pred.groupby(['Season', 'Contestant_Norm']).agg({
        'Week': 'max',
        'Judge_Score': 'mean',
        'Est_Fan_Vote': 'mean'
    }).reset_index()
    
    age_col = [c for c in df_meta.columns if 'age' in c.lower()][0]
    ind_col = [c for c in df_meta.columns if 'industry' in c.lower()][0]
    ctry_col = [c for c in df_meta.columns if 'country' in c.lower()][0]
    state_col = [c for c in df_meta.columns if 'state' in c.lower()][0]
    
    df_analysis = contestant_stats.merge(
        df_meta[['Contestant_Norm', 'Season', age_col, ind_col, ctry_col, state_col]], 
        on=['Contestant_Norm', 'Season'], 
        how='left'
    )
    
    df_analysis[age_col] = pd.to_numeric(df_analysis[age_col], errors='coerce')
    df_analysis[age_col] = df_analysis[age_col].fillna(df_analysis[age_col].mean())
    df_analysis[ind_col] = df_analysis[ind_col].fillna('Other')
    
    # Feature Eng
    top_ind = df_analysis[ind_col].value_counts().nlargest(8).index.tolist()
    df_analysis['Industry_Expanded'] = df_analysis[ind_col].apply(lambda x: x if x in top_ind else 'Other')
    
    # Region Mapping
    us_regions = {
        'Northeast': ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont', 'New Jersey', 'New York', 'Pennsylvania'],
        'Midwest': ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin', 'Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota'],
        'South': ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia', 'District of Columbia', 'West Virginia', 'Alabama', 'Kentucky', 'Mississippi', 'Tennessee', 'Arkansas', 'Louisiana', 'Oklahoma', 'Texas'],
        'West': ['Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Utah', 'Wyoming', 'Alaska', 'California', 'Hawaii', 'Oregon', 'Washington']
    }
    def get_region(row):
        c = row[ctry_col]
        s = row[state_col]
        if str(c).strip() != 'United States': return 'International'
        if pd.isna(s): return 'US_Unknown'
        s = str(s).strip()
        for r, states in us_regions.items():
            if s in states: return f"US_{r}"
        return 'US_Other'

    df_analysis['Region_Detailed'] = df_analysis.apply(get_region, axis=1)
    
    scaler = StandardScaler()
    df_analysis['Age_Std'] = scaler.fit_transform(df_analysis[[age_col]])
    
    df_encoded = pd.get_dummies(df_analysis, columns=['Industry_Expanded', 'Region_Detailed'], drop_first=True)
    df_encoded.columns = [c.replace('/', '_').replace(' ', '_').replace('-', '_') for c in df_encoded.columns]
    
    season_max_weeks = df_pred.groupby('Season')['Week'].max()
    df_encoded['Season_Max'] = df_encoded['Season'].map(season_max_weeks)
    df_encoded['Is_Eliminated'] = np.where(df_encoded['Week'] < df_encoded['Season_Max'], 1, 0)
    
    predictors = ['Age_Std'] + [c for c in df_encoded.columns if 'Industry_Expanded_' in c] + [c for c in df_encoded.columns if 'Region_Detailed_' in c]
    
    model_cols = ['Judge_Score', 'Est_Fan_Vote', 'Week', 'Is_Eliminated'] + predictors
    df_model = df_encoded[model_cols].copy().astype(float).dropna()
    df_model = df_model[df_model['Week'] > 0]
    
    print(f"Modeling Rows: {len(df_model)}")
    formula = " + ".join(predictors)
    
    print("Running OLS Judge...")
    model_judge = smf.ols(f'Judge_Score ~ {formula}', data=df_model).fit()
    
    print("Running OLS Fan...")
    model_fan = smf.ols(f'Est_Fan_Vote ~ {formula}', data=df_model).fit()
    
    print("Running Cox PH...")
    # Use formula API for safer name handling in results
    # Status variable needs to be in data
    try:
        cox_model = PHReg(df_model['Week'], df_model[predictors], status=df_model['Is_Eliminated']).fit()
        # Convert params to Series if not already
        if not isinstance(cox_model.params, pd.Series):
             cox_params = pd.Series(cox_model.params, index=predictors) # Assume order matches if no drops
        else:
             cox_params = cox_model.params
    except:
        cox_params = pd.Series()

    print("Running RF...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(df_model[predictors], df_model['Week'])
    
    results = []
    for idx, feat in enumerate(predictors):
        # Use safe extraction
        c_hr = np.nan
        if feat in cox_params.index:
             c_hr = np.exp(cox_params[feat])
        
        row = {
            'Feature': feat,
            'Judge_Coef': safe_get_param(model_judge, feat),
            'Fan_Coef': safe_get_param(model_fan, feat),
            'Cox_HR': c_hr,
            'RF_Importance': rf.feature_importances_[idx]
        }
        results.append(row)
        
    final_df = pd.DataFrame(results).round(4)
    print("\n=== Detailed Feature Impact Analysis ===")
    print(final_df[['Feature', 'Judge_Coef', 'Fan_Coef', 'Cox_HR', 'RF_Importance']].to_string(index=False))
    
    final_df.to_csv('feature_impact_analysis_detailed.csv', index=False)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()