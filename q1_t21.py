import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置与加载
# ==========================================
INPUT_PRED = 'all_seasons_final_prediction_with_bounds.csv'
INPUT_META = 'cleaned_DWTS_data.csv'
OUTPUT_FILE = 'feature_augmented_fan_votes.csv'

def load_data():
    try:
        df_pred = pd.read_csv(INPUT_PRED)
        df_meta = pd.read_csv(INPUT_META)
        
        # Standardize Names
        if 'celebrity_name' in df_meta.columns:
            df_meta = df_meta.rename(columns={'celebrity_name': 'Contestant', 'season': 'Season'})
        
        df_meta['Contestant_Norm'] = df_meta['Contestant'].astype(str).str.strip().str.lower()
        df_pred['Contestant_Norm'] = df_pred['Contestant'].astype(str).str.strip().str.lower()
        
        return df_pred, df_meta
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# ==========================================
# 2. 特征工程
# ==========================================
def prepare_features(df_pred, df_meta):
    # Find Age/Industry cols
    age_col = [c for c in df_meta.columns if 'age' in c.lower()][0]
    ind_col = [c for c in df_meta.columns if 'industry' in c.lower()][0]
    
    # Merge
    df_merged = df_pred.merge(
        df_meta[['Contestant_Norm', 'Season', age_col, ind_col]],
        on=['Contestant_Norm', 'Season'],
        how='left'
    )
    
    # Cleaning
    df_merged[age_col] = pd.to_numeric(df_merged[age_col], errors='coerce').fillna(35) # Fill avg age
    df_merged[ind_col] = df_merged[ind_col].fillna('Other')
    
    # Top Industries
    top_n = df_merged[ind_col].value_counts().nlargest(5).index.tolist()
    df_merged['Industry_Clean'] = df_merged[ind_col].apply(lambda x: x if x in top_n else 'Other')
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_merged, columns=['Industry_Clean'], drop_first=False) # Keep all for regression
    
    # Features List
    ind_features = [c for c in df_encoded.columns if 'Industry_Clean_' in c]
    features = ['Judge_Score', age_col] + ind_features
    
    # Fill NAs in features
    for f in features:
        df_encoded[f] = df_encoded[f].fillna(0)
        
    return df_encoded, features

# ==========================================
# 3. 核心算法: 特征增强型 EM (Latent Variable Regression)
# ==========================================

def check_constraint_rank(j_scores, f_votes, loser_idx):
    """Rank Rule Constraint Check"""
    # Max Total = Loser
    # Rank: 1=Best. rankdata 'min'.
    j_ranks = stats.rankdata(-j_scores, method='min')
    f_ranks = stats.rankdata(-f_votes, method='min')
    total = j_ranks + f_ranks
    
    # Check if Loser has Max Total
    max_val = np.max(total)
    candidates = np.where(total == max_val)[0]
    
    # Tie-breaker: Worse Fan Rank (High Number) loses
    if len(candidates) > 1:
        # Check if loser is among candidates
        if loser_idx not in candidates: return False
        
        # Check tie-breaker
        f_ranks_sub = f_ranks[candidates]
        worst_f = np.max(f_ranks_sub)
        real_loser_f_rank = f_ranks[loser_idx]
        
        if real_loser_f_rank == worst_f:
            return True
        return False
    else:
        return (candidates[0] == loser_idx)

def check_constraint_percent(j_scores, f_votes, loser_idx):
    """Percent Rule Constraint Check"""
    # Min Total = Loser
    j_sum = np.sum(j_scores)
    j_pct = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
    total = j_pct + f_votes
    
    min_idx = np.argmin(total)
    return (min_idx == loser_idx)

def sample_constrained_votes(group_df, prior_votes, rule, loser_name, n_samples=50):
    """E-Step: Sample votes using Prior as mean, filtering by constraint"""
    n = len(group_df)
    names = group_df['Contestant'].values
    j_scores = group_df['Judge_Score'].values
    
    try:
        loser_idx = np.where(names == loser_name)[0][0]
    except:
        return prior_votes # No loser found (e.g., Final), return prior
        
    valid_samples = []
    
    # Adaptive Sampling
    # Prior is our best guess from regression.
    # We sample Dirichlet around this prior.
    # Alpha = Prior * Strength. Higher Strength = More confidence in Regression.
    strength = 50 
    alpha = prior_votes * strength + 1 # +1 to avoid zeros
    
    for _ in range(n_samples * 2): # Try more to get enough valid
        sample = np.random.dirichlet(alpha)
        
        # Check Constraint
        if rule == 'RANK':
            is_valid = check_constraint_rank(j_scores, sample, loser_idx)
        else:
            is_valid = check_constraint_percent(j_scores, sample, loser_idx)
            
        if is_valid:
            valid_samples.append(sample)
            if len(valid_samples) >= n_samples: break
            
    if not valid_samples:
        # Fallback: Just return Prior (soft constraint failure)
        # Or force constraint? Let's just return Prior to avoid crash
        return prior_votes
        
    return np.mean(valid_samples, axis=0)

def run_em_algorithm(df, features, iterations=3):
    """
    Expectation-Maximization Loop:
    1. Train Regression (Votes ~ Features)
    2. Predict Priors
    3. Sample Posteriors (Constrained)
    4. Update Votes
    """
    
    # Initialize Votes: Uniform
    # We need a column 'Current_Est_Vote'
    # Start with what we have from Q1 as a good seed, OR uniform?
    # Let's start with Q1 results as 'Current_Est_Vote' to speed up convergence
    if 'Est_Fan_Vote' in df.columns:
        df['Current_Est_Vote'] = df['Est_Fan_Vote']
    else:
        # Uniform init if Q1 not available
        df['Current_Est_Vote'] = df.groupby(['Season', 'Week'])['Judge_Score'].transform(lambda x: 1.0 / len(x))
        
    print(f"Starting EM Algorithm ({iterations} iterations)...")
    
    for i in range(iterations):
        print(f"  Iteration {i+1}...")
        
        # --- M-Step: Learn Impact of Features ---
        # Train Regressor: Current_Est_Vote ~ Judge_Score + Age + Industry
        # We use Random Forest for non-linearity
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        X = df[features]
        y = df['Current_Est_Vote']
        rf.fit(X, y)
        
        # Predict Prior for next step
        df['Prior_Vote'] = rf.predict(X)
        
        # Normalize Priors (Must sum to 1 per week)
        # This represents "Expected Vote based on Demographics/Score"
        df['Prior_Vote'] = df.groupby(['Season', 'Week'])['Prior_Vote'].transform(lambda x: x / x.sum())
        
        # --- E-Step: Update Estimates with Constraints ---
        new_votes_list = []
        
        # Process each week
        grouped = df.groupby(['Season', 'Week'])
        for (s, w), group in grouped:
            if len(group) < 2: 
                new_votes_list.extend(group['Prior_Vote'].values)
                continue
                
            # Identify Loser
            loser_row = group[group['Actual_Status'].astype(str).str.contains('Loser', case=False, na=False)]
            if loser_row.empty:
                # No elimination (e.g. Week 1 sometimes), trust Prior
                new_votes_list.extend(group['Prior_Vote'].values)
                continue
            
            loser_name = loser_row['Contestant'].iloc[0]
            rule = group['Rule'].iloc[0]
            priors = group['Prior_Vote'].values
            
            # Sample Posterior
            posterior = sample_constrained_votes(group, priors, rule, loser_name)
            new_votes_list.extend(posterior)
            
        # Update
        df['Current_Est_Vote'] = new_votes_list
        
    return df

# ==========================================
# 主流程
# ==========================================
def main():
    # 1. Load
    df_pred, df_meta = load_data()
    if df_pred is None: return
    
    # 2. Prepare
    print("Preparing features...")
    df_encoded, features = prepare_features(df_pred, df_meta)
    print(f"Features used: {features}")
    
    # 3. Run EM
    df_final = run_em_algorithm(df_encoded, features, iterations=3)
    
    # 4. Save
    # Keep key columns
    output_cols = ['Season', 'Week', 'Contestant', 'Judge_Score', 'Current_Est_Vote', 'Prior_Vote', 'Actual_Status', 'Rule'] + features
    df_final[output_cols].to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Algorithm Complete. Results saved to {OUTPUT_FILE}")
    print("Summary of New Predictions:")
    print(df_final[['Contestant', 'Judge_Score', 'Current_Est_Vote', 'Prior_Vote']].head())

if __name__ == "__main__":
    main()