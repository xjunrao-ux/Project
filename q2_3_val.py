import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 1. Load Data
try:
    df = pd.read_csv('all_seasons_final_prediction_with_bounds.csv')
    print("Loaded all_seasons_final_prediction_with_bounds.csv")
except:
    try:
        df = pd.read_csv('all_seasons_final_prediction.csv')
        print("Loaded all_seasons_final_prediction.csv")
    except:
        print("Error: Input file not found.")
        df = None

# 2. Helper Functions
def get_rank_metrics(sub_df):
    j_ranks = stats.rankdata(-sub_df['Judge_Score'].values, method='min')
    f_ranks = stats.rankdata(-sub_df['Est_Fan_Vote'].values, method='min')
    total = j_ranks + f_ranks
    return total

def get_percent_metrics(sub_df):
    j_scores = sub_df['Judge_Score'].values
    j_sum = np.sum(j_scores)
    j_pct = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
    total = j_pct + sub_df['Est_Fan_Vote'].values
    return total

def analyze_season_28_hypothesis(df):
    if df is None: return

    results = []
    
    # Iterate through all weeks
    grouped = df.groupby(['Season', 'Week'])
    
    for (s, w), group in grouped:
        if len(group) < 2: continue
        
        # Check if there is an actual loser
        actual_loser_row = group[group['Actual_Status'].astype(str).str.contains('Loser', case=False, na=False)]
        if actual_loser_row.empty: continue
        actual_loser = actual_loser_row['Contestant'].iloc[0]
        
        # Current Rule
        rule = group['Rule'].iloc[0]
        
        # 1. Calculate Standard Loser
        if rule == 'RANK':
            total = get_rank_metrics(group)
            # Worst is MAX
            worst_val = np.max(total)
            candidates = np.where(total == worst_val)[0]
            # Tie-breaker: Worse Fan Rank (High number) loses
            if len(candidates) > 1:
                f_ranks = stats.rankdata(-group['Est_Fan_Vote'].values, method='min')
                sub_f = f_ranks[candidates]
                worst_f = np.max(sub_f)
                idx = candidates[np.where(sub_f == worst_f)[0][0]]
            else:
                idx = candidates[0]
            
            # Identify Bottom 2 (Largest Totals)
            sorted_idx = np.argsort(-total) # Descending
            bottom2_idx = sorted_idx[:2]
            
        else: # PERCENT
            total = get_percent_metrics(group)
            # Worst is MIN
            idx = np.argmin(total)
            
            # Identify Bottom 2 (Smallest Totals)
            sorted_idx = np.argsort(total) # Ascending
            bottom2_idx = sorted_idx[:2]
            
        standard_loser = group['Contestant'].iloc[idx]
        
        # 2. Calculate Judges Save Loser
        # Bottom 2 -> Lower Judge Score Eliminated
        p1_idx = bottom2_idx[0]
        p2_idx = bottom2_idx[1]
        
        j_scores = group['Judge_Score'].values
        names = group['Contestant'].values
        
        if j_scores[p1_idx] < j_scores[p2_idx]:
            save_loser = names[p1_idx]
        elif j_scores[p1_idx] > j_scores[p2_idx]:
            save_loser = names[p2_idx]
        else:
            save_loser = names[p1_idx] # Tie -> Original
            
        results.append({
            'Season': s,
            'Week': w,
            'Rule': rule,
            'Actual_Loser': actual_loser,
            'Standard_Loser': standard_loser,
            'Save_Loser': save_loser,
            'Match_Standard': (standard_loser == actual_loser),
            'Match_Save': (save_loser == actual_loser),
            'Is_Rescued_By_Save': (standard_loser == actual_loser) and (save_loser != actual_loser), # Standard was right, Save changed it to wrong (Bad Save?)
            'Is_Corrected_By_Save': (standard_loser != actual_loser) and (save_loser == actual_loser) # Standard was wrong, Save fixed it!
        })
        
    res_df = pd.DataFrame(results)
    
    # Aggregate by Season
    season_stats = res_df.groupby('Season').agg({
        'Match_Standard': 'mean',
        'Match_Save': 'mean',
        'Is_Corrected_By_Save': 'sum'
    }).reset_index()
    
    season_stats['Standard_Acc'] = round(season_stats['Match_Standard'] * 100, 1)
    season_stats['Save_Acc'] = round(season_stats['Match_Save'] * 100, 1)
    
    print("\n=== Season Accuracy Comparison (Standard vs With Save) ===")
    print(season_stats[['Season', 'Standard_Acc', 'Save_Acc', 'Is_Corrected_By_Save']].to_string(index=False))
    
    # Analyze S28 transition
    print("\n=== Focus: Season 27 vs 28+ ===")
    subset = season_stats[season_stats['Season'].isin([26, 27, 28, 29, 30])]
    print(subset[['Season', 'Standard_Acc', 'Save_Acc', 'Is_Corrected_By_Save']])

analyze_season_28_hypothesis(df)