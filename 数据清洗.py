import pandas as pd
import numpy as np
import re

def clean_dwts_data(input_file, output_file):
    # 1. 加载数据 [cite: 82, 92]
    df = pd.read_csv(input_file)

    # 2. 行业类别标准化与合并 (Industry Mapping)
    # 基础清理：统一大小写并去除空格
    df['celebrity_industry'] = df['celebrity_industry'].str.strip().str.title()

    # 核心映射逻辑：合并相近类别，突出 Digital Creator
    industry_mapping = {
        # 体育类合并
        'Nfl Player': 'Athlete', 'Olympian': 'Athlete', 'Race Car Driver': 'Athlete',
        'Ice Skater': 'Athlete', 'Ufc Fighter': 'Athlete', 'Racing Driver': 'Athlete',
        
        # 演艺类合并
        'Disney Star': 'Actor/Actress', 'Marvel Star': 'Actor/Actress', 'Actor': 'Actor/Actress',
        
        # 音乐类合并
        'Country Singer': 'Singer/Rapper', 'Pop Star': 'Singer/Rapper', 'Music Legend': 'Singer/Rapper', 
        'Dj': 'Singer/Rapper', 'Musician': 'Singer/Rapper',
        
        # 互联网/社交媒体 -> 统一合并为 Digital Creator
        'Social Media Personality': 'Digital Creator',
        'Social Media Star': 'Digital Creator',
        'Influencer': 'Digital Creator',
        'Blogger': 'Digital Creator',
        
        # 真人秀类合并
        'Reality Star': 'Reality Tv', 'Reality Tv Star': 'Reality Tv', 'Bravo Star': 'Reality Tv',
        
        # 媒体与主持类合并
        'Talk Show Host': 'Tv/Radio Host', 'Tv Host': 'Tv/Radio Host', 'Radio Personality': 'Tv/Radio Host',
        'Journalist': 'Media/Writing', 'Author': 'Media/Writing', 'News Anchor': 'Media/Writing',
        'Sports Broadcaster': 'Media/Writing'
    }
    df['celebrity_industry'] = df['celebrity_industry'].replace(industry_mapping)

    # 3. 解析结果列 (Extract Elimination Week) [cite: 109, 120]
    def extract_elim_week(res_str):
        if pd.isna(res_str): return np.nan
        # 匹配 "Week X"
        match = re.search(r'Week\s+(\d+)', str(res_str))
        if match: return int(match.group(1))
        return np.nan

    df['eliminated_week'] = df['results'].apply(extract_elim_week)
    df['withdrew'] = df['results'].str.contains('Withdrew', case=False, na=False)

    # 4. 每周评委分数处理与聚合 [cite: 98, 103, 118]
    score_cols = [col for col in df.columns if 'score' in col]
    # 提取所有存在的周数 (通常为 1-11)
    weeks_found = sorted(list(set([re.search(r'week(\d+)', col).group(1) for col in score_cols])))

    for w in weeks_found:
        w_cols = [col for col in score_cols if f'week{w}_' in col]
        
        # 计算该选手的每周平均分 (自动忽略 NaN) [cite: 103]
        df[f'avg_score_w{w}'] = df[w_cols].mean(axis=1)
        
        # 计算该选手的每周总分
        df[f'total_score_w{w}'] = df[w_cols].sum(axis=1, min_count=1)

        # 核心增强：计算“当周全场活跃选手的总分”，用于计算“百分比法”的分母 
        # 逻辑：在该周分数 > 0 的选手才是当周在场并参与投票的
        df[f'pool_total_score_w{w}'] = df.groupby('season')[f'total_score_w{w}'].transform(lambda x: x[x > 0].sum())
        
        # 计算该选手在该周的评委得分百分比 (Judges Score Percent)
        df[f'judge_pct_w{w}'] = df[f'total_score_w{w}'] / df[f'pool_total_score_w{w}']

    # 5. 特征工程 (Age & Origin)
    # 年龄分段 [cite: 83]
    df['age_group'] = pd.cut(df['celebrity_age_during_season'],
                             bins=[0, 25, 40, 60, 100],
                             labels=['Young', 'Adult', 'Middle-Aged', 'Senior'])

    # 地区属性分类
    def categorize_origin(row):
        country = str(row['celebrity_homecountry/region']).strip().title()
        if any(keyword in country for keyword in ['United States', 'U.S.', 'Us']):
            return 'Domestic'
        elif pd.isna(row['celebrity_homecountry/region']):
            return 'Unknown'
        return 'International'

    df['origin_type'] = df.apply(categorize_origin, axis=1)

    # 6. 保存清洗后的数据
    df.to_csv(output_file, index=False)
    print(f"数据清洗完成！清洗后的数据已保存至: {output_file}")
    
    # 打印一些关键统计信息供检查
    print("\n--- 行业分布情况 ---")
    print(df['celebrity_industry'].value_counts().head(10))

# 运行清洗
if __name__ == "__main__":
    clean_dwts_data('2026_MCM_Problem_C_Data.csv', 'cleaned_DWTS_data.csv')