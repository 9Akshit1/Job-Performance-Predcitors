import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
import re
import os
warnings.filterwarnings('ignore')

def clean_numeric_column(series, column_name):
    """
    Clean numeric columns by removing special characters and converting to float
    Handles cases like: '0.366', '~0.15', '<0.001', '>0.05', etc.
    """
    def clean_value(val):
        if pd.isna(val) or val == '':
            return np.nan
        
        # Convert to string and strip whitespace
        val_str = str(val).strip()
        
        # Remove special characters but keep the numeric part
        # This regex finds numbers (including decimals and scientific notation)
        numeric_match = re.search(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)', val_str)
        
        if numeric_match:
            try:
                return float(numeric_match.group(1))
            except ValueError:
                return np.nan
        else:
            return np.nan
    
    cleaned = series.apply(clean_value)
    print(f"Cleaned {column_name}: {cleaned.notna().sum()} valid values out of {len(series)}")
    return cleaned

# Create graphs directory if it doesn't exist
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# Read the CSV file
df = pd.read_csv(r'C:\Users\eruku\Akshith\AI_Internship_2025\Job_Performance_Predictors_Akshit_Erukulla - Final.csv')

# Clean numeric columns
df['Pearson_r_clean'] = clean_numeric_column(df['Pearson_r'], 'Pearson_r')
df['p_value_clean'] = clean_numeric_column(df['p_value'], 'p_value')
df['R_squared_clean'] = clean_numeric_column(df['R_squared'], 'R_squared')
df['Beta_weight_clean'] = clean_numeric_column(df['Beta_weight'], 'Beta_weight')
df['Odds_ratio_clean'] = clean_numeric_column(df['Odds_ratio'], 'Odds_ratio')

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Print unique predictors analysis
print("=== UNIQUE PREDICTORS ANALYSIS ===")
unique_predictors = df['Predictor'].unique()
print(f"Total unique predictors: {len(unique_predictors)}")
print("\nList of all unique predictors:")
for i, predictor in enumerate(sorted(unique_predictors), 1):
    print(f"{i:2d}. {predictor}")

# Simplify predictor names for duplicates/similar concepts
predictor_mapping = {
    'Emotional Intelligence (ability-based)': 'Emotional Intelligence',
    'Emotional Intelligence (self-report mixed)': 'Emotional Intelligence', 
    'Emotional Intelligence (self-report ability)': 'Emotional Intelligence',
    'Work Motivation': 'Motivation',
    'Autonomous Motivation': 'Motivation',
    'Controlled Motivation': 'Motivation',
    'EQ (Extraversion Competency)': 'Emotional Intelligence',
    'Self-Efficacy (moderated by Task Complexity)': 'Self-Efficacy',
    'Self-Efficacy (moderated by Goal Setting)': 'Self-Efficacy'
}

df['Predictor_Simplified'] = df['Predictor'].replace(predictor_mapping)
simplified_predictors = df['Predictor_Simplified'].unique()
print(f"\nAfter simplification: {len(simplified_predictors)} unique predictors")

# 1. Scatter plot of all predictors with correlation values
plt.figure(figsize=(12, 8))
df_with_r = df[df['Pearson_r_clean'].notna()]
if not df_with_r.empty:
    x_pos = range(len(df_with_r))
    plt.scatter(x_pos, df_with_r['Pearson_r_clean'], alpha=0.7, s=50)
    plt.title('Pearson Correlations by Predictor', fontsize=14, fontweight='bold')
    plt.ylabel('Pearson r')
    plt.xlabel('Predictor Index')
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('graphs/01_pearson_correlations_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Bar graph of categories
plt.figure(figsize=(12, 8))
category_counts = df['Category'].value_counts()
bars = plt.bar(range(len(category_counts)), category_counts.values)
plt.title('Distribution of Predictor Categories', fontsize=14, fontweight='bold')
plt.ylabel('Count')
plt.xlabel('Categories')
plt.xticks(range(len(category_counts)), category_counts.index, rotation=45, ha='right')
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             str(category_counts.values[i]), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('graphs/02_predictor_categories.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Bar graph comparing statistical measures
plt.figure(figsize=(12, 8))
stat_measures = ['Pearson_r_clean', 'p_value_clean', 'R_squared_clean', 'Beta_weight_clean', 'Odds_ratio_clean']
stat_labels = ['Pearson r', 'p-value', 'R²', 'Beta weight', 'Odds ratio']
stat_counts = [df[col].notna().sum() for col in stat_measures]
bars = plt.bar(stat_labels, stat_counts, color=['skyblue', 'lightgreen', 'salmon', 'gold', 'plum'])
plt.title('Availability of Statistical Measures', fontsize=14, fontweight='bold')
plt.ylabel('Count of Non-null Values')
plt.xticks(rotation=45, ha='right')
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(stat_counts[i]), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('graphs/03_statistical_measures_availability.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Pie chart of job domains
plt.figure(figsize=(10, 8))
job_domain_counts = df['Job_Domain'].value_counts()
plt.pie(job_domain_counts.values, labels=job_domain_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Job Domains', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('graphs/04_job_domains_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Pie chart of study types
plt.figure(figsize=(10, 8))
study_type_counts = df['Study_Type'].value_counts()
plt.pie(study_type_counts.values, labels=study_type_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Study Types', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('graphs/05_study_types_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Bar graph of measurement types
plt.figure(figsize=(12, 8))
measurement_counts = df['Measurement_Type'].value_counts()
bars = plt.bar(range(len(measurement_counts)), measurement_counts.values)
plt.title('Distribution of Measurement Types', fontsize=14, fontweight='bold')
plt.ylabel('Count')
plt.xlabel('Measurement Types')
plt.xticks(range(len(measurement_counts)), measurement_counts.index, rotation=45, ha='right')
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             str(measurement_counts.values[i]), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('graphs/06_measurement_types.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Scatter plot of sample sizes
plt.figure(figsize=(12, 8))
df['N_clean'] = df['N'].astype(str).str.replace(',', '').str.replace(' individuals', '')
df['N_clean'] = pd.to_numeric(df['N_clean'].str.extract('(\d+)')[0], errors='coerce')
df_with_n = df[df['N_clean'].notna() & (df['N_clean'] > 0)]

if not df_with_n.empty:
    plt.scatter(range(len(df_with_n)), df_with_n['N_clean'], alpha=0.7, s=50)
    plt.title('Sample Sizes Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Sample Size (N)')
    plt.xlabel('Study Index')
    plt.yscale('log')  # Log scale due to wide range
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/07_sample_sizes_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Bar graph of years
plt.figure(figsize=(14, 8))
df['Year_clean'] = pd.to_numeric(df['Year'], errors='coerce')
df_with_year = df[df['Year_clean'].notna()]
year_counts = df_with_year['Year_clean'].value_counts().sort_index()

bars = plt.bar(year_counts.index, year_counts.values)
plt.title('Distribution of Study Years', fontsize=14, fontweight='bold')
plt.ylabel('Count')
plt.xlabel('Year')
plt.xticks(rotation=45)
for i, (year, count) in enumerate(year_counts.items()):
    plt.text(year, count + 0.1, str(count), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('graphs/08_study_years.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Correlation magnitude distribution
plt.figure(figsize=(12, 8))
df_corr = df[df['Pearson_r_clean'].notna()]
if not df_corr.empty:
    correlation_ranges = pd.cut(df_corr['Pearson_r_clean'].abs(), 
                              bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0], 
                              labels=['Very Weak\n(0-0.1)', 'Weak\n(0.1-0.3)', 
                                     'Moderate\n(0.3-0.5)', 'Strong\n(0.5-0.7)', 
                                     'Very Strong\n(0.7-1.0)'])
    range_counts = correlation_ranges.value_counts().sort_index()
    bars = plt.bar(range(len(range_counts)), range_counts.values, 
                   color=['lightcoral', 'gold', 'lightgreen', 'dodgerblue', 'purple'])
    plt.title('Distribution of Correlation Strengths', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    plt.xlabel('Correlation Strength')
    plt.xticks(range(len(range_counts)), range_counts.index, rotation=45, ha='right')
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 str(range_counts.values[i]), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('graphs/09_correlation_strengths.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. Predictor frequency (simplified)
plt.figure(figsize=(12, 10))
predictor_counts = df['Predictor_Simplified'].value_counts().head(15)
bars = plt.barh(range(len(predictor_counts)), predictor_counts.values)
plt.title('Top 15 Most Studied Predictors', fontsize=14, fontweight='bold')
plt.xlabel('Count')
plt.yticks(range(len(predictor_counts)), predictor_counts.index)
for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
             str(predictor_counts.values[i]), ha='left', va='center')
plt.tight_layout()
plt.savefig('graphs/10_top_predictors.png', dpi=300, bbox_inches='tight')
plt.close()

# 11. Word cloud of sample context and notes
plt.figure(figsize=(12, 8))
text_data = []
for _, row in df.iterrows():
    if pd.notna(row['Sample_Context']):
        text_data.append(str(row['Sample_Context']))
    if pd.notna(row['Notes']):
        text_data.append(str(row['Notes']))

if text_data:
    all_text = ' '.join(text_data)
    common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                   'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                   'have', 'has', 'had', 'from', 'study', 'studies', 'analysis', 'research'}
    
    try:
        wordcloud = WordCloud(width=800, height=600, background_color='white',
                             stopwords=common_words, max_words=100).generate(all_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Context & Notes', fontsize=14, fontweight='bold')
    except Exception as e:
        plt.text(0.5, 0.5, f'WordCloud error: {str(e)}', ha='center', va='center', 
                transform=plt.gca().transAxes)
        plt.title('Word Cloud - Error', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'No text data available', ha='center', va='center', 
             transform=plt.gca().transAxes)
    plt.title('Word Cloud - No Data', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('graphs/11_wordcloud_context_notes.png', dpi=300, bbox_inches='tight')
plt.close()

# 12. Additional visualization: Correlation vs Sample Size
plt.figure(figsize=(12, 8))
df_scatter = df[(df['Pearson_r_clean'].notna()) & (df['N_clean'].notna()) & (df['N_clean'] > 0)]
if not df_scatter.empty:
    plt.scatter(df_scatter['N_clean'], df_scatter['Pearson_r_clean'].abs(), alpha=0.6, s=50)
    plt.title('Correlation Magnitude vs Sample Size', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Size (N)')
    plt.ylabel('|Pearson r|')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/12_correlation_vs_sample_size.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations have been saved to the 'graphs' folder!")

# Summary statistics
print("\n=== DATASET SUMMARY ===")
print(f"Total number of entries: {len(df)}")
print(f"Unique predictors (original): {len(unique_predictors)}")
print(f"Unique predictors (simplified): {len(simplified_predictors)}")
print(f"Unique categories: {len(df['Category'].unique())}")
print(f"Unique job domains: {len(df['Job_Domain'].unique())}")
print(f"Year range: {df['Year_clean'].min():.0f} - {df['Year_clean'].max():.0f}")
print(f"Sample size range: {df['N_clean'].min():.0f} - {df['N_clean'].max():.0f}")

# Statistical measures availability
print("\n=== STATISTICAL MEASURES AVAILABILITY ===")
measure_pairs = [
    ('Pearson_r_clean', 'Pearson r'),
    ('p_value_clean', 'p-value'),
    ('R_squared_clean', 'R²'),
    ('Beta_weight_clean', 'Beta weight'),
    ('Odds_ratio_clean', 'Odds ratio')
]

for col, label in measure_pairs:
    count = df[col].notna().sum()
    percentage = (count / len(df)) * 100
    print(f"{label}: {count} entries ({percentage:.1f}%)")

print("\n=== TOP CATEGORIES ===")
print(df['Category'].value_counts())

print("\n=== SIMPLIFIED PREDICTORS LIST ===")
for i, predictor in enumerate(sorted(simplified_predictors), 1):
    print(f"{i:2d}. {predictor}")

# Additional analysis: Show some examples of the special characters found
print("\n=== SPECIAL CHARACTER EXAMPLES ===")
print("Original Pearson_r values with special characters:")
special_chars = df[df['Pearson_r'].notna()]['Pearson_r'].astype(str)
special_examples = special_chars[special_chars.str.contains(r'[~<>≤≥]', na=False)].head(10)
for i, example in enumerate(special_examples):
    print(f"  {i+1}. '{example}'")

print("\nOriginal p_value values with special characters:")
p_special = df[df['p_value'].notna()]['p_value'].astype(str)
p_examples = p_special[p_special.str.contains(r'[~<>≤≥]', na=False)].head(10)
for i, example in enumerate(p_examples):
    print(f"  {i+1}. '{example}'")

print(f"\nAll graphs saved successfully in the 'graphs' folder:")
graph_files = [
    "01_pearson_correlations_scatter.png",
    "02_predictor_categories.png", 
    "03_statistical_measures_availability.png",
    "04_job_domains_pie.png",
    "05_study_types_pie.png",
    "06_measurement_types.png",
    "07_sample_sizes_scatter.png",
    "08_study_years.png",
    "09_correlation_strengths.png",
    "10_top_predictors.png",
    "11_wordcloud_context_notes.png",
    "12_correlation_vs_sample_size.png"
]

for i, filename in enumerate(graph_files, 1):
    print(f"{i:2d}. {filename}")