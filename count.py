import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv(r'C:\Users\eruku\Akshith\AI_Internship_2025\Job_Performance_Predictors_Akshit_Erukulla - Final.csv')

print("=== STUDY COUNT ANALYSIS ===")
print(f"Total number of entries in dataset: {len(df)}")

# Method 1: Check if there's a direct study identifier column
study_id_columns = [col for col in df.columns if 'study' in col.lower() or 'id' in col.lower()]
print(f"\nPotential study identifier columns found: {study_id_columns}")

# Method 2: Count unique combinations of key study characteristics
# This assumes that each unique combination represents a different study
study_characteristics = ['Year', 'N', 'Sample_Context', 'Job_Domain', 'Study_Type']

# Check which characteristics are available
available_chars = [col for col in study_characteristics if col in df.columns]
print(f"Available study characteristics: {available_chars}")

if available_chars:
    # Create a composite key from available characteristics
    df_study_key = df[available_chars].copy()
    
    # Clean the data for better matching
    for col in available_chars:
        if col == 'N':
            # Clean sample sizes
            df_study_key[col] = df_study_key[col].astype(str).str.replace(',', '').str.replace(' individuals', '')
            df_study_key[col] = pd.to_numeric(df_study_key[col].str.extract('(\d+)')[0], errors='coerce')
        elif col == 'Year':
            df_study_key[col] = pd.to_numeric(df_study_key[col], errors='coerce')
        else:
            # For text columns, standardize formatting
            df_study_key[col] = df_study_key[col].astype(str).str.strip().str.lower()
    
    # Count unique combinations
    unique_studies = df_study_key.drop_duplicates()
    estimated_studies = len(unique_studies)
    
    print(f"\nEstimated number of unique studies (Method 1): {estimated_studies}")
    print(f"Based on unique combinations of: {', '.join(available_chars)}")

# Method 3: Count unique combinations of Year + Sample Size (more conservative)
if 'Year' in df.columns and 'N' in df.columns:
    df_conservative = df[['Year', 'N']].copy()
    
    # Clean Year and N
    df_conservative['Year_clean'] = pd.to_numeric(df_conservative['Year'], errors='coerce')
    df_conservative['N_clean'] = df_conservative['N'].astype(str).str.replace(',', '').str.replace(' individuals', '')
    df_conservative['N_clean'] = pd.to_numeric(df_conservative['N_clean'].str.extract('(\d+)')[0], errors='coerce')
    
    # Remove rows with missing data
    df_conservative_clean = df_conservative.dropna()
    unique_year_n = df_conservative_clean[['Year_clean', 'N_clean']].drop_duplicates()
    conservative_estimate = len(unique_year_n)
    
    print(f"\nEstimated number of unique studies (Method 2 - Conservative): {conservative_estimate}")
    print("Based on unique combinations of Year + Sample Size")

# Method 4: If there's a DOI, URL, or citation column
citation_columns = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['doi', 'url', 'citation', 'reference', 'source', 'author'])]

if citation_columns:
    print(f"\nFound potential citation/source columns: {citation_columns}")
    
    for col in citation_columns:
        unique_citations = df[col].dropna().nunique()
        print(f"Unique values in '{col}': {unique_citations}")

# Method 5: Analyze patterns in the data
print(f"\n=== ADDITIONAL ANALYSIS ===")

# Show distribution of entries per year
if 'Year' in df.columns:
    df['Year_clean'] = pd.to_numeric(df['Year'], errors='coerce')
    year_counts = df['Year_clean'].value_counts().sort_index()
    print(f"\nEntries per year:")
    for year, count in year_counts.items():
        if not pd.isna(year):
            print(f"  {int(year):4d}: {count:2d} entries")

# Show distribution of entries per sample size
if 'N' in df.columns:
    df['N_clean'] = df['N'].astype(str).str.replace(',', '').str.replace(' individuals', '')
    df['N_clean'] = pd.to_numeric(df['N_clean'].str.extract('(\d+)')[0], errors='coerce')
    
    sample_size_counts = df['N_clean'].value_counts().sort_index().head(20)
    print(f"\nMost common sample sizes (top 20):")
    for n, count in sample_size_counts.items():
        if not pd.isna(n):
            print(f"  N={int(n):4d}: {count:2d} entries")

# Method 6: Check for repeated combinations that might indicate same study
print(f"\n=== DUPLICATE DETECTION ===")

if 'Year' in df.columns and 'N' in df.columns:
    # Find cases where same year and sample size appear multiple times
    year_n_counts = df.groupby(['Year', 'N']).size().reset_index(name='count')
    duplicates = year_n_counts[year_n_counts['count'] > 1].sort_values('count', ascending=False)
    
    if not duplicates.empty:
        print("Potential same studies (same Year + Sample Size):")
        print(duplicates.head(10).to_string(index=False))
    else:
        print("No obvious duplicates found based on Year + Sample Size")

# Summary
print(f"\n=== SUMMARY ===")
print(f"Total dataset entries: {len(df)}")

if 'available_chars' in locals() and available_chars:
    print(f"Estimated unique studies (comprehensive): {estimated_studies}")

if 'conservative_estimate' in locals():
    print(f"Estimated unique studies (conservative): {conservative_estimate}")

print(f"\nNote: These are estimates based on available data.")
print("The actual number of unique studies may differ depending on:")
print("- Whether the same study contributed multiple effect sizes")
print("- Whether studies were split across multiple rows")
print("- Data quality and consistency in recording")