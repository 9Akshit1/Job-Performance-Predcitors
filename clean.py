import pandas as pd
import numpy as np
import re

# Read the CSV file with error handling
try:
    # Try reading with different parameters to handle parsing issues
    df = pd.read_csv(r'C:\Users\eruku\Akshith\AI_Internship_2025\proper_cl\final.csv', 
                     quoting=1,  # QUOTE_ALL
                     escapechar='\\',
                     on_bad_lines='skip')
    print(f"Successfully loaded {len(df)} rows")
except Exception as e:
    print(f"Error reading CSV: {e}")
    # Try alternative approach
    df = pd.read_csv('final.csv', 
                     sep=',',
                     quotechar='"',
                     on_bad_lines='skip',
                     encoding='utf-8')

# Display basic info about the dataset
print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# Check for parsing issues
print("\nFirst few rows:")
print(df.head())

# Analyze the Predictor and Category columns
print("\nUnique Predictors:")
predictors = df['Predictor'].unique()
for i, pred in enumerate(predictors):
    print(f"{i+1}. {pred}")

print("\nUnique Categories:")
categories = df['Category'].unique()
for i, cat in enumerate(categories):
    print(f"{i+1}. {cat}")

# Create simplified mapping dictionaries
predictor_mapping = {
    # Personality traits
    'Achievement': 'Achievement',
    'Customer Orientation': 'Customer_Orientation',
    'Innovative': 'Innovation',
    'Open to Change': 'Adaptability',
    'Self Development': 'Self_Development',
    'Agreeableness': 'Agreeableness',
    'Conscientiousness': 'Conscientiousness',
    'Extraversion': 'Extraversion',
    'Neuroticism': 'Neuroticism',
    'Openness to Experience': 'Openness',
    'Proactive Personality': 'Proactivity',
    'Drive': 'Drive',
    'Adaptability': 'Adaptability',
    
    # Cognitive abilities
    'Intelligence': 'Intelligence',
    'General Mental Ability (GMA)': 'General_Mental_Ability',
    'Cognitive Ability': 'Cognitive_Ability',
    'Situational Judgment Test (SJT)': 'Situational_Judgment',
    'Judgment': 'Judgment',
    
    # Emotional Intelligence
    'Self Confidence': 'Self_Confidence',
    'Emotional Intelligence': 'Emotional_Intelligence',
    'Emotional Intelligence (ability-based)': 'Emotional_Intelligence',
    'Emotional Intelligence (self-report mixed)': 'Emotional_Intelligence',
    'Emotional Intelligence (self-report ability)': 'Emotional_Intelligence',
    'EQ (Extraversion Competency)': 'Emotional_Intelligence',
    'Social Intelligence': 'Social_Intelligence',
    
    # Motivation
    'Motivation': 'Motivation',
    'Work Motivation': 'Work_Motivation',
    'Autonomous Motivation': 'Autonomous_Motivation',
    'Controlled Motivation': 'Controlled_Motivation',
    'Self-Efficacy': 'Self_Efficacy',
    'Self-Efficacy (moderated by Task Complexity)': 'Self_Efficacy',
    'Self-Efficacy (moderated by Goal Setting)': 'Self_Efficacy',
    'Achievement Orientation': 'Achievement_Orientation',
    'job_satisfaction': 'Job_Satisfaction',
    'Job Satisfaction': 'Job_Satisfaction',
    
    # Skills
    'Multi-Tasking': 'Multi_Tasking',
    'Collaboration': 'Collaboration',
    'Collaborative Leadership': 'Collaborative_Leadership',
    'Research & Collaboration (interaction)': 'Research_Collaboration',
    'Teamwork': 'Teamwork',
    'Leadership': 'Leadership',
    
    # Stress factors (negative predictors)
    'Technostress': 'Technostress',
    'Technological Uncertainty': 'Tech_Uncertainty',
    'Technological Insecurity': 'Tech_Insecurity',
    'Technological Complexity': 'Tech_Complexity',
    'Technology Invasion': 'Tech_Invasion',
    'Technology-Induced Overload': 'Tech_Overload',
    'Workload': 'Workload',
    'Procedural Injustice': 'Procedural_Injustice',
    'Role Ambiguity': 'Role_Ambiguity',
    'Work-family Conflict': 'Work_Family_Conflict',
    'Physical Environment': 'Physical_Environment',
    'Job Burnout': 'Job_Burnout',
    
    # Character strengths
    'Perseverance': 'Perseverance',
    'Kindness': 'Kindness',
    
    # Other
    'Potential for Promotion': 'Promotion_Potential',
    'Structured Interview': 'Structured_Interview',
    'Education': 'Education',
    'Integrity Tests': 'Integrity',
    'Job Performance': 'Job_Performance'
}

category_mapping = {
    'Personality': 'Personality',
    'Cognitive': 'Cognitive',
    'Emotional Intelligence': 'Emotional_Intelligence',
    'Emotional': 'Emotional_Intelligence',
    'Motivation': 'Motivation',
    'Skills': 'Skills',
    'Social/Interpersonal': 'Social_Skills',
    'Stress': 'Stress_Factors',
    'Leadership': 'Leadership',
    'Character Strength': 'Character_Strengths',
    'Personality/Motivation': 'Personality',
    'Personality/Resilience': 'Personality',
    'Interview_Performance': 'Assessment_Methods',
    'Interview': 'Assessment_Methods',
    'Education': 'Background_Factors',
    'Outcome': 'Outcomes',
    'attitude': 'Attitudes'
}

# Apply the mappings
df['Predictor_Clean'] = df['Predictor'].map(predictor_mapping)
df['Category_Clean'] = df['Category'].map(category_mapping)

# Handle any unmapped values
unmapped_predictors = df[df['Predictor_Clean'].isna()]['Predictor'].unique()
unmapped_categories = df[df['Category_Clean'].isna()]['Category'].unique()

if len(unmapped_predictors) > 0:
    print(f"\nUnmapped predictors: {unmapped_predictors}")
    # Fill with original values for unmapped items
    df['Predictor_Clean'] = df['Predictor_Clean'].fillna(df['Predictor'])

if len(unmapped_categories) > 0:
    print(f"\nUnmapped categories: {unmapped_categories}")
    # Fill with original values for unmapped items
    df['Category_Clean'] = df['Category_Clean'].fillna(df['Category'])

# Replace original columns with cleaned versions
df['Predictor'] = df['Predictor_Clean']
df['Category'] = df['Category_Clean']
df = df.drop(['Predictor_Clean', 'Category_Clean'], axis=1)

# Clean up other columns that might cause issues
# Fix the N column - extract numeric values properly
df['N_clean'] = df['N'].astype(str)
df['N_clean'] = df['N_clean'].str.replace(',', '')  # Remove commas
df['N_clean'] = df['N_clean'].str.extract(r'(\d+)', expand=False)  # Fix regex warning
df['N_clean'] = pd.to_numeric(df['N_clean'], errors='coerce')

# Replace the original N column
df['N'] = df['N_clean']
df = df.drop('N_clean', axis=1)

# Clean up text fields that might have embedded quotes or commas
text_columns = ['Source_APA', 'Source_Link', 'Notes', 'Sample_Context']
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('"', '""')  # Escape quotes
        df[col] = df[col].str.replace('\n', ' ')  # Remove newlines
        df[col] = df[col].str.replace('\r', ' ')  # Remove carriage returns

# Save the cleaned dataset
df.to_csv('final_cleaned.csv', index=False, quoting=1)  # Use QUOTE_ALL to prevent parsing issues

print(f"\nCleaned dataset saved as 'final_cleaned.csv'")
print(f"Final dataset shape: {df.shape}")
print("\nCleaned Predictors:")
for pred in sorted(df['Predictor'].unique()):
    print(f"  - {pred}")
print("\nCleaned Categories:")
for cat in sorted(df['Category'].unique()):
    print(f"  - {cat}")