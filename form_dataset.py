import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import csv
import warnings
warnings.filterwarnings('ignore')

# LLM and AI imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install: pip install transformers torch sentence-transformers")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    print("Please install: pip install spacy && python -m spacy download en_core_web_sm")

class IntelligentDataExtractor:
    def __init__(self):
        """Initialize the AI-powered data extractor with multiple models."""
        print("🤖 Initializing Intelligent Data Extractor with AI Models...")
        
        self.initialize_ai_models()
        self.setup_extraction_templates()
        self.setup_validation_rules()
        
        print("✅ AI Extractor ready!")
    
    def initialize_ai_models(self):
        """Initialize multiple AI models for different extraction tasks."""
        print("🔄 Loading AI models...")
        
        # Question-Answering model for targeted extraction
        try:
            self.qa_model = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                tokenizer="deepset/roberta-base-squad2"
            )
            print("✅ QA Model loaded")
        except:
            try:
                self.qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
                print("✅ Fallback QA Model loaded")
            except:
                print("⚠️ QA Model failed to load")
                self.qa_model = None
        
        # Summarization model for context understanding
        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            print("✅ Summarization Model loaded")
        except:
            try:
                self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                print("✅ Fallback Summarization Model loaded")
            except:
                print("⚠️ Summarization Model failed to load")
                self.summarizer = None
        
        # Sentence similarity for semantic matching
        try:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Similarity Model loaded")
        except:
            print("⚠️ Similarity Model failed to load")
            self.similarity_model = None
        
        # Text classification for content categorization
        try:
            self.classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium"
            )
            print("✅ Classification Model loaded")
        except:
            print("⚠️ Classification Model failed to load")
            self.classifier = None
        
        # Named Entity Recognition
        try:
            self.ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
            print("✅ NER Model loaded")
        except:
            print("⚠️ NER Model failed to load")
            self.ner_model = None
    
    def setup_extraction_templates(self):
        """Setup templates and questions for AI extraction."""
        
        # Questions for QA model to extract specific information
        self.qa_questions = {
            'predictor': [
                "What is the main predictor or variable being studied in relation to job performance?",
                "What factor is being tested as a predictor of job performance?",
                "What is the independent variable or predictor in this study?",
                "What characteristic or ability is being measured as a performance predictor?"
            ],
            'correlation': [
                "What is the correlation coefficient between the predictor and job performance?",
                "What is the Pearson correlation value reported in this study?",
                "What is the correlation value (r) found in the research?",
                "What correlation coefficient is reported for job performance prediction?"
            ],
            'p_value': [
                "What is the p-value or significance level reported?",
                "What is the statistical significance of the correlation?",
                "Is the relationship statistically significant and at what level?",
                "What is the probability value (p) reported?"
            ],
            'sample_size': [
                "What is the sample size (N) of the study?",
                "How many participants were included in the research?",
                "What is the number of subjects in the study?",
                "How large was the sample size?"
            ],
            'r_squared': [
                "What is the R-squared value or variance explained?",
                "What percentage of variance is explained by the predictor?",
                "What is the coefficient of determination (R²)?",
                "How much variance in job performance is explained?"
            ],
            'beta_weight': [
                "What is the beta weight or regression coefficient?",
                "What is the standardized beta coefficient reported?",
                "What is the regression weight for the predictor?",
                "What is the beta value in the regression analysis?"
            ],
            'study_type': [
                "What type of study is this (meta-analysis, correlation, experimental)?",
                "What research design was used in this study?",
                "Is this a meta-analysis, longitudinal study, or cross-sectional study?",
                "What methodology was employed in the research?"
            ],
            'job_domain': [
                "What job domain or industry was studied?",
                "What type of work or profession was the focus?",
                "What occupational field was examined?",
                "What industry or work context was studied?"
            ],
            'measurement_type': [
                "How was the predictor measured (self-report, objective test, etc.)?",
                "What type of measurement or assessment was used?",
                "Was this measured through tests, questionnaires, or observations?",
                "What measurement method was employed?"
            ],
            'year': [
                "What year was this study published or conducted?",
                "When was this research published?",
                "What is the publication year of this study?",
                "In what year was this research conducted?"
            ],
            'authors': [
                "Who are the authors of this study?",
                "What are the names of the researchers?",
                "Who conducted this research?",
                "What are the author names?"
            ]
        }
        
        # Patterns for statistical values
        self.statistical_patterns = {
            'correlation': [
                r'r\s*=\s*[-+]?[0-9]*\.?[0-9]+',
                r'correlation\s+(?:coefficient\s+)?(?:of\s+)?[-+]?[0-9]*\.?[0-9]+',
                r'pearson\s+(?:r\s*=\s*)?[-+]?[0-9]*\.?[0-9]+',
                r'correlation\s*[-+]?[0-9]*\.?[0-9]+',
                r'[-+]?0\.[0-9]+(?=.*correlation)',
                r'[-+]?0\.[0-9]+(?=.*job performance)'
            ],
            'p_value': [
                r'p\s*[<>=]\s*[0-9]*\.?[0-9]+',
                r'significance\s*[<>=]\s*[0-9]*\.?[0-9]+',
                r'sig\.\s*[<>=]\s*[0-9]*\.?[0-9]+',
                r'p-value\s*[<>=]\s*[0-9]*\.?[0-9]+',
                r'\.000(?=.*significance)',
                r'\.00[0-9](?=.*significant)'
            ],
            'sample_size': [
                r'[Nn]\s*=\s*[\d,]+',
                r'sample\s+size\s*=?\s*[\d,]+',
                r'participants\s*=?\s*[\d,]+',
                r'[\d,]+\s+(?:participants|subjects)',
                r'(?:total|sample)\s+of\s+[\d,]+',
                r'\b[2-9]\d{2,}\b(?=.*(?:sample|participant|subject))'
            ],
            'r_squared': [
                r'[Rr]²\s*=\s*[0-9]*\.?[0-9]+',
                r'[Rr]-squared\s*=\s*[0-9]*\.?[0-9]+',
                r'variance\s+explained\s*=?\s*[0-9]*\.?[0-9]+',
                r'explained\s+variance\s*=?\s*[0-9]*\.?[0-9]+'
            ],
            'beta_weight': [
                r'β\s*=\s*[-+]?[0-9]*\.?[0-9]+',
                r'beta\s*=\s*[-+]?[0-9]*\.?[0-9]+',
                r'beta\s+weight\s*=?\s*[-+]?[0-9]*\.?[0-9]+',
                r'regression\s+coefficient\s*=?\s*[-+]?[0-9]*\.?[0-9]+'
            ]
        }
        
        # Category mapping for predictors
        self.predictor_categories = {
            'cognitive': [
                'cognitive ability', 'intelligence', 'IQ', 'general mental ability', 'GMA',
                'reasoning', 'cognitive test', 'aptitude', 'mental capacity', 'intellectual',
                'cognitive skills', 'problem solving', 'analytical', 'verbal reasoning',
                'numerical reasoning', 'abstract reasoning', 'critical thinking'
            ],
            'personality': [
                'personality', 'big five', 'conscientiousness', 'extraversion', 'agreeableness',
                'neuroticism', 'openness', 'personality traits', 'temperament', 'character',
                'NEO-PI', 'emotional stability', 'personality test', 'personality assessment'
            ],
            'experience': [
                'experience', 'work experience', 'job experience', 'tenure', 'years of experience',
                'seniority', 'career length', 'employment history', 'professional experience'
            ],
            'education': [
                'education', 'educational level', 'degree', 'qualification', 'academic',
                'educational background', 'schooling', 'training', 'certification', 'GPA'
            ],
            'skills': [
                'skills', 'competencies', 'abilities', 'technical skills', 'soft skills',
                'job skills', 'professional skills', 'capabilities', 'expertise'
            ],
            'interview': [
                'interview', 'structured interview', 'unstructured interview', 'behavioral interview',
                'interview performance', 'interview rating', 'interview assessment'
            ],
            'integrity': [
                'integrity', 'integrity test', 'honesty', 'ethical behavior', 'moral character',
                'overt integrity', 'personality-based integrity'
            ],
            'leadership': [
                'leadership', 'leadership skills', 'management ability', 'supervisory skills',
                'leadership potential', 'leadership behavior', 'management competency'
            ],
            'technology': [
                'technology', 'technostress', 'technological', 'digital', 'computer',
                'technology stress', 'technological uncertainty', 'technology complexity'
            ]
        }
    
    def setup_validation_rules(self):
        """Setup validation rules for extracted data."""
        
        self.validation_rules = {
            'correlation': {
                'range': (-1.0, 1.0),
                'pattern': r'^[-+]?[0-1]?\.?\d*$',
                'common_invalid': ['1', '0', '-1', '']
            },
            'p_value': {
                'range': (0.0, 1.0),
                'pattern': r'^[0-1]?\.?\d*$',
                'common_values': ['0.001', '0.01', '0.05', '0.000']
            },
            'sample_size': {
                'range': (1, 1000000),
                'pattern': r'^\d+$',
                'min_reasonable': 10
            },
            'year': {
                'range': (1950, 2025),
                'pattern': r'^(19|20)\d{2}$'
            }
        }
    
    def load_data_files(self, json_path: str, txt_path: str) -> Tuple[Dict, str]:
        """Load and parse both JSON and TXT files."""
        print(f"📂 Loading data files...")
        
        # Load JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            print(f"✅ JSON loaded: {len(json.dumps(json_data))} characters")
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            json_data = {}
        
        # Load TXT
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                txt_data = f.read()
            print(f"✅ TXT loaded: {len(txt_data)} characters")
        except Exception as e:
            print(f"❌ Error loading TXT: {e}")
            txt_data = ""
        
        return json_data, txt_data
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better AI processing."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('\\n', ' ')
        text = text.replace('\n', ' ')
        
        # Fix table formatting issues
        text = re.sub(r'(\d+)\s+(\d+)', r'\1 \2', text)  # Fix number spacing
        text = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1 \2', text)  # Fix word spacing
        
        return text.strip()
    
    def extract_using_qa_model(self, text: str, questions: List[str]) -> List[str]:
        """Use QA model to extract information based on questions."""
        if not self.qa_model:
            return []
        
        answers = []
        
        # Limit text length for model
        max_length = 4000
        if len(text) > max_length:
            # Try to find the most relevant chunk
            text = text[:max_length]
        
        for question in questions:
            try:
                result = self.qa_model(question=question, context=text)
                if result['score'] > 0.1:  # Confidence threshold
                    answers.append({
                        'answer': result['answer'],
                        'confidence': result['score'],
                        'question': question
                    })
            except Exception as e:
                continue
        
        return answers
    
    def extract_statistical_values(self, text: str) -> Dict[str, List]:
        """Extract statistical values using pattern matching and AI."""
        extracted = {
            'correlations': [],
            'p_values': [],
            'sample_sizes': [],
            'r_squared': [],
            'beta_weights': []
        }
        
        # Pattern-based extraction
        for stat_type, patterns in self.statistical_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get context around the match
                    start = max(0, match.start() - 200)
                    end = min(len(text), match.end() + 200)
                    context = text[start:end]
                    
                    # Check if it's original research (not cited)
                    if not self.is_cited_finding(context):
                        value = match.group()
                        
                        if stat_type == 'correlation':
                            extracted['correlations'].append({
                                'value': value,
                                'context': context,
                                'extraction_method': 'pattern'
                            })
                        elif stat_type == 'p_value':
                            extracted['p_values'].append({
                                'value': value,
                                'context': context,
                                'extraction_method': 'pattern'
                            })
                        elif stat_type == 'sample_size':
                            extracted['sample_sizes'].append({
                                'value': value,
                                'context': context,
                                'extraction_method': 'pattern'
                            })
                        elif stat_type == 'r_squared':
                            extracted['r_squared'].append({
                                'value': value,
                                'context': context,
                                'extraction_method': 'pattern'
                            })
                        elif stat_type == 'beta_weight':
                            extracted['beta_weights'].append({
                                'value': value,
                                'context': context,
                                'extraction_method': 'pattern'
                            })
        
        return extracted
    
    def is_cited_finding(self, context: str) -> bool:
        """Check if a statistical finding is from a cited source."""
        context_lower = context.lower()
        
        # Citation indicators
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2020)
            r'\([^)]*et\s+al\.[^)]*\)',  # (Smith et al., 2020)
            r'according\s+to',
            r'as\s+(?:reported|found|shown)\s+by',
            r'previous\s+(?:research|studies|study)',
            r'prior\s+(?:research|studies|study)',
            r'literature\s+(?:suggests|shows|indicates)',
            r'meta-analysis\s+(?:by|of)',
            r'review\s+(?:by|of)'
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return True
        
        return False
    
    def categorize_predictor(self, predictor_text: str) -> str:
        """Categorize the predictor using semantic similarity."""
        predictor_lower = predictor_text.lower()
        
        # Direct keyword matching first
        for category, keywords in self.predictor_categories.items():
            for keyword in keywords:
                if keyword in predictor_lower:
                    return category
        
        # Semantic similarity if available
        if self.similarity_model:
            try:
                predictor_embedding = self.similarity_model.encode([predictor_text])
                
                best_category = 'other'
                best_score = 0
                
                for category, keywords in self.predictor_categories.items():
                    category_embeddings = self.similarity_model.encode(keywords)
                    similarities = self.similarity_model.similarity(predictor_embedding, category_embeddings)
                    max_sim = float(similarities.max())
                    
                    if max_sim > best_score:
                        best_score = max_sim
                        best_category = category
                
                if best_score > 0.3:  # Similarity threshold
                    return best_category
            except:
                pass
        
        return 'other'
    
    def parse_table_content(self, table_content: str) -> Dict[str, Any]:
        """Intelligently parse table content to extract values."""
        table_data = {
            'correlations': [],
            'p_values': [],
            'sample_sizes': [],
            'predictors': []
        }
        
        # Clean table content
        cleaned_content = self.preprocess_text(table_content)
        
        # Extract correlation values from table
        correlation_patterns = [
            r'[-+]?0\.\d{1,3}(?!\d)',  # Values like -0.94, 0.51
            r'[-+]?1\.0+',  # Perfect correlations
            r'correlation[:\s]*[-+]?[0-1]?\.\d+'
        ]
        
        for pattern in correlation_patterns:
            matches = re.finditer(pattern, cleaned_content, re.IGNORECASE)
            for match in matches:
                value = match.group().strip()
                # Validate correlation range
                try:
                    num_val = float(re.findall(r'[-+]?[0-1]?\.\d+', value)[0])
                    if -1.0 <= num_val <= 1.0 and abs(num_val) != 1.0:  # Exclude perfect correlations (often diagonal)
                        table_data['correlations'].append({
                            'value': value,
                            'numeric_value': num_val,
                            'source': 'table'
                        })
                except (ValueError, IndexError):
                    continue
        
        # Extract sample sizes
        n_patterns = [
            r'[Nn]\s*=?\s*\d+',
            r'sample\s+size\s*=?\s*\d+',
            r'\b\d{2,4}\b(?=\s*(?:participants|subjects))'
        ]
        
        for pattern in n_patterns:
            matches = re.finditer(pattern, cleaned_content, re.IGNORECASE)
            for match in matches:
                value = match.group().strip()
                table_data['sample_sizes'].append({
                    'value': value,
                    'source': 'table'
                })
        
        # Extract p-values
        p_patterns = [
            r'\.000',
            r'p\s*[<>=]\s*\.?\d+',
            r'significance\s*[<>=]\s*\.?\d+'
        ]
        
        for pattern in p_patterns:
            matches = re.finditer(pattern, cleaned_content, re.IGNORECASE)
            for match in matches:
                value = match.group().strip()
                table_data['p_values'].append({
                    'value': value,
                    'source': 'table'
                })
        
        return table_data
    
    def extract_comprehensive_data(self, json_data: Dict, txt_data: str) -> Dict[str, Any]:
        """Main extraction method using all AI techniques."""
        print("🤖 Starting comprehensive AI extraction...")
        
        # Combine all text sources
        all_text = txt_data + " " + json.dumps(json_data)
        cleaned_text = self.preprocess_text(all_text)
        
        extracted_data = {
            'predictor': '',
            'category': '',
            'pearson_r': '',
            'p_value': '',
            'r_squared': '',
            'beta_weight': '',
            'odds_ratio': '',
            'job_domain': '',
            'sample_context': '',
            'study_type': '',
            'measurement_type': '',
            'N': '',
            'year': '',
            'source_apa': '',
            'source_link': '',
            'notes': []
        }
        
        # Extract using QA model
        print("🔍 Extracting with QA model...")
        
        # Extract predictor
        predictor_answers = self.extract_using_qa_model(cleaned_text, self.qa_questions['predictor'])
        if predictor_answers:
            best_predictor = max(predictor_answers, key=lambda x: x['confidence'])
            extracted_data['predictor'] = best_predictor['answer']
            extracted_data['category'] = self.categorize_predictor(best_predictor['answer'])
        
        # Extract correlation
        correlation_answers = self.extract_using_qa_model(cleaned_text, self.qa_questions['correlation'])
        if correlation_answers:
            best_correlation = max(correlation_answers, key=lambda x: x['confidence'])
            extracted_data['pearson_r'] = self.clean_statistical_value(best_correlation['answer'], 'correlation')
        
        # Extract other statistical values
        for stat_type in ['p_value', 'sample_size', 'r_squared', 'beta_weight']:
            if stat_type in self.qa_questions:
                answers = self.extract_using_qa_model(cleaned_text, self.qa_questions[stat_type])
                if answers:
                    best_answer = max(answers, key=lambda x: x['confidence'])
                    if stat_type == 'sample_size':
                        extracted_data['N'] = self.clean_statistical_value(best_answer['answer'], stat_type)
                    else:
                        extracted_data[stat_type] = self.clean_statistical_value(best_answer['answer'], stat_type)
        
        # Extract study characteristics
        for char_type in ['study_type', 'job_domain', 'measurement_type', 'year', 'authors']:
            if char_type in self.qa_questions:
                answers = self.extract_using_qa_model(cleaned_text, self.qa_questions[char_type])
                if answers:
                    best_answer = max(answers, key=lambda x: x['confidence'])
                    if char_type == 'authors':
                        # Extract from source info
                        pass
                    else:
                        extracted_data[char_type] = best_answer['answer']
        
        # Extract using pattern matching
        print("📊 Extracting with pattern matching...")
        statistical_values = self.extract_statistical_values(cleaned_text)
        
        # Supplement with pattern-based findings
        if not extracted_data['pearson_r'] and statistical_values['correlations']:
            best_corr = max(statistical_values['correlations'], key=lambda x: len(x['context']))
            extracted_data['pearson_r'] = self.clean_statistical_value(best_corr['value'], 'correlation')
        
        if not extracted_data['p_value'] and statistical_values['p_values']:
            best_p = max(statistical_values['p_values'], key=lambda x: len(x['context']))
            extracted_data['p_value'] = self.clean_statistical_value(best_p['value'], 'p_value')
        
        if not extracted_data['N'] and statistical_values['sample_sizes']:
            best_n = max(statistical_values['sample_sizes'], key=lambda x: len(x['context']))
            extracted_data['N'] = self.clean_statistical_value(best_n['value'], 'sample_size')
        
        # Extract from tables
        print("📈 Extracting from tables...")
        if 'TABLES_EXTRACTED' in json_data:
            for table in json_data['TABLES_EXTRACTED']:
                if table.get('contains_statistics'):
                    table_data = self.parse_table_content(table.get('table_content', ''))
                    
                    # Use table data to supplement findings
                    if not extracted_data['pearson_r'] and table_data['correlations']:
                        best_table_corr = max(table_data['correlations'], key=lambda x: abs(x['numeric_value']))
                        if abs(best_table_corr['numeric_value']) < 0.95:  # Exclude diagonal values
                            extracted_data['pearson_r'] = str(best_table_corr['numeric_value'])
        
        # Extract metadata from JSON
        print("📋 Extracting metadata...")
        if 'PAPER_METADATA' in json_data:
            metadata = json_data['PAPER_METADATA']
            if not extracted_data['year']:
                extracted_data['year'] = metadata.get('year', '')
            if not extracted_data['source_apa']:
                extracted_data['source_apa'] = metadata.get('source_apa', '')
            extracted_data['source_link'] = metadata.get('source_link', '')
        
        if 'STUDY_OVERVIEW' in json_data:
            overview = json_data['STUDY_OVERVIEW']
            if not extracted_data['study_type']:
                extracted_data['study_type'] = overview.get('study_type', '')
            if not extracted_data['job_domain']:
                extracted_data['job_domain'] = overview.get('job_domain', '')
            if not extracted_data['measurement_type']:
                extracted_data['measurement_type'] = overview.get('measurement_type', '')
            if not extracted_data['N']:
                sample_n = overview.get('sample_n', '')
                extracted_data['N'] = self.clean_statistical_value(sample_n, 'sample_size')
            extracted_data['sample_context'] = overview.get('sample_context', '')
        
        # Generate notes
        extracted_data['notes'] = self.generate_extraction_notes(extracted_data, json_data, statistical_values)
        
        # Final validation and cleaning
        extracted_data = self.validate_and_clean_data(extracted_data)
        
        print("✅ Extraction complete!")
        return extracted_data
    
    def clean_statistical_value(self, value: str, value_type: str) -> str:
        """Clean and validate statistical values."""
        if not value:
            return ''
        
        # Extract numeric part
        if value_type == 'correlation':
            # Look for correlation values
            matches = re.findall(r'[-+]?[0-1]?\.\d+', value)
            if matches:
                try:
                    num_val = float(matches[0])
                    if -1.0 <= num_val <= 1.0:
                        return str(num_val)
                except ValueError:
                    pass
        
        elif value_type == 'p_value':
            # Look for p-values
            matches = re.findall(r'[0-1]?\.\d+', value)
            if matches:
                try:
                    num_val = float(matches[0])
                    if 0.0 <= num_val <= 1.0:
                        return str(num_val)
                except ValueError:
                    pass
            # Handle special cases
            if '.000' in value:
                return '< 0.001'
        
        elif value_type == 'sample_size':
            # Extract numbers
            matches = re.findall(r'\d+', value)
            if matches:
                try:
                    num_val = int(matches[0])
                    if 10 <= num_val <= 1000000:  # Reasonable range
                        return str(num_val)
                except ValueError:
                    pass
        
        elif value_type in ['r_squared', 'beta_weight']:
            # Extract decimal values
            matches = re.findall(r'[-+]?[0-1]?\.\d+', value)
            if matches:
                return matches[0]
        
        return value.strip()
    
    def generate_extraction_notes(self, extracted_data: Dict, json_data: Dict, statistical_values: Dict) -> List[str]:
        """Generate notes about the extraction process and findings."""
        notes = []
        
        # Note the type of predictor
        if extracted_data['predictor']:
            notes.append(f"Primary predictor: {extracted_data['predictor']}")
        
    def generate_extraction_notes(self, extracted_data: Dict, json_data: Dict, statistical_values: Dict) -> List[str]:
        """Generate notes about the extraction process and findings."""
        notes = []
        
        # Note the type of predictor
        if extracted_data['predictor']:
            notes.append(f"Primary predictor: {extracted_data['predictor']}")
        
        # Note statistical findings
        if extracted_data['pearson_r']:
            try:
                corr_val = float(extracted_data['pearson_r'])
                if abs(corr_val) > 0.7:
                    notes.append("Strong correlation found")
                elif abs(corr_val) > 0.3:
                    notes.append("Moderate correlation found")
                else:
                    notes.append("Weak correlation found")
                
                if corr_val < 0:
                    notes.append("Negative relationship with job performance")
                else:
                    notes.append("Positive relationship with job performance")
            except ValueError:
                pass
        
        # Note study quality indicators
        if extracted_data['N']:
            try:
                n_val = int(extracted_data['N'])
                if n_val > 1000:
                    notes.append("Large sample size")
                elif n_val > 100:
                    notes.append("Adequate sample size")
                else:
                    notes.append("Small sample size")
            except ValueError:
                pass
        
        # Note if multiple correlations were found
        if len(statistical_values.get('correlations', [])) > 1:
            notes.append(f"Multiple correlations found ({len(statistical_values['correlations'])})")
        
        # Note table presence
        if 'TABLES_EXTRACTED' in json_data:
            stat_tables = [t for t in json_data['TABLES_EXTRACTED'] if t.get('contains_statistics')]
            if stat_tables:
                notes.append(f"Data extracted from {len(stat_tables)} statistical tables")
        
        # Note any issues
        if not extracted_data['pearson_r']:
            notes.append("No clear correlation value found")
        
        if extracted_data['study_type'] == 'meta_analysis':
            notes.append("Meta-analysis study - aggregated results")
        
        return notes
    
    def validate_and_clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Final validation and cleaning of extracted data."""
        
        # Validate correlation
        if data['pearson_r']:
            try:
                corr_val = float(data['pearson_r'])
                if not (-1.0 <= corr_val <= 1.0):
                    data['pearson_r'] = ''
                    data['notes'].append("Invalid correlation value removed")
            except ValueError:
                data['pearson_r'] = ''
        
        # Validate p-value
        if data['p_value']:
            # Handle special cases
            if data['p_value'] in ['.000', '0.000']:
                data['p_value'] = '< 0.001'
            else:
                try:
                    p_val = float(data['p_value'].replace('<', '').replace('>', '').strip())
                    if not (0.0 <= p_val <= 1.0):
                        data['p_value'] = ''
                except ValueError:
                    pass
        
        # Validate year
        if data['year']:
            try:
                year_val = int(data['year'])
                if not (1950 <= year_val <= 2025):
                    data['year'] = ''
            except ValueError:
                data['year'] = ''
        
        # Clean text fields
        for field in ['predictor', 'study_type', 'job_domain', 'measurement_type']:
            if data[field]:
                data[field] = data[field].strip().replace('\n', ' ')
        
        # Ensure notes is a list
        if isinstance(data['notes'], str):
            data['notes'] = [data['notes']]
        
        return data
    
    def process_files(self, json_path: str, txt_path: str, output_path: str = None) -> Dict[str, Any]:
        """Main processing method."""
        print("🚀 Starting AI-powered data extraction...")
        
        # Load files
        json_data, txt_data = self.load_data_files(json_path, txt_path)
        
        # Extract data
        extracted_data = self.extract_comprehensive_data(json_data, txt_data)
        
        # Create output
        if output_path:
            self.save_extracted_data(extracted_data, output_path)
        
        return extracted_data
    
    def save_extracted_data(self, data: Dict[str, Any], output_path: str):
        """Save extracted data in multiple formats."""
        
        # Create DataFrame for CSV output
        df_data = {
            'ID': [1],  # Will be incremented for multiple papers
            'Predictor': [data['predictor']],
            'Category': [data['category']],
            'Pearson_r': [data['pearson_r']],
            'p_value': [data['p_value']],
            'R_squared': [data['r_squared']],
            'Beta_weight': [data['beta_weight']],
            'Odds_ratio': [data['odds_ratio']],
            'Job_Domain': [data['job_domain']],
            'Sample_Context': [data['sample_context']],
            'Study_Type': [data['study_type']],
            'Measurement_Type': [data['measurement_type']],
            'N': [data['N']],
            'Year': [data['year']],
            'Source_APA': [data['source_apa']],
            'Source_Link': [data['source_link']],
            'Notes': ['; '.join(data['notes']) if isinstance(data['notes'], list) else data['notes']]
        }
        
        df = pd.DataFrame(df_data)
        
        # Save as CSV
        csv_path = output_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV saved: {csv_path}")
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON saved: {output_path}")
        
        # Save detailed report
        report_path = output_path.replace('.json', '_report.txt')
        self.save_detailed_report(data, df, report_path)
        print(f"✅ Report saved: {report_path}")
    
    def save_detailed_report(self, data: Dict[str, Any], df: pd.DataFrame, report_path: str):
        """Save a detailed extraction report."""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AI-POWERED RESEARCH DATA EXTRACTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 EXTRACTED DATASET ROW\n")
            f.write("-" * 50 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            f.write("🔍 EXTRACTION DETAILS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Predictor: {data['predictor']}\n")
            f.write(f"Category: {data['category']}\n")
            f.write(f"Correlation: {data['pearson_r']}\n")
            f.write(f"P-value: {data['p_value']}\n")
            f.write(f"Sample Size: {data['N']}\n")
            f.write(f"Study Type: {data['study_type']}\n")
            f.write(f"Job Domain: {data['job_domain']}\n")
            f.write(f"Year: {data['year']}\n\n")
            
            f.write("📝 EXTRACTION NOTES\n")
            f.write("-" * 50 + "\n")
            if isinstance(data['notes'], list):
                for i, note in enumerate(data['notes'], 1):
                    f.write(f"{i}. {note}\n")
            else:
                f.write(f"{data['notes']}\n")
            f.write("\n")
            
            f.write("✅ QUALITY INDICATORS\n")
            f.write("-" * 50 + "\n")
            quality_score = 0
            total_fields = 16
            
            for key, value in data.items():
                if key != 'notes' and value:
                    quality_score += 1
            
            f.write(f"Data Completeness: {quality_score}/{total_fields} fields ({quality_score/total_fields*100:.1f}%)\n")
            
            if data['pearson_r']:
                f.write("✅ Primary correlation value extracted\n")
            else:
                f.write("⚠️ No correlation value found\n")
            
            if data['N']:
                f.write("✅ Sample size identified\n")
            else:
                f.write("⚠️ Sample size not found\n")
            
            if data['predictor']:
                f.write("✅ Predictor identified\n")
            else:
                f.write("⚠️ Predictor not clearly identified\n")

class BatchProcessor:
    """Process multiple files in batch."""
    
    def __init__(self):
        self.extractor = IntelligentDataExtractor()
        self.results = []
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all JSON/TXT file pairs in a directory."""
        print(f"📁 Processing directory: {input_dir}")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all JSON files
        json_files = list(input_path.glob("*.json"))
        
        for json_file in json_files:
            # Look for corresponding TXT file
            txt_file = json_file.with_suffix('.txt')
            if not txt_file.exists():
                # Try alternative naming
                txt_file = input_path / f"{json_file.stem}_AI_ready.txt"
                if not txt_file.exists():
                    print(f"⚠️ No corresponding TXT file for {json_file.name}")
                    continue
            
            print(f"\n🔄 Processing: {json_file.name}")
            
            try:
                # Process files
                output_file = output_path / f"{json_file.stem}_extracted.json"
                extracted_data = self.extractor.process_files(
                    str(json_file), 
                    str(txt_file), 
                    str(output_file)
                )
                
                # Add ID for dataset
                extracted_data['ID'] = len(self.results) + 1
                self.results.append(extracted_data)
                
                print(f"✅ Processed: {json_file.name}")
                
            except Exception as e:
                print(f"❌ Error processing {json_file.name}: {e}")
                continue
        
        # Create combined dataset
        self.create_combined_dataset(output_path)
    
    def create_combined_dataset(self, output_path: Path):
        """Create a combined dataset from all processed files."""
        print(f"\n📊 Creating combined dataset...")
        
        if not self.results:
            print("❌ No results to combine")
            return
        
        # Create DataFrame
        df_data = {
            'ID': [],
            'Predictor': [],
            'Category': [],
            'Pearson_r': [],
            'p_value': [],
            'R_squared': [],
            'Beta_weight': [],
            'Odds_ratio': [],
            'Job_Domain': [],
            'Sample_Context': [],
            'Study_Type': [],
            'Measurement_Type': [],
            'N': [],
            'Year': [],
            'Source_APA': [],
            'Source_Link': [],
            'Notes': []
        }
        
        for i, result in enumerate(self.results, 1):
            df_data['ID'].append(i)
            df_data['Predictor'].append(result.get('predictor', ''))
            df_data['Category'].append(result.get('category', ''))
            df_data['Pearson_r'].append(result.get('pearson_r', ''))
            df_data['p_value'].append(result.get('p_value', ''))
            df_data['R_squared'].append(result.get('r_squared', ''))
            df_data['Beta_weight'].append(result.get('beta_weight', ''))
            df_data['Odds_ratio'].append(result.get('odds_ratio', ''))
            df_data['Job_Domain'].append(result.get('job_domain', ''))
            df_data['Sample_Context'].append(result.get('sample_context', ''))
            df_data['Study_Type'].append(result.get('study_type', ''))
            df_data['Measurement_Type'].append(result.get('measurement_type', ''))
            df_data['N'].append(result.get('N', ''))
            df_data['Year'].append(result.get('year', ''))
            df_data['Source_APA'].append(result.get('source_apa', ''))
            df_data['Source_Link'].append(result.get('source_link', ''))
            notes = result.get('notes', [])
            df_data['Notes'].append('; '.join(notes) if isinstance(notes, list) else notes)
        
        df = pd.DataFrame(df_data)
        
        # Save combined dataset
        combined_csv = output_path / "combined_dataset.csv"
        df.to_csv(combined_csv, index=False)
        
        combined_json = output_path / "combined_dataset.json"
        with open(combined_json, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Combined dataset saved:")
        print(f"   📄 CSV: {combined_csv}")
        print(f"   📄 JSON: {combined_json}")
        print(f"   📊 Total records: {len(self.results)}")
        
        # Show summary statistics
        print(f"\n📈 DATASET SUMMARY")
        print(f"   Records with correlations: {sum(1 for r in self.results if r.get('pearson_r'))}")
        print(f"   Records with p-values: {sum(1 for r in self.results if r.get('p_value'))}")
        print(f"   Records with sample sizes: {sum(1 for r in self.results if r.get('N'))}")
        print(f"   Average data completeness: {df.count().mean()/len(df.columns)*100:.1f}%")

def main():
    """Main function for demonstration."""
    print("🤖 AI-POWERED RESEARCH DATA EXTRACTOR")
    print("=" * 60)
    
    # Example usage for single file pair
    json_path = r'C:\Users\eruku\Akshith\AI_Internship_2025\output\TAROT_A_Hierarchical_Framework_with_Multitask_Co-Pretraining_on_Semi-Structured_Data_towards_Effec_comprehensive_data.json'
    txt_path = r'C:\Users\eruku\Akshith\AI_Internship_2025\output\TAROT_A_Hierarchical_Framework_with_Multitask_Co-Pretraining_on_Semi-Structured_Data_towards_Effec_AI_ready.txt'
    output_path = r'C:\Users\eruku\Akshith\AI_Internship_2025\proper_cl\structured_results.json'
    
    # Initialize extractor
    extractor = IntelligentDataExtractor()
    
    # Process files (if they exist)
    if os.path.exists(json_path) and os.path.exists(txt_path):
        print("🔄 Processing single file pair...")
        result = extractor.process_files(json_path, txt_path, output_path)
        print("✅ Single file processing complete!")
        
        # Display result
        print("\n📊 EXTRACTED DATA:")
        for key, value in result.items():
            if value:
                print(f"  {key}: {value}")
    else:
        print("📁 Single files not found, demonstrating batch processing...")
        
        # Batch processing example
        input_dir = r"C:\Users\eruku\Akshith\AI_Internship_2025\output"     # Directory with JSON/TXT pairs
        output_dir = r"C:\Users\eruku\Akshith\AI_Internship_2025\proper"  # Output directory
        
        if os.path.exists(input_dir):
            processor = BatchProcessor()
            processor.process_directory(input_dir, output_dir)
        else:
            print(f"❌ Input directory '{input_dir}' not found")
            print("\n💡 USAGE INSTRUCTIONS:")
            print("1. For single file: Update json_path and txt_path variables")
            print("2. For batch processing: Create 'input_data' directory with JSON/TXT pairs")
            print("3. Run the script to extract data automatically")

if __name__ == "__main__":
    main()