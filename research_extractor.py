import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
import warnings
warnings.filterwarnings('ignore')

# PDF processing
try:
    import PyPDF2
    import fitz  # PyMuPDF
    import pdfplumber
except ImportError:
    print("Please install: pip install PyPDF2 PyMuPDF pdfplumber")

# NLP and AI
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    print("Please install: pip install transformers sentence-transformers torch")

try:
    import spacy
    # Try to load spacy model
    nlp = spacy.load("en_core_web_sm")
except:
    print("Please install: pip install spacy && python -m spacy download en_core_web_sm")

# Text processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
except ImportError:
    print("Please install: pip install nltk")
nltk.download('punkt_tab')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ComprehensiveResearchExtractor:
    def __init__(self):
        """Initialize the comprehensive extractor with advanced NLP capabilities."""
        print("🚀 Initializing Comprehensive Research Paper Extractor...")
        
        # Initialize models
        self.initialize_models()
        
        # Comprehensive keyword dictionaries
        self.setup_comprehensive_dictionaries()
        
        # Advanced patterns for statistical extraction
        self.setup_advanced_patterns()
        
        print("✅ Initialization complete!")
    
    def initialize_models(self):
        """Initialize all NLP models with error handling."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("⚠️ SpaCy model not available, using basic text processing")
            self.nlp = None
        
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            print("⚠️ SentenceTransformer not available")
            self.sentence_model = None
        
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except:
            print("⚠️ Summarization model not available")
            self.summarizer = None
        
        try:
            self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        except:
            print("⚠️ QA model not available")
            self.qa_pipeline = None
        
        try:
            self.classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        except:
            print("⚠️ Classification model not available")
            self.classifier = None
        
        self.lemmatizer = WordNetLemmatizer() if 'WordNetLemmatizer' in globals() else None
    
    def setup_comprehensive_dictionaries(self):
        """Setup comprehensive dictionaries for all extraction categories."""
        
        # Job Performance Predictors with extensive synonyms
        self.predictor_categories = {
            'cognitive_ability': [
                'cognitive ability', 'intelligence', 'IQ', 'general mental ability', 'GMA', 
                'reasoning ability', 'cognitive skills', 'mental capacity', 'intellectual ability',
                'cognitive test', 'aptitude test', 'ability test', 'cognitive assessment',
                'wonderlic', 'raven matrices', 'cognitive battery', 'intelligence quotient',
                'verbal reasoning', 'numerical reasoning', 'abstract reasoning', 'logical reasoning',
                'problem solving', 'critical thinking', 'analytical ability', 'cognitive performance'
            ],
            'personality': [
                'personality', 'big five', 'conscientiousness', 'extraversion', 'agreeableness', 
                'neuroticism', 'openness', 'personality traits', 'personality factors',
                'NEO-PI', 'big 5', 'five factor model', 'FFM', 'personality inventory',
                'personality assessment', 'temperament', 'character traits', 'behavioral style',
                'MBTI', 'myers-briggs', 'personality test', 'personality measure', 'trait',
                'emotional stability', 'openness to experience', 'personality dimension'
            ],
            'work_experience': [
                'work experience', 'job experience', 'tenure', 'years of experience', 
                'prior experience', 'previous experience', 'employment history', 'career length',
                'job tenure', 'work history', 'professional experience', 'occupational experience',
                'years on job', 'time in position', 'seniority', 'experience level',
                'years of service', 'job longevity', 'career experience', 'work background'
            ],
            'education': [
                'education', 'educational level', 'degree', 'qualification', 'academic achievement',
                'educational background', 'academic credentials', 'schooling', 'training',
                'diploma', 'certification', 'academic performance', 'GPA', 'grade point average',
                'educational attainment', 'academic record', 'university', 'college',
                'educational qualification', 'academic degree', 'academic background'
            ],
            'skills': [
                'skills', 'competencies', 'abilities', 'technical skills', 'soft skills',
                'job skills', 'professional skills', 'skill set', 'capabilities',
                'proficiencies', 'expertise', 'skill level', 'skill assessment',
                'hard skills', 'transferable skills', 'specialized skills', 'core competencies'
            ],
            'motivation': [
                'motivation', 'goal orientation', 'achievement motivation', 'intrinsic motivation',
                'extrinsic motivation', 'motivational factors', 'drive', 'ambition',
                'goal setting', 'achievement need', 'work motivation', 'job motivation',
                'motivational orientation', 'goal commitment', 'achievement striving',
                'work engagement', 'job involvement', 'motivational level'
            ],
            'leadership': [
                'leadership', 'leadership skills', 'management ability', 'supervisory skills',
                'leadership potential', 'leadership behavior', 'leadership style',
                'leadership effectiveness', 'management skills', 'supervisory ability',
                'leadership competence', 'leadership capacity', 'managerial skills',
                'leadership performance', 'management competency', 'supervisory competence'
            ],
            'teamwork': [
                'teamwork', 'collaboration', 'team skills', 'interpersonal skills',
                'team performance', 'team effectiveness', 'cooperative behavior',
                'team player', 'collaborative skills', 'group work', 'team dynamics',
                'social skills', 'interpersonal competence', 'relationship skills',
                'team collaboration', 'group collaboration', 'interpersonal relations'
            ],
            'communication': [
                'communication', 'communication skills', 'verbal ability', 'written communication',
                'oral communication', 'presentation skills', 'speaking skills',
                'listening skills', 'communication effectiveness', 'language skills',
                'articulation', 'expression', 'communication competence', 'verbal skills',
                'written skills', 'communication ability', 'interpersonal communication'
            ],
            'emotional_intelligence': [
                'emotional intelligence', 'EQ', 'emotional quotient', 'social intelligence',
                'emotional competence', 'emotional skills', 'emotional awareness',
                'self-awareness', 'empathy', 'emotional regulation', 'social awareness',
                'relationship management', 'emotional maturity', 'emotional ability',
                'social-emotional skills', 'emotional understanding'
            ],
            'job_performance': [
                'job performance', 'work performance', 'performance rating', 'performance evaluation',
                'performance appraisal', 'job effectiveness', 'work effectiveness',
                'performance measure', 'performance outcome', 'performance criteria',
                'task performance', 'contextual performance', 'overall performance',
                'work quality', 'job success', 'performance score', 'work productivity'
            ],
            'interview_performance': [
                'interview performance', 'interview rating', 'interview score', 'interview evaluation',
                'structured interview', 'unstructured interview', 'behavioral interview',
                'interview assessment', 'interview effectiveness', 'interview validity',
                'interview outcome', 'interview success', 'interview results'
            ],
            'assessment_center': [
                'assessment center', 'assessment centre', 'AC', 'assessment exercise',
                'simulation exercise', 'group exercise', 'in-basket exercise',
                'leaderless group discussion', 'role play', 'assessment rating'
            ]
        }
        
        # Job domains with comprehensive coverage
        self.job_domains = {
            'healthcare': ['healthcare', 'medical', 'nursing', 'physician', 'hospital', 'clinical', 'health', 'nurse', 'doctor'],
            'education': ['education', 'teaching', 'academic', 'school', 'university', 'educational', 'teacher', 'professor'],
            'technology': ['technology', 'IT', 'software', 'programming', 'tech', 'computer', 'digital', 'engineer'],
            'finance': ['finance', 'banking', 'financial', 'accounting', 'economic', 'investment', 'bank'],
            'manufacturing': ['manufacturing', 'production', 'industrial', 'factory', 'assembly', 'operations'],
            'retail': ['retail', 'sales', 'customer service', 'store', 'commerce', 'salesperson'],
            'government': ['government', 'public sector', 'civil service', 'federal', 'state', 'municipal'],
            'military': ['military', 'army', 'navy', 'air force', 'defense', 'soldier', 'officer'],
            'police': ['police', 'law enforcement', 'officer', 'security', 'criminal justice'],
            'management': ['management', 'executive', 'leadership', 'supervisor', 'manager', 'managerial'],
            'engineering': ['engineering', 'engineer', 'technical', 'mechanical', 'electrical', 'civil'],
            'research': ['research', 'scientist', 'laboratory', 'academic research', 'R&D', 'researcher']
        }
        
        # Study types
        self.study_types = {
            'meta_analysis': ['meta-analysis', 'meta analysis', 'systematic review', 'meta-analytic'],
            'longitudinal': ['longitudinal', 'longitudinal study', 'follow-up', 'over time', 'panel study'],
            'cross_sectional': ['cross-sectional', 'cross sectional', 'survey study', 'snapshot'],
            'experimental': ['experimental', 'experiment', 'randomized', 'controlled trial', 'RCT'],
            'quasi_experimental': ['quasi-experimental', 'quasi experimental', 'natural experiment'],
            'correlational': ['correlational', 'correlation study', 'relationship study'],
            'case_study': ['case study', 'case-study', 'case analysis'],
            'field_study': ['field study', 'field research', 'organizational study', 'workplace study'],
            'laboratory': ['laboratory', 'lab study', 'controlled environment'],
            'mixed_methods': ['mixed methods', 'mixed-methods', 'qualitative and quantitative']
        }
        
        # Measurement types
        self.measurement_types = {
            'self_report': ['self-report', 'self report', 'questionnaire', 'survey', 'self-assessment', 'self-rating'],
            'objective': ['objective', 'performance test', 'cognitive test', 'ability test', 'objective measure'],
            'supervisor_rating': ['supervisor rating', 'manager rating', 'superior rating', 'boss rating', 'supervisor evaluation'],
            'peer_rating': ['peer rating', 'colleague rating', '360 degree', '360-degree', 'peer evaluation'],
            'behavioral': ['behavioral', 'behavior', 'observation', 'observed behavior', 'behavioral assessment'],
            'archival': ['archival', 'records', 'personnel files', 'company data', 'organizational records']
        }
    
    def setup_advanced_patterns(self):
        """Setup advanced regex patterns for comprehensive statistical extraction."""
        
        # Comprehensive statistical patterns
        self.statistical_patterns = {
            'correlation': [
                # Basic patterns
                r'r\s*=\s*[-+]?[0-9]*\.?[0-9]+',
                r'correlation\s+(?:coefficient\s+)?(?:of\s+)?[-+]?[0-9]*\.?[0-9]+',
                r'pearson\s+(?:r\s*=\s*|correlation\s*=\s*)?[-+]?[0-9]*\.?[0-9]+',
                r'spearman\s+(?:rho\s*=\s*|correlation\s*=\s*)?[-+]?[0-9]*\.?[0-9]+',
                r'r\([\d,\s]+\)\s*=\s*[-+]?[0-9]*\.?[0-9]+',
                # Advanced patterns
                r'correlated\s+(?:significantly\s+)?(?:at\s+)?[-+]?[0-9]*\.?[0-9]+',
                r'correlation\s+between\s+\w+\s+and\s+\w+\s+(?:was\s+|is\s+)?[-+]?[0-9]*\.?[0-9]+',
                r'relationship\s+(?:of\s+)?[-+]?[0-9]*\.?[0-9]+',
                r'associated\s+(?:with\s+)?[-+]?[0-9]*\.?[0-9]+',
                # Table patterns
                r'(?:pearson|spearman)\s*\n?\s*[-+]?[0-9]*\.?[0-9]+',
                r'correlation\s*\n?\s*[-+]?[0-9]*\.?[0-9]+',
                # Contextual patterns
                r'validity\s+(?:coefficient\s+)?(?:of\s+)?[-+]?[0-9]*\.?[0-9]+',
                r'reliability\s+(?:coefficient\s+)?(?:of\s+)?[-+]?[0-9]*\.?[0-9]+'
            ],
            'regression': [
                r'β\s*=\s*[-+]?[0-9]*\.?[0-9]+',
                r'beta\s*=\s*[-+]?[0-9]*\.?[0-9]+',
                r'b\s*=\s*[-+]?[0-9]*\.?[0-9]+',
                r'regression\s+coefficient\s*=\s*[-+]?[0-9]*\.?[0-9]+',
                r'standardized\s+coefficient\s*=\s*[-+]?[0-9]*\.?[0-9]+',
                r'unstandardized\s+coefficient\s*=\s*[-+]?[0-9]*\.?[0-9]+',
                r'slope\s*=\s*[-+]?[0-9]*\.?[0-9]+',
                r'beta\s+weight\s*=\s*[-+]?[0-9]*\.?[0-9]+'
            ],
            'p_value': [
                r'p\s*[<>=]\s*[0-9]*\.?[0-9]+',
                r'p-value\s*[<>=]\s*[0-9]*\.?[0-9]+',
                r'significance\s*[<>=]\s*[0-9]*\.?[0-9]+',
                r'sig\.\s*[<>=]\s*[0-9]*\.?[0-9]+',
                r'probability\s*[<>=]\s*[0-9]*\.?[0-9]+',
                r'p\s*\(\s*[<>=]\s*[0-9]*\.?[0-9]+\s*\)',
                r'significant\s+at\s+[0-9]*\.?[0-9]+',
                r'α\s*=\s*[0-9]*\.?[0-9]+',
                r'alpha\s*=\s*[0-9]*\.?[0-9]+'
            ],
            'r_squared': [
                r'[Rr]²\s*=\s*[0-9]*\.?[0-9]+',
                r'[Rr]-squared\s*=\s*[0-9]*\.?[0-9]+',
                r'[Rr]2\s*=\s*[0-9]*\.?[0-9]+',
                r'coefficient\s+of\s+determination\s*=\s*[0-9]*\.?[0-9]+',
                r'variance\s+explained\s*=\s*[0-9]*\.?[0-9]+',
                r'explained\s+variance\s*=\s*[0-9]*\.?[0-9]+'
            ],
            'odds_ratio': [
                r'odds\s+ratio\s*=\s*[0-9]*\.?[0-9]+',
                r'OR\s*=\s*[0-9]*\.?[0-9]+',
                r'exp\(b\)\s*=\s*[0-9]*\.?[0-9]+',
                r'odds\s*=\s*[0-9]*\.?[0-9]+'
            ],
            'sample_size': [
                r'[Nn]\s*=\s*[\d,]+',
                r'sample\s+size\s+(?:of\s+)?[\d,]+',
                r'participants\s*\(\s*[Nn]\s*=\s*[\d,]+\s*\)',
                r'[\d,]+\s+participants',
                r'[\d,]+\s+subjects',
                r'total\s+(?:of\s+)?[\d,]+\s+(?:participants|subjects)',
                r'study\s+included\s+[\d,]+',
                r'comprised\s+[\d,]+\s+(?:participants|subjects)',
                r'[\d,]+\s+individuals?',
                r'[\d,]+\s+employees?',
                r'[\d,]+\s+workers?'
            ]
        }
        
        # Citation detection patterns
        self.citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2020)
            r'\([^)]*et\s+al\.[^)]*\)',  # (Smith et al., 2020)
            r'according\s+to\s+[A-Z][a-z]+',
            r'as\s+(?:reported|found|shown)\s+by\s+[A-Z][a-z]+',
            r'previous\s+(?:research|studies|study|work)',
            r'prior\s+(?:research|studies|study|work)',
            r'earlier\s+(?:research|studies|study|work)',
            r'literature\s+(?:suggests|shows|indicates|reports)',
            r'research\s+(?:suggests|shows|indicates|reports)',
            r'studies\s+(?:suggest|show|indicate|report)',
            r'meta-analysis\s+(?:by|of|found)',
            r'review\s+(?:by|of|found)',
            r'[A-Z][a-z]+\s+(?:and|&)\s+[A-Z][a-z]+\s+\(\d{4}\)',
            r'[A-Z][a-z]+\s+et\s+al\.\s+\(\d{4}\)'
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extract text from PDF using multiple methods."""
        print(f"📄 Extracting text from: {os.path.basename(pdf_path)}")
        
        full_text = ""
        metadata = {}
        
        # Method 1: PyMuPDF (best for most PDFs)
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata or {}
            
            for page in doc:
                full_text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"⚠️ PyMuPDF failed: {e}")
        
        # Method 2: pdfplumber (good for tables)
        if not full_text.strip():
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text
            except Exception as e:
                print(f"⚠️ pdfplumber failed: {e}")
        
        # Method 3: PyPDF2 (fallback)
        if not full_text.strip():
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        full_text += page.extract_text()
            except Exception as e:
                print(f"⚠️ PyPDF2 failed: {e}")
        
        return full_text, metadata
    
    def extract_tables_comprehensive(self, pdf_path: str) -> List[Dict]:
        """Extract tables with comprehensive analysis."""
        print("📊 Extracting tables...")
        
        tables_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 1 and len(table[0]) > 1:
                            try:
                                # Convert to DataFrame
                                df = pd.DataFrame(table[1:], columns=table[0])
                                df = df.dropna(how='all').reset_index(drop=True)
                                
                                if len(df) > 0:
                                    table_analysis = self.analyze_table_content(df, page_num, table_idx)
                                    tables_data.append(table_analysis)
                            except Exception as e:
                                print(f"⚠️ Error processing table: {e}")
        except Exception as e:
            print(f"⚠️ Table extraction failed: {e}")
        
        return tables_data
    
    def analyze_table_content(self, df: pd.DataFrame, page_num: int, table_idx: int) -> Dict:
        """Analyze table content for statistical information."""
        table_str = df.to_string().lower()
        
        analysis = {
            'page': page_num + 1,
            'table_index': table_idx + 1,
            'raw_table': df.to_dict(),
            'table_string': df.to_string(),
            'contains_statistics': False,
            'statistical_content': [],
            'predictor_content': [],
            'table_type': 'unknown'
        }
        
        # Check for statistical content
        stat_indicators = ['correlation', 'r =', 'β', 'beta', 'p <', 'p =', 'sig', 'mean', 'std']
        if any(indicator in table_str for indicator in stat_indicators):
            analysis['contains_statistics'] = True
            
            # Extract statistical values from table
            for stat_type, patterns in self.statistical_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, table_str, re.IGNORECASE)
                    if matches:
                        analysis['statistical_content'].extend([{
                            'type': stat_type,
                            'values': matches,
                            'pattern': pattern
                        }])
        
        # Check for predictor content
        for category, keywords in self.predictor_categories.items():
            for keyword in keywords:
                if keyword.lower() in table_str:
                    analysis['predictor_content'].append({
                        'category': category,
                        'keyword': keyword
                    })
        
        # Determine table type
        if 'correlation' in table_str:
            analysis['table_type'] = 'correlation_matrix'
        elif any(term in table_str for term in ['β', 'beta', 'regression']):
            analysis['table_type'] = 'regression_results'
        elif any(term in table_str for term in ['mean', 'std', 'descriptive']):
            analysis['table_type'] = 'descriptive_statistics'
        
        return analysis
    
    def identify_paper_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract paper sections."""
        sections = {}
        
        section_patterns = {
            'abstract': r'(?:^|\n)\s*(?:abstract|summary)[\s]*:?\s*\n',
            'introduction': r'(?:^|\n)\s*(?:introduction|background)[\s]*:?\s*\n',
            'literature_review': r'(?:^|\n)\s*(?:literature review|related work|theoretical background)[\s]*:?\s*\n',
            'methods': r'(?:^|\n)\s*(?:methods?|methodology|procedure|participants?|study design)[\s]*:?\s*\n',
            'results': r'(?:^|\n)\s*(?:results?|findings?)[\s]*:?\s*\n',
            'discussion': r'(?:^|\n)\s*(?:discussion|implications?)[\s]*:?\s*\n',
            'conclusion': r'(?:^|\n)\s*(?:conclusion|summary|limitations)[\s]*:?\s*\n',
            'references': r'(?:^|\n)\s*(?:references?|bibliography|works cited)[\s]*:?\s*\n'
        }
        
        for section_name, pattern in section_patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            if matches:
                start_pos = matches[0].end()
                end_pos = len(text)
                
                # Find end of section
                for other_name, other_pattern in section_patterns.items():
                    if other_name != section_name:
                        other_matches = list(re.finditer(other_pattern, text[start_pos:], re.IGNORECASE | re.MULTILINE))
                        if other_matches:
                            potential_end = start_pos + other_matches[0].start()
                            if potential_end < end_pos:
                                end_pos = potential_end
                
                sections[section_name] = text[start_pos:end_pos].strip()
        
        return sections
    
    def is_cited_information(self, text_snippet: str, position: int, window: int = 400) -> Dict[str, Any]:
        """Advanced citation detection with context analysis."""
        start = max(0, position - window)
        end = min(len(text_snippet), position + window)
        context = text_snippet[start:end]
        
        citation_indicators = []
        confidence_score = 0.0
        
        # Check for direct citation patterns
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            for match in matches:
                citation_indicators.append({
                    'type': 'direct_citation',
                    'text': match.group(),
                    'pattern': pattern,
                    'distance': abs(match.start() - (position - start))
                })
                confidence_score += 0.3
        
        # Check for author-year patterns
        author_year_pattern = r'[A-Z][a-z]+(?:\s+(?:and|&|\s)\s+[A-Z][a-z]+)*\s*\([12]\d{3}\)'
        author_matches = re.finditer(author_year_pattern, context)
        for match in author_matches:
            citation_indicators.append({
                'type': 'author_year',
                'text': match.group(),
                'distance': abs(match.start() - (position - start))
            })
            confidence_score += 0.4
        
        # Check for reference indicators
        ref_indicators = [
            'previous research', 'prior study', 'earlier work', 'literature shows',
            'research indicates', 'studies suggest', 'found by', 'reported by',
            'according to', 'as shown by', 'demonstrated by', 'meta-analysis',
            'systematic review'
        ]
        
        context_lower = context.lower()
        for indicator in ref_indicators:
            if indicator in context_lower:
                citation_indicators.append({
                    'type': 'reference_indicator',
                    'text': indicator,
                    'distance': context_lower.find(indicator)
                })
                confidence_score += 0.2
        
        is_cited = confidence_score > 0.3
        
        return {
            'is_cited': is_cited,
            'confidence': min(confidence_score, 1.0),
            'indicators': citation_indicators,
            'context': context
        }
    
    def extract_comprehensive_statistics(self, text: str, sections: Dict[str, str]) -> Dict[str, List]:
        """Extract statistics with comprehensive analysis and citation checking."""
        print("📈 Extracting comprehensive statistics...")
        
        statistics = {
            'correlations': [],
            'regressions': [],
            'p_values': [],
            'r_squared': [],
            'odds_ratios': [],
            'sample_sizes': []
        }
        
        # Focus on results and methods sections primarily
        target_sections = ['results', 'methods', 'discussion']
        target_text = ""
        
        for section in target_sections:
            if section in sections:
                target_text += f"\n\n=== {section.upper()} SECTION ===\n\n" + sections[section]
        
        if not target_text.strip():
            target_text = text
        
        # Extract statistics with citation analysis
        stat_mapping = {
            'correlations': self.statistical_patterns['correlation'],
            'regressions': self.statistical_patterns['regression'],
            'p_values': self.statistical_patterns['p_value'],
            'r_squared': self.statistical_patterns['r_squared'],
            'odds_ratios': self.statistical_patterns['odds_ratio'],
            'sample_sizes': self.statistical_patterns['sample_size']
        }
        
        for stat_type, patterns in stat_mapping.items():
            for pattern in patterns:
                matches = re.finditer(pattern, target_text, re.IGNORECASE)
                for match in matches:
                    # Analyze citation context
                    citation_analysis = self.is_cited_information(target_text, match.start())
                    
                    # Only include if not clearly cited
                    if not citation_analysis['is_cited'] or citation_analysis['confidence'] < 0.5:
                        # Extract extended context
                        context_start = max(0, match.start() - 500)
                        context_end = min(len(target_text), match.end() + 500)
                        extended_context = target_text[context_start:context_end].strip()
                        
                        stat_entry = {
                            'value': match.group().strip(),
                            'context': extended_context,
                            'position': match.start(),
                            'pattern_used': pattern,
                            'citation_analysis': citation_analysis,
                            'confidence': 1.0 - citation_analysis['confidence']
                        }
                        
                        statistics[stat_type].append(stat_entry)
        
        # Remove duplicates and sort by confidence
        for stat_type in statistics:
            seen_values = set()
            unique_stats = []
            for stat in statistics[stat_type]:
                if stat['value'] not in seen_values:
                    seen_values.add(stat['value'])
                    unique_stats.append(stat)
            statistics[stat_type] = sorted(unique_stats, key=lambda x: x['confidence'], reverse=True)
        
        return statistics
    
    def extract_predictors_with_context(self, text: str, sections: Dict[str, str]) -> List[Dict]:
        """Extract predictors with comprehensive context analysis."""
        print("🎯 Extracting predictors with context...")
        
        predictors = []
        
        # Use full text but focus on key sections
        focus_text = ""
        for section in ['introduction', 'methods', 'results', 'discussion']:
            if section in sections:
                focus_text += f" {sections[section]}"
        
        if not focus_text.strip():
            focus_text = text
        
        focus_text_lower = focus_text.lower()
        
        for category, keywords in self.predictor_categories.items():
            category_matches = []
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Find all occurrences
                start = 0
                while True:
                    pos = focus_text_lower.find(keyword_lower, start)
                    if pos == -1:
                        break
                    
                    # Extract comprehensive context
                    context_start = max(0, pos - 600)
                    context_end = min(len(focus_text), pos + len(keyword) + 600)
                    context = focus_text[context_start:context_end].strip()
                    
                    # Check if this is original research context
                    citation_analysis = self.is_cited_information(focus_text, pos, 400)
                    
                    if not citation_analysis['is_cited'] or citation_analysis['confidence'] < 0.4:
                        confidence = self.calculate_predictor_confidence(context, keyword, citation_analysis)
                        
                        category_matches.append({
                            'predictor': keyword,
                            'category': category,
                            'context': context,
                            'position': pos,
                            'confidence': confidence,
                            'citation_analysis': citation_analysis
                        })
                    
                    start = pos + 1
            
            # Keep best match per category
            if category_matches:
                best_match = max(category_matches, key=lambda x: x['confidence'])
                predictors.append(best_match)
        
        return predictors
    
    def calculate_predictor_confidence(self, context: str, predictor: str, citation_analysis: Dict) -> float:
        """Calculate confidence score for predictor identification."""
        context_lower = context.lower()
        confidence = 0.6  # Base confidence
        
        # Boost for measurement terms
        measurement_terms = ['measured', 'assessed', 'evaluated', 'tested', 'administered', 'collected']
        for term in measurement_terms:
            if term in context_lower:
                confidence += 0.1
        
        # Boost for statistical terms
        statistical_terms = ['correlation', 'regression', 'significant', 'predict', 'relationship', 'analysis']
        for term in statistical_terms:
            if term in context_lower:
                confidence += 0.1
        
        # Boost for original research indicators
        original_terms = ['our study', 'this study', 'we found', 'we measured', 'results show', 'data indicate']
        for term in original_terms:
            if term in context_lower:
                confidence += 0.2
        
        # Reduce for citation confidence
        confidence -= citation_analysis['confidence'] * 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def extract_study_metadata(self, text: str, pdf_path: str) -> Dict[str, str]:
        """Extract comprehensive study metadata."""
        metadata = {'source_path': pdf_path}
        
        # Extract title
        lines = text.split('\n')
        potential_titles = []
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            if (20 <= len(line) <= 300 and not re.match(r'^[A-Z\s]+', line) and not re.search(r'^\d+', line) and not line.lower().startswith(('abstract', 'introduction', 'keywords', 'doi', 'volume'))):
                potential_titles.append((line, i))
        
        metadata['title'] = potential_titles[0][0] if potential_titles else 'Unknown Title'
        
        # Extract year with multiple patterns
        year_patterns = [
            r'\b(19|20)\d{2}\b',
            r'published\s+in\s+(19|20)\d{2}',
            r'copyright\s+(19|20)\d{2}',
            r'\((19|20)\d{2}\)'
        ]
        
        years_found = []
        for pattern in year_patterns:
            matches = re.findall(pattern, text[:4000])
            for match in matches:
                year = match if isinstance(match, str) else match[0] + match[1]
                years_found.append(year)
        
        valid_years = [year for year in years_found if 1990 <= int(year) <= 2025]
        metadata['year'] = valid_years[0] if valid_years else 'Unknown'
        
        # Extract authors
        author_section = text[:3000]
        author_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*),\s*([A-Z]\.(?:\s*[A-Z]\.)*)',
            r'([A-Z]\.(?:\s*[A-Z]\.)*)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'([A-Z][a-z]+)\s+([A-Z][a-z]+)'
        ]
        
        authors_found = []
        for pattern in author_patterns:
            matches = re.findall(pattern, author_section)
            for match in matches[:5]:
                if isinstance(match, tuple):
                    author = f"{match[0]} {match[1]}".strip()
                    if (len(author) > 3 and 
                        not author.lower() in ['the', 'and', 'for', 'this', 'that', 'abstract', 'keywords']):
                        authors_found.append(author)
        
        metadata['authors'] = ', '.join(authors_found[:3]) if authors_found else 'Unknown Authors'
        
        return metadata
    
    def identify_study_characteristics(self, text: str, sections: Dict[str, str]) -> Dict[str, str]:
        """Identify comprehensive study characteristics."""
        characteristics = {
            'study_type': 'not_specified',
            'job_domain': 'general',
            'measurement_type': 'not_specified'
        }
        
        text_lower = text.lower()
        
        # Study type identification
        for study_type, keywords in self.study_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    characteristics['study_type'] = study_type
                    break
            if characteristics['study_type'] != 'not_specified':
                break
        
        # Job domain identification
        for domain, keywords in self.job_domains.items():
            for keyword in keywords:
                if keyword in text_lower:
                    characteristics['job_domain'] = domain
                    break
            if characteristics['job_domain'] != 'general':
                break
        
        # Measurement type identification
        for measurement, keywords in self.measurement_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    characteristics['measurement_type'] = measurement
                    break
            if characteristics['measurement_type'] != 'not_specified':
                break
        
        return characteristics
    
    def extract_doi_from_text(self, text: str) -> str:
        """Extract DOI from paper text."""
        doi_patterns = [
            r'doi:\s*(10\.\d+/[^\s]+)',
            r'DOI:\s*(10\.\d+/[^\s]+)',
            r'https://doi\.org/(10\.\d+/[^\s]+)',
            r'http://dx\.doi\.org/(10\.\d+/[^\s]+)',
            r'(10\.\d+/[^\s]+)'
        ]
        
        for pattern in doi_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return 'Not found'
    
    def generate_comprehensive_summary(self, text: str, sections: Dict[str, str]) -> str:
        """Generate comprehensive summary using available models."""
        try:
            # Priority: Abstract > Results > Introduction
            summary_text = ""
            if 'abstract' in sections and len(sections['abstract'].strip()) > 50:
                summary_text = sections['abstract']
            elif 'results' in sections and len(sections['results'].strip()) > 50:
                summary_text = sections['results'][:2000]
            elif 'introduction' in sections and len(sections['introduction'].strip()) > 50:
                summary_text = sections['introduction'][:2000]
            else:
                summary_text = text[:2000]
            
            if self.summarizer and len(summary_text) > 100:
                try:
                    if len(summary_text) > 1024:
                        # Split into chunks
                        chunks = [summary_text[i:i+1024] for i in range(0, len(summary_text), 512)]
                        summaries = []
                        for chunk in chunks[:3]:
                            if len(chunk.strip()) > 100:
                                summary = self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                                summaries.append(summary[0]['summary_text'])
                        return ' '.join(summaries)
                    else:
                        summary = self.summarizer(summary_text, max_length=300, min_length=100, do_sample=False)
                        return summary[0]['summary_text']
                except:
                    pass
            
            # Fallback: extractive summary
            sentences = sent_tokenize(summary_text) if 'sent_tokenize' in globals() else summary_text.split('. ')
            return '. '.join(sentences[:5]) + '.'
            
        except Exception as e:
            print(f"⚠️ Summarization failed: {e}")
            sentences = text.split('. ')
            return '. '.join(sentences[:3]) + '.'
    
    def create_comprehensive_output(self, pdf_path: str, text: str, sections: Dict[str, str], 
                              tables: List[Dict], statistics: Dict[str, List], 
                              predictors: List[Dict], metadata: Dict[str, str], 
                              characteristics: Dict[str, str]) -> Dict[str, Any]:
        """Create comprehensive structured output."""
        
        # Generate summary
        summary = self.generate_comprehensive_summary(text, sections)
        
        # Create APA citation
        authors = metadata.get('authors', 'Unknown Authors')
        year = metadata.get('year', 'n.d.')
        title = metadata.get('title', 'Unknown Title')
        apa_citation = f"{authors} ({year}). {title}."
        
        # Analyze sample context
        sample_context = ""
        if 'methods' in sections:
            methods_text = sections['methods']
            # Extract first few sentences about participants
            sentences = sent_tokenize(methods_text) if 'sent_tokenize' in globals() else methods_text.split('. ')
            participant_sentences = []
            for sentence in sentences[:10]:
                if any(word in sentence.lower() for word in ['participant', 'subject', 'sample', 'employee', 'worker', 'student']):
                    participant_sentences.append(sentence)
            sample_context = '. '.join(participant_sentences[:3])
        
        # Extract sample size from statistics
        sample_n = "Not specified"
        if statistics['sample_sizes']:
            sample_n = statistics['sample_sizes'][0]['value']
        
        # Compile notes
        notes = []
        
        # Add predictor information
        if predictors:
            predictor_list = [p['predictor'] for p in predictors[:5]]
            notes.append(f"Key predictors identified: {', '.join(predictor_list)}")
        
        # Add statistical summary
        stat_summary = []
        for stat_type, stat_list in statistics.items():
            if stat_list:
                stat_summary.append(f"{len(stat_list)} {stat_type}")
        if stat_summary:
            notes.append(f"Statistics found: {', '.join(stat_summary)}")
        
        # Add table summary
        if tables:
            table_types = [t['table_type'] for t in tables if t['contains_statistics']]
            if table_types:
                notes.append(f"Statistical tables: {', '.join(set(table_types))}")
        
        # Add quality indicators
        original_stats_count = sum(len(stats) for stats in statistics.values())
        if original_stats_count > 10:
            notes.append("High confidence in original findings extraction")
        elif original_stats_count > 5:
            notes.append("Medium confidence in original findings extraction")
        else:
            notes.append("Low statistical content found")
        
        output_data = {
            'PAPER_METADATA': {
                'title': metadata['title'],
                'authors': metadata['authors'],
                'year': metadata['year'],
                'journal': metadata.get('journal', 'Unknown Journal'),  # ADD
                'doi': metadata.get('doi', 'Not found'),               # ADD
                'url': metadata.get('url', ''),                        # ADD
                'source_database': metadata.get('source_database', ''), # ADD
                'citation_count': metadata.get('citation_count', 0),    # ADD
                'search_query': metadata.get('search_query', ''),       # ADD
                'source_apa': f"{metadata['authors']} ({metadata['year']}). {metadata['title']}. {metadata.get('journal', 'Unknown Journal')}.",
                'source_link': metadata.get('url', metadata['source_path'])
            },
                
            'STUDY_OVERVIEW': {
                'study_type': characteristics['study_type'],
                'job_domain': characteristics['job_domain'],
                'measurement_type': characteristics['measurement_type'],
                'sample_n': sample_n,
                'sample_context': sample_context
            },
            
            'PAPER_SUMMARY': summary,
            
            'PREDICTORS_IDENTIFIED': [
                {
                    'category': p['category'],
                    'predictor': p['predictor'],
                    'context': p['context'][:400],
                    'confidence': round(p['confidence'], 2)
                }
                for p in predictors
            ],
            
            'STATISTICAL_FINDINGS': {
                'correlations': [
                    {
                        'value': s['value'],
                        'context': s['context'][:300],
                        'confidence': round(s['confidence'], 2)
                    }
                    for s in statistics['correlations'][:10]
                ],
                'regressions': [
                    {
                        'value': s['value'],
                        'context': s['context'][:300],
                        'confidence': round(s['confidence'], 2)
                    }
                    for s in statistics['regressions'][:10]
                ],
                'p_values': [s['value'] for s in statistics['p_values'][:5]],
                'r_squared': [s['value'] for s in statistics['r_squared'][:5]],
                'odds_ratios': [s['value'] for s in statistics['odds_ratios'][:5]]
            },
            
            'TABLES_EXTRACTED': [
                {
                    'page': t['page'],
                    'table_index': t['table_index'],
                    'table_type': t['table_type'],
                    'contains_statistics': t['contains_statistics'],
                    'table_content': t['table_string'] if t['contains_statistics'] else "Non-statistical table",
                    'statistical_summary': t['statistical_content'][:3]
                }
                for t in tables
            ],
            
            'RESEARCH_CONTEXT': {
                'methods_section': sections.get('methods', 'Not identified')[:1000],
                'results_section': sections.get('results', 'Not identified')[:1000],
                'discussion_section': sections.get('discussion', 'Not identified')[:500]
            },
            
            'NOTES': notes,
            
            'EXTRACTION_QUALITY': {
                'sections_found': len(sections),
                'tables_found': len(tables),
                'statistical_tables': len([t for t in tables if t['contains_statistics']]),
                'predictors_identified': len(predictors),
                'statistics_extracted': sum(len(stats) for stats in statistics.values()),
                'confidence_level': 'high' if original_stats_count > 10 else 'medium' if original_stats_count > 5 else 'low'
            }
        }
        
        return output_data
    
    def save_comprehensive_txt_output(self, output_data: Dict[str, Any], output_path: str):
        """Save comprehensive human-readable output optimized for AI processing."""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE RESEARCH PAPER EXTRACTION FOR AI PROCESSING\n")
            f.write("=" * 100 + "\n\n")
            
            # Enhanced Paper Metadata - MODIFY THIS SECTION
            f.write("📄 PAPER METADATA (WEB + PDF EXTRACTION)\n")
            f.write("-" * 50 + "\n")
            metadata = output_data['PAPER_METADATA']
            f.write(f"TITLE: {metadata['title']}\n")
            f.write(f"AUTHORS: {metadata['authors']}\n")
            f.write(f"YEAR: {metadata['year']}\n")
            f.write(f"JOURNAL/VENUE: {metadata['journal']}\n")          # ADD
            f.write(f"DOI: {metadata['doi']}\n")                        # ADD
            f.write(f"URL: {metadata['url']}\n")                        # ADD
            f.write(f"SOURCE DATABASE: {metadata['source_database']}\n") # ADD
            f.write(f"CITATION COUNT: {metadata['citation_count']}\n")   # ADD
            f.write(f"SEARCH QUERY: {metadata['search_query']}\n")       # ADD
            f.write(f"APA CITATION: {metadata['source_apa']}\n")
            f.write(f"LOCAL PATH: {metadata['source_link']}\n\n")
            
            # Study Overview
            f.write("🔬 STUDY CHARACTERISTICS\n")
            f.write("-" * 50 + "\n")
            overview = output_data['STUDY_OVERVIEW']
            f.write(f"STUDY TYPE: {overview['study_type']}\n")
            f.write(f"JOB DOMAIN: {overview['job_domain']}\n")
            f.write(f"MEASUREMENT TYPE: {overview['measurement_type']}\n")
            f.write(f"SAMPLE SIZE (N): {overview['sample_n']}\n")
            f.write(f"SAMPLE CONTEXT: {overview['sample_context']}\n\n")
            
            # Paper Summary
            f.write("📋 PAPER SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write(f"{output_data['PAPER_SUMMARY']}\n\n")
            
            # Predictors (Key for AI processing)
            f.write("🎯 JOB PERFORMANCE PREDICTORS IDENTIFIED\n")
            f.write("-" * 50 + "\n")
            for i, predictor in enumerate(output_data['PREDICTORS_IDENTIFIED'], 1):
                f.write(f"{i}. PREDICTOR: {predictor['predictor']}\n")
                f.write(f"   CATEGORY: {predictor['category']}\n")
                f.write(f"   CONFIDENCE: {predictor['confidence']}\n")
                f.write(f"   CONTEXT: {predictor['context']}\n\n")
            
            # Statistical Findings (Key for AI processing)
            f.write("📊 ORIGINAL STATISTICAL FINDINGS\n")
            f.write("-" * 50 + "\n")
            
            stats = output_data['STATISTICAL_FINDINGS']
            
            if stats['correlations']:
                f.write("CORRELATIONS FOUND:\n")
                for i, corr in enumerate(stats['correlations'], 1):
                    f.write(f"  {i}. VALUE: {corr['value']}\n")
                    f.write(f"     CONFIDENCE: {corr['confidence']}\n")
                    f.write(f"     CONTEXT: {corr['context']}\n\n")
            
            if stats['regressions']:
                f.write("REGRESSION COEFFICIENTS FOUND:\n")
                for i, reg in enumerate(stats['regressions'], 1):
                    f.write(f"  {i}. VALUE: {reg['value']}\n")
                    f.write(f"     CONFIDENCE: {reg['confidence']}\n")
                    f.write(f"     CONTEXT: {reg['context']}\n\n")
            
            if stats['p_values']:
                f.write(f"P-VALUES: {', '.join(stats['p_values'])}\n")
            if stats['r_squared']:
                f.write(f"R-SQUARED VALUES: {', '.join(stats['r_squared'])}\n")
            if stats['odds_ratios']:
                f.write(f"ODDS RATIOS: {', '.join(stats['odds_ratios'])}\n")
            f.write("\n")
            
            # Tables (Important for comprehensive data)
            f.write("📈 STATISTICAL TABLES EXTRACTED\n")
            f.write("-" * 50 + "\n")
            for table in output_data['TABLES_EXTRACTED']:
                if table['contains_statistics']:
                    f.write(f"TABLE {table['table_index']} (Page {table['page']}) - TYPE: {table['table_type']}\n")
                    f.write("TABLE CONTENT:\n")
                    f.write(table['table_content'])
                    f.write("\n" + "-" * 30 + "\n\n")
            
            # Research Context (For AI to understand study design)
            f.write("🔍 RESEARCH CONTEXT FOR AI ANALYSIS\n")
            f.write("-" * 50 + "\n")
            context = output_data['RESEARCH_CONTEXT']
            
            f.write("METHODS SECTION EXCERPT:\n")
            f.write(context['methods_section'])
            f.write("\n\n")
            
            f.write("RESULTS SECTION EXCERPT:\n")
            f.write(context['results_section'])
            f.write("\n\n")
            
            f.write("DISCUSSION SECTION EXCERPT:\n")
            f.write(context['discussion_section'])
            f.write("\n\n")
            
            # Notes (Critical information for AI)
            f.write("📝 IMPORTANT NOTES FOR AI PROCESSING\n")
            f.write("-" * 50 + "\n")
            for i, note in enumerate(output_data['NOTES'], 1):
                f.write(f"{i}. {note}\n")
            f.write("\n")
            
            # Quality indicators
            f.write("✅ EXTRACTION QUALITY INDICATORS\n")
            f.write("-" * 50 + "\n")
            quality = output_data['EXTRACTION_QUALITY']
            f.write(f"CONFIDENCE LEVEL: {quality['confidence_level']}\n")
            f.write(f"SECTIONS IDENTIFIED: {quality['sections_found']}\n")
            f.write(f"TABLES FOUND: {quality['tables_found']}\n")
            f.write(f"STATISTICAL TABLES: {quality['statistical_tables']}\n")
            f.write(f"PREDICTORS IDENTIFIED: {quality['predictors_identified']}\n")
            f.write(f"STATISTICS EXTRACTED: {quality['statistics_extracted']}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("END OF EXTRACTION - READY FOR AI ANALYSIS\n")
            f.write("=" * 100 + "\n")
    
    def process_research_paper(self, pdf_path: str, output_dir: str = None, web_metadata: Dict = None) -> str:
        """Main method to process a single research paper comprehensively."""
        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)
        
        print(f"\n🚀 PROCESSING: {os.path.basename(pdf_path)}")
        print("=" * 80)
        
        try:
            # Extract text and basic info
            full_text, pdf_metadata = self.extract_text_from_pdf(pdf_path)
            
            if not full_text.strip():
                print("❌ No text extracted from PDF")
                return ""
            
            print(f"✅ Extracted {len(full_text):,} characters")
            
            # Extract tables
            tables = self.extract_tables_comprehensive(pdf_path)
            print(f"✅ Found {len(tables)} tables ({len([t for t in tables if t['contains_statistics']])} statistical)")
            
            # Identify sections
            sections = self.identify_paper_sections(full_text)
            print(f"✅ Identified sections: {list(sections.keys())}")
            
            # Extract metadata
            metadata = self.extract_study_metadata(full_text, pdf_path)
            
            # MERGE WEB METADATA IF PROVIDED - ADD THIS
            if web_metadata:
                # Prioritize web metadata over PDF-extracted metadata
                metadata.update({
                    'title': web_metadata.get('web_title', metadata.get('title', 'Unknown Title')),
                    'authors': web_metadata.get('web_authors', metadata.get('authors', 'Unknown Authors')),
                    'year': web_metadata.get('web_year', metadata.get('year', 'Unknown')),
                    'url': web_metadata.get('web_url', ''),
                    'journal': web_metadata.get('web_journal', 'Unknown Journal'),
                    'source_database': web_metadata.get('web_source', ''),
                    'citation_count': web_metadata.get('web_citations', 0),
                    'search_query': web_metadata.get('search_query', ''),
                    'doi': self.extract_doi_from_text(full_text)  # Extract DOI from PDF
                })

            print(f"✅ Extracted metadata: {metadata['title'][:50]}...")
            
            # Extract characteristics
            characteristics = self.identify_study_characteristics(full_text, sections)
            print(f"✅ Study type: {characteristics['study_type']}, Domain: {characteristics['job_domain']}")
            
            # Extract statistics (excluding cited ones)
            statistics = self.extract_comprehensive_statistics(full_text, sections)
            total_stats = sum(len(stats) for stats in statistics.values())
            print(f"✅ Extracted {total_stats} original statistical findings")
            
            # Extract predictors
            predictors = self.extract_predictors_with_context(full_text, sections)
            print(f"✅ Identified {len(predictors)} job performance predictors")
            
            # Create comprehensive output
            output_data = self.create_comprehensive_output(
                pdf_path, full_text, sections, tables, statistics, 
                predictors, metadata, characteristics
            )
            
            # Save outputs
            filename_base = Path(pdf_path).stem
            
            # JSON output for programmatic use
            json_output_path = os.path.join(output_dir, f"{filename_base}_comprehensive_data.json")
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # TXT output for AI processing
            txt_output_path = os.path.join(output_dir, f"{filename_base}_AI_ready.txt")
            self.save_comprehensive_txt_output(output_data, txt_output_path)
            
            print(f"\n✅ PROCESSING COMPLETE!")
            print(f"📁 JSON Output: {json_output_path}")
            print(f"📄 AI-Ready TXT: {txt_output_path}")
            
            return txt_output_path
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            import traceback
            traceback.print_exc()
            return ""

def main():
    """Main function for standalone execution."""
    print("🔬 Comprehensive Research Paper Extractor")
    print("=" * 60)
    
    # Configuration - modify these paths as needed
    input_path = r"C:\Users\eruku\Akshith\AI_Internship_2025\papers\Selecting_Leadership_An_Analysis_of_Predictors_in_Assessing_Leadership_Potential.pdf"
    output_dir = r"C:\Users\eruku\Akshith\AI_Internship_2025\output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = ComprehensiveResearchExtractor()
    
    # Process PDF
    if os.path.isfile(input_path) and input_path.endswith('.pdf'):
        result = extractor.process_research_paper(input_path, output_dir)
        if result:
            print(f"\n🎉 Success! AI-ready file created: {result}")
        else:
            print("\n❌ Processing failed")
    else:
        print("❌ Invalid input path. Please provide a valid PDF file.")

if __name__ == "__main__":
    main()