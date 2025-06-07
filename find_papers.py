import os
import re
import time
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import warnings
warnings.filterwarnings('ignore')

# Web scraping
try:
    import requests
    from bs4 import BeautifulSoup
    import selenium
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
except ImportError:
    print("Please install: pip install requests beautifulsoup4 selenium")

# Import our comprehensive extractor
try:
    from research_extractor import ComprehensiveResearchExtractor
except ImportError:
    print("Make sure comprehensive_extractor.py is in the same directory")

class ResearchPaperScraper:
    def __init__(self, papers_dir: str = "papers", output_dir: str = "output"):
        """Initialize the research paper scraper."""
        print("🔍 Initializing Research Paper Scraper...")
        
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        
        # Create directories
        os.makedirs(self.papers_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize extractor
        try:
            self.extractor = ComprehensiveResearchExtractor()
        except:
            print("⚠️ Extractor not available, will only download papers")
            self.extractor = None
        
        # Enhanced search queries - categorized for comprehensive coverage
        self.search_queries = self._build_comprehensive_queries()
        
        # Headers for web requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Results storage
        self.found_papers = []
        
        print("✅ Scraper initialized!")
    
    def _build_comprehensive_queries(self) -> List[str]:
        """Build comprehensive search queries targeting meta-analyses and correlation studies."""
        
        # Core meta-analysis queries
        meta_analysis_queries = [
            "meta-analysis job performance predictors validity",
            "meta-analysis personnel selection validity coefficients",
            "meta-analysis cognitive ability job performance correlation",
            "meta-analysis personality job performance validity",
            "meta-analysis work experience job performance correlation",
            "meta-analysis structured interview validity job performance",
            "meta-analysis assessment center job performance correlation",
            "meta-analysis biographical data job performance prediction",
            "meta-analysis emotional intelligence job performance correlation",
            "meta-analysis leadership effectiveness job performance correlation",
            "meta-analysis teamwork skills job performance validity",
            "meta-analysis communication skills job performance correlation"
        ]
        
        # Specific statistical correlation queries
        correlation_queries = [
            "job performance correlation coefficient cognitive ability",
            "job performance correlation r personality traits",
            "job performance correlation coefficient work experience",
            "job performance correlation r education level",
            "job performance correlation coefficient interview scores",
            "job performance correlation r assessment center ratings",
            "job performance correlation coefficient biographical predictors",
            "job performance correlation r emotional intelligence measures",
            "job performance correlation coefficient leadership ratings",
            "job performance correlation r teamwork effectiveness",
            "job performance correlation coefficient communication skills",
            "job performance validity coefficient selection methods"
        ]
        
        # Validity generalization studies
        validity_generalization = [
            "validity generalization job performance predictors",
            "validity generalization cognitive ability job performance",
            "validity generalization personality job performance",
            "validity generalization structured interview job performance",
            "validity generalization assessment center job performance",
            "validity generalization biographical data job performance",
            "validity generalization work sample job performance",
            "validity generalization situational judgment test job performance"
        ]
        
        # Hunter & Schmidt style meta-analyses
        hunter_schmidt_queries = [
            "Hunter Schmidt meta-analysis job performance",
            "validity generalization Hunter Schmidt job performance",
            "corrected correlation job performance predictors",
            "true score correlation job performance",
            "artifact correction job performance correlation",
            "population correlation job performance predictors"
        ]
        
        # Specific predictor categories with statistical focus
        predictor_specific = [
            "Big Five personality job performance correlation meta-analysis",
            "conscientiousness job performance correlation coefficient",
            "general mental ability job performance correlation meta-analysis",
            "cognitive ability test validity job performance correlation",
            "structured behavioral interview validity correlation job performance",
            "situational interview validity correlation job performance",
            "work sample test validity correlation job performance",
            "situational judgment test validity correlation job performance",
            "assessment center validity correlation job performance",
            "biodata validity correlation job performance",
            "reference check validity correlation job performance",
            "education validity correlation job performance",
            "work experience validity correlation job performance",
            "training performance correlation job performance",
            "emotional intelligence EQ correlation job performance meta-analysis"
        ]
        
        # Journal-specific and methodology queries
        methodological_queries = [
            "Personnel Psychology meta-analysis job performance",
            "Journal Applied Psychology meta-analysis job performance correlation",
            "Psychological Bulletin meta-analysis job performance",
            "Schmidt Hunter validity generalization job performance",
            "Ones Viswesvaran meta-analysis job performance",
            "Barrick Mount personality job performance meta-analysis",
            "Judge Bono personality job performance meta-analysis",
            "Salgado cognitive ability job performance meta-analysis",
            "McDaniel structured interview meta-analysis job performance",
            "Arthur Bennett assessment center meta-analysis job performance"
        ]
        
        # Performance criteria specific queries
        performance_criteria = [
            "task performance correlation predictors meta-analysis",
            "contextual performance correlation predictors meta-analysis",
            "overall job performance correlation predictors meta-analysis",
            "supervisor rating correlation predictors meta-analysis",
            "objective performance correlation predictors meta-analysis",
            "sales performance correlation predictors meta-analysis",
            "managerial performance correlation predictors meta-analysis",
            "training performance correlation predictors meta-analysis"
        ]
        
        # Advanced statistical terms
        advanced_statistical = [
            "corrected validity coefficient job performance",
            "operational validity job performance correlation",
            "criterion-related validity job performance correlation",
            "predictive validity job performance correlation coefficient",
            "concurrent validity job performance correlation coefficient",
            "incremental validity job performance predictors correlation",
            "multiple correlation job performance predictors",
            "beta weights job performance predictors correlation"
        ]
        
        # Combine all queries
        all_queries = (meta_analysis_queries + correlation_queries + validity_generalization + 
                      hunter_schmidt_queries + predictor_specific + methodological_queries + 
                      performance_criteria + advanced_statistical)
        
        return all_queries
    
    def setup_selenium_driver(self):
        """Setup Selenium WebDriver with Chrome."""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # Run in background
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'--user-agent={self.headers["User-Agent"]}')
            
            # Try to create driver
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            print(f"⚠️ Selenium setup failed: {e}")
            print("Please install ChromeDriver: https://chromedriver.chromium.org/")
            return None
    
    def search_google_scholar(self, query: str, max_results: int = 20) -> List[Dict]:
        """Enhanced Google Scholar search with better filtering for meta-analyses."""
        print(f"🔍 Searching Google Scholar: {query}")
        
        papers = []
        driver = self.setup_selenium_driver()
        
        if not driver:
            print("⚠️ Cannot search Google Scholar without Selenium")
            return papers
        
        try:
            # Enhanced query formatting for Google Scholar
            enhanced_query = self._enhance_query_for_scholar(query)
            search_url = f"https://scholar.google.com/scholar?q={enhanced_query}&hl=en&as_ylo=1990&as_yhi=2024"
            driver.get(search_url)
            
            # Wait for results to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "gs_rt"))
            )
            
            # Extract paper information
            results = driver.find_elements(By.CLASS_NAME, "gs_ri")
            
            for i, result in enumerate(results[:max_results]):
                try:
                    paper_info = self.extract_google_scholar_info(result)
                    if paper_info and self._is_relevant_paper(paper_info):
                        paper_info['search_query'] = query
                        paper_info['source'] = 'google_scholar'
                        paper_info['relevance_score'] = self._calculate_relevance_score(paper_info, query)
                        papers.append(paper_info)
                        print(f"  📄 Found: {paper_info['title'][:60]}... (Score: {paper_info['relevance_score']:.2f})")
                except Exception as e:
                    print(f"  ⚠️ Error extracting paper {i}: {e}")
                    continue
            
        except TimeoutException:
            print("⚠️ Google Scholar search timed out")
        except Exception as e:
            print(f"⚠️ Google Scholar search failed: {e}")
        finally:
            driver.quit()
        
        # Sort by relevance score
        papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return papers
    
    def _enhance_query_for_scholar(self, query: str) -> str:
        """Enhance query specifically for Google Scholar with advanced operators."""
        # Add quotation marks for exact phrases
        if 'meta-analysis' in query:
            query = query.replace('meta-analysis', '"meta-analysis"')
        if 'correlation' in query:
            query = query.replace('correlation', '"correlation"')
        if 'validity' in query:
            query = query.replace('validity', '"validity"')
        
        # Add additional terms to increase precision
        if 'job performance' in query:
            query += ' ("criterion validity" OR "predictive validity" OR "r =" OR "correlation coefficient")'
        
        return query.replace(' ', '+')
    
    def _is_relevant_paper(self, paper_info: Dict) -> bool:
        """Check if paper is relevant based on title, abstract, and other factors."""
        title = paper_info.get('title', '').lower()
        abstract = paper_info.get('abstract', '').lower()
        full_text = f"{title} {abstract}"
        
        # High relevance indicators
        high_relevance_terms = [
            'meta-analysis', 'meta analysis', 'validity generalization',
            'correlation coefficient', 'r =', 'r=', 'validity coefficient',
            'corrected correlation', 'true score correlation',
            'hunter schmidt', 'personnel psychology', 'job performance'
        ]
        
        # Must have at least one high relevance term
        has_high_relevance = any(term in full_text for term in high_relevance_terms)
        
        # Statistical indicators
        statistical_terms = [
            'correlation', 'validity', 'coefficient', 'r =', 'r=',
            'statistical significance', 'effect size', 'confidence interval'
        ]
        has_statistical = any(term in full_text for term in statistical_terms)
        
        # Performance-related terms
        performance_terms = [
            'job performance', 'work performance', 'task performance',
            'contextual performance', 'overall performance', 'supervisor rating',
            'performance rating', 'performance criteria', 'performance measure'
        ]
        has_performance = any(term in full_text for term in performance_terms)
        
        return has_high_relevance and has_statistical and has_performance
    
    def _calculate_relevance_score(self, paper_info: Dict, query: str) -> float:
        """Calculate relevance score for ranking papers."""
        title = paper_info.get('title', '').lower()
        abstract = paper_info.get('abstract', '').lower()
        citations = paper_info.get('citations', 0)
        year = paper_info.get('year', '0')
        
        full_text = f"{title} {abstract}"
        score = 0.0
        
        # Meta-analysis bonus (highest priority)
        if any(term in full_text for term in ['meta-analysis', 'meta analysis', 'validity generalization']):
            score += 3.0
        
        # Statistical correlation terms
        correlation_terms = ['correlation coefficient', 'r =', 'r=', 'validity coefficient', 'corrected correlation']
        score += sum(2.0 for term in correlation_terms if term in full_text)
        
        # Job performance terms
        performance_terms = ['job performance', 'work performance', 'task performance']
        score += sum(1.5 for term in performance_terms if term in full_text)
        
        # Predictor terms from query
        query_terms = query.lower().split()
        score += sum(0.5 for term in query_terms if term in full_text and len(term) > 3)
        
        # Citation bonus (quality indicator)
        if citations > 100:
            score += 1.0
        elif citations > 50:
            score += 0.5
        elif citations > 10:
            score += 0.25
        
        # Recency bonus
        try:
            year_int = int(year)
            if year_int >= 2010:
                score += 0.5
            if year_int >= 2015:
                score += 0.25
        except:
            pass
        
        # Journal quality indicators
        journal_terms = ['personnel psychology', 'journal applied psychology', 'psychological bulletin']
        if any(term in paper_info.get('authors_and_publication', '').lower() for term in journal_terms):
            score += 1.0
        
        return score
    
    def extract_google_scholar_info(self, result_element) -> Optional[Dict]:
        """Enhanced extraction of paper information from Google Scholar result."""
        paper_info = {}
        
        try:
            # Title and link
            title_element = result_element.find_element(By.CLASS_NAME, "gs_rt")
            title_link = title_element.find_element(By.TAG_NAME, "a")
            paper_info['title'] = title_link.text.strip()
            paper_info['url'] = title_link.get_attribute('href')
            
            # Authors and publication info
            author_element = result_element.find_element(By.CLASS_NAME, "gs_a")
            author_text = author_element.text
            paper_info['authors_and_publication'] = author_text
            
            # Extract year if possible
            year_match = re.search(r'\b(19|20)\d{2}\b', author_text)
            paper_info['year'] = year_match.group() if year_match else 'Unknown'

            # Enhanced journal extraction
            try:
                # Pattern: "Author - Journal, Year - Publisher"
                journal_patterns = [
                    r'-\s*([^,\-]+(?:Journal|Review|Bulletin|Psychology|Management)[^,\-]*)',
                    r'-\s*([A-Za-z\s]+(?:of|in|for)[^,\-]+)',
                    r'-\s*(Personnel Psychology|Journal of Applied Psychology|Psychological Bulletin|Academy of Management Journal)',
                    r'-\s*([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
                ]
                
                journal = 'Unknown Journal'
                for pattern in journal_patterns:
                    journal_match = re.search(pattern, author_text, re.IGNORECASE)
                    if journal_match:
                        journal = journal_match.group(1).strip()
                        break
                
                paper_info['journal'] = journal
                
            except NoSuchElementException:
                paper_info['journal'] = 'Unknown Journal'
                
            # Enhanced abstract/snippet extraction
            try:
                snippet_element = result_element.find_element(By.CLASS_NAME, "gs_rs")
                abstract_text = snippet_element.text.strip()
                # Clean up the abstract
                abstract_text = re.sub(r'\s+', ' ', abstract_text)
                paper_info['abstract'] = abstract_text
            except NoSuchElementException:
                paper_info['abstract'] = ''
            
            # Enhanced citation count extraction
            try:
                citation_element = result_element.find_element(By.PARTIAL_LINK_TEXT, "Cited by")
                citation_text = citation_element.text
                citation_match = re.search(r'Cited by (\d+)', citation_text)
                paper_info['citations'] = int(citation_match.group(1)) if citation_match else 0
            except NoSuchElementException:
                paper_info['citations'] = 0
            
            # Enhanced PDF link detection
            pdf_links = []
            try:
                # Look for various PDF indicators
                pdf_selectors = [
                    'a[href*=".pdf"]',
                    'a[title*="PDF"]',
                    'a:contains("[PDF]")',
                    'a[href*="researchgate.net/publication"]',
                    'a[href*="arxiv.org/pdf"]'
                ]
                
                for selector in pdf_selectors:
                    try:
                        pdf_elements = result_element.find_elements(By.CSS_SELECTOR, selector)
                        for pdf_elem in pdf_elements:
                            pdf_url = pdf_elem.get_attribute('href')
                            if pdf_url and pdf_url not in pdf_links:
                                pdf_links.append(pdf_url)
                    except:
                        continue
                        
                # Also check for [PDF] text links
                try:
                    pdf_text_elements = result_element.find_elements(By.PARTIAL_LINK_TEXT, "[PDF]")
                    for pdf_elem in pdf_text_elements:
                        pdf_url = pdf_elem.get_attribute('href')
                        if pdf_url and pdf_url not in pdf_links:
                            pdf_links.append(pdf_url)
                except:
                    pass
                    
            except:
                pass
            
            paper_info['pdf_links'] = pdf_links
            
            return paper_info
            
        except Exception as e:
            print(f"⚠️ Error extracting paper info: {e}")
            return None
    
    def search_researchgate(self, query: str, max_results: int = 15) -> List[Dict]:
        """Enhanced ResearchGate search with better filtering."""
        print(f"🔍 Searching ResearchGate: {query}")
        
        papers = []
        
        try:
            # Enhanced query with filters
            enhanced_query = f"{query} meta-analysis correlation"
            search_url = f"https://www.researchgate.net/search/publication?q={enhanced_query.replace(' ', '%20')}"
            response = requests.get(search_url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Enhanced ResearchGate parsing
                publication_items = soup.find_all(['div', 'article'], class_=re.compile(r'publication|result'))
                
                for item in publication_items[:max_results]:
                    try:
                        # Try multiple selectors for title
                        title_elem = (item.find('a', href=re.compile(r'/publication/\d+')) or
                                    item.find('h3') or
                                    item.find('h4') or
                                    item.find('a', class_=re.compile(r'title|name')))
                        
                        if title_elem:
                            title = title_elem.get_text().strip()
                            if len(title) > 20 and self._is_relevant_title(title):
                                url = title_elem.get('href', '')
                                if url and not url.startswith('http'):
                                    url = urljoin('https://www.researchgate.net', url)
                                
                                # Try to extract additional info
                                authors = ''
                                year = 'Unknown'
                                abstract = ''
                                
                                # Look for authors
                                author_elem = item.find(['span', 'div'], class_=re.compile(r'author|name'))
                                if author_elem:
                                    authors = author_elem.get_text().strip()
                                
                                # Look for year
                                year_match = re.search(r'\b(19|20)\d{2}\b', item.get_text())
                                if year_match:
                                    year = year_match.group()
                                
                                # Look for abstract snippet
                                abstract_elem = item.find(['p', 'div'], class_=re.compile(r'abstract|summary|description'))
                                if abstract_elem:
                                    abstract = abstract_elem.get_text().strip()[:500]
                                
                                paper_info = {
                                    'title': title,
                                    'url': url,
                                    'source': 'researchgate',
                                    'search_query': query,
                                    'authors_and_publication': authors or 'ResearchGate',
                                    'year': year,
                                    'abstract': abstract,
                                    'citations': 0,
                                    'pdf_links': [url] if url else [],
                                    'journal': 'ResearchGate',
                                    'relevance_score': self._calculate_relevance_score_simple(title, abstract, query)
                                }
                                papers.append(paper_info)
                                print(f"  📄 Found: {title[:60]}...")
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"⚠️ ResearchGate search failed: {e}")
        
        # Sort by relevance
        papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return papers
    
    def _is_relevant_title(self, title: str) -> bool:
        """Check if title indicates a relevant research paper."""
        title_lower = title.lower()
        
        # Must contain job/work performance
        performance_terms = ['job performance', 'work performance', 'performance', 'effectiveness']
        has_performance = any(term in title_lower for term in performance_terms)
        
        # Must contain statistical/correlation terms
        statistical_terms = ['meta-analysis', 'correlation', 'validity', 'prediction', 'predictor']
        has_statistical = any(term in title_lower for term in statistical_terms)
        
        # Exclude irrelevant papers
        exclude_terms = ['financial performance', 'company performance', 'organizational performance', 
                        'firm performance', 'economic performance', 'market performance']
        is_excluded = any(term in title_lower for term in exclude_terms)
        
        return has_performance and has_statistical and not is_excluded
    
    def _calculate_relevance_score_simple(self, title: str, abstract: str, query: str) -> float:
        """Simple relevance score calculation."""
        full_text = f"{title} {abstract}".lower()
        query_terms = query.lower().split()
        
        score = 0.0
        
        # Meta-analysis bonus
        if 'meta-analysis' in full_text or 'meta analysis' in full_text:
            score += 2.0
        
        # Correlation bonus
        if 'correlation' in full_text:
            score += 1.5
        
        # Job performance bonus
        if 'job performance' in full_text:
            score += 1.0
        
        # Query term matches
        for term in query_terms:
            if len(term) > 3 and term in full_text:
                score += 0.5
        
        return score
    
    def search_arxiv(self, query: str, max_results: int = 15) -> List[Dict]:
        """Enhanced arXiv search for psychology/management papers."""
        print(f"🔍 Searching arXiv: {query}")
        
        papers = []
        
        try:
            # Enhanced query for arXiv (focus on relevant categories)
            categories = "cat:cs.HC+OR+cat:stat.AP+OR+cat:q-bio.NC"  # Human-Computer Interaction, Statistics, Neuroscience
            enhanced_query = f"({query.replace(' ', '+')}) AND ({categories})"
            
            api_url = f"http://export.arxiv.org/api/query?search_query={enhanced_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
            response = requests.get(api_url, timeout=15)
            
            if response.status_code == 200:
                try:
                    from xml.etree import ElementTree as ET
                    root = ET.fromstring(response.content)
                    
                    ns = {'atom': 'http://www.w3.org/2005/Atom'}
                    entries = root.findall('atom:entry', ns)
                    
                    for entry in entries:
                        try:
                            title = entry.find('atom:title', ns).text.strip()
                            abstract = entry.find('atom:summary', ns).text.strip()
                            
                            # Filter for relevance
                            if self._is_relevant_arxiv_paper(title, abstract):
                                authors = []
                                for author in entry.findall('atom:author', ns):
                                    name = author.find('atom:name', ns).text
                                    authors.append(name)
                                
                                published = entry.find('atom:published', ns).text[:4]
                                
                                pdf_link = None
                                for link in entry.findall('atom:link', ns):
                                    if link.get('type') == 'application/pdf':
                                        pdf_link = link.get('href')
                                        break
                                
                                paper_info = {
                                    'title': title,
                                    'url': entry.find('atom:id', ns).text,
                                    'source': 'arxiv',
                                    'search_query': query,
                                    'authors_and_publication': ', '.join(authors),
                                    'year': published,
                                    'abstract': abstract,
                                    'citations': 0,
                                    'pdf_links': [pdf_link] if pdf_link else [],
                                    'journal': 'arXiv',
                                    'relevance_score': self._calculate_relevance_score_simple(title, abstract, query)
                                }
                                
                                papers.append(paper_info)
                                print(f"  📄 Found: {title[:60]}...")
                            
                        except Exception as e:
                            continue
                            
                except ImportError:
                    print("⚠️ XML parsing not available")
                    
        except Exception as e:
            print(f"⚠️ arXiv search failed: {e}")
        
        papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return papers
    
    def _is_relevant_arxiv_paper(self, title: str, abstract: str) -> bool:
        """Check if arXiv paper is relevant to job performance research."""
        full_text = f"{title} {abstract}".lower()
        
        # Job performance terms
        performance_terms = ['job performance', 'work performance', 'employee performance', 
                           'workplace performance', 'occupational performance']
        has_performance = any(term in full_text for term in performance_terms)
        
        # Research methodology terms
        method_terms = ['correlation', 'regression', 'meta-analysis', 'statistical analysis',
                       'empirical study', 'survey', 'validation', 'prediction']
        has_method = any(term in full_text for term in method_terms)
        
        return has_performance and has_method
    
    def search_pubmed(self, query: str, max_results: int = 15) -> List[Dict]:
        """Enhanced PubMed search focusing on occupational health and psychology."""
        print(f"🔍 Searching PubMed: {query}")
        
        papers = []
        
        try:
            # Enhanced PubMed query with MeSH terms and filters
            enhanced_query = f"({query}) AND (occupational health[MeSH] OR psychology, industrial[MeSH] OR workplace OR job performance)"
            search_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={enhanced_query.replace(' ', '+')}&size={max_results}&sort=relevance"
            
            response = requests.get(search_url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                articles = soup.find_all('article', class_='full-docsum')
                
                for article in articles:
                    try:
                        title_elem = article.find('a', class_='docsum-title')
                        if title_elem:
                            title = title_elem.get_text().strip()
                            
                            # Filter for relevance
                            if self._is_relevant_title(title):
                                url = urljoin('https://pubmed.ncbi.nlm.nih.gov', title_elem['href'])
                                
                                # Authors
                                authors_elem = article.find('span', class_='docsum-authors')
                                authors = authors_elem.get_text().strip() if authors_elem else 'Unknown'
                                
                                # Journal and year
                                journal_elem = article.find('span', class_='docsum-journal-citation')
                                journal_text = journal_elem.get_text() if journal_elem else ''
                                
                                year_match = re.search(r'\b(19|20)\d{2}\b', journal_text)
                                year = year_match.group() if year_match else 'Unknown'
                                
                                # Extract journal name
                                journal_match = re.search(r'^([^.]+)', journal_text)
                                journal = journal_match.group(1).strip() if journal_match else 'Unknown Journal'
                                
                                # Abstract (if available)
                                abstract_elem = article.find('span', class_='docsum-snippet')
                                abstract = abstract_elem.get_text().strip() if abstract_elem else ''
                                
                                paper_info = {
                                    'title': title,
                                    'url': url,
                                    'source': 'pubmed',
                                    'search_query': query,
                                    'authors_and_publication': f"{authors} - {journal}",
                                    'year': year,
                                    'abstract': abstract,
                                    'citations': 0,
                                    'pdf_links': [],
                                    'journal': journal,
                                    'relevance_score': self._calculate_relevance_score_simple(title, abstract, query)
                                }
                                
                                papers.append(paper_info)
                                print(f"  📄 Found: {title[:60]}...")
                            
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"⚠️ PubMed search failed: {e}")
        
        papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return papers
    
    def download_pdf(self, pdf_url: str, filename: str) -> bool:
        """Download PDF from URL, or extract web content if not a PDF."""
        try:
            print(f"⬇️ Downloading: {filename}")
            
            # First, check if this is actually a PDF URL
            response = requests.head(pdf_url, headers=self.headers, timeout=10, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            
            # Check if it's actually a PDF
            if 'application/pdf' in content_type or pdf_url.lower().endswith('.pdf'):
                print(f"✅ Confirmed PDF content type: {content_type}")
                
                # Download the actual PDF
                response = requests.get(pdf_url, headers=self.headers, timeout=30, stream=True)
                
                if response.status_code == 200:
                    filepath = os.path.join(self.papers_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"✅ Downloaded PDF: {filepath}")
                    return True
                else:
                    print(f"❌ PDF download failed: HTTP {response.status_code}")
                    return False
            else:
                print(f"⚠️ Not a PDF (content-type: {content_type})")
                print(f"🔄 Extracting content from webpage instead...")
                
                # Extract content from the webpage
                return self.extract_and_save_web_content(pdf_url, filename)
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            print(f"🔄 Attempting to extract webpage content as fallback...")
            return self.extract_and_save_web_content(pdf_url, filename)

    def extract_and_save_web_content(self, url: str, filename: str) -> bool:
        """Extract content from webpage and save as text file."""
        try:
            print(f"🌐 Extracting content from: {url[:60]}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            paper_data = {
                'url': url,
                'title': '',
                'authors': '',
                'year': '',
                'journal': '',
                'abstract': '',
                'doi': '',
                'keywords': [],
                'study_type': '',
                'full_text_sections': []
            }
            
            # Extract title (multiple strategies)
            title_selectors = [
                'h1.title', 'h1#title', '.title', 'h1',
                '[data-test="paper-title"]', '.paper-title',
                'meta[name="citation_title"]', '.article-title',
                '#title', '.entry-title', '.post-title'
            ]
            
            for selector in title_selectors:
                if selector.startswith('meta'):
                    meta_tag = soup.find('meta', attrs={'name': 'citation_title'})
                    if meta_tag and meta_tag.get('content'):
                        paper_data['title'] = meta_tag.get('content', '').strip()
                        break
                else:
                    title_elem = soup.select_one(selector)
                    if title_elem and title_elem.get_text().strip():
                        paper_data['title'] = title_elem.get_text().strip()
                        break
            
            # Extract authors (multiple strategies)
            author_selectors = [
                '[data-test="author-list"]', '.authors', '.author-list',
                'meta[name="citation_author"]', '.author', '[class*="author"]',
                '.byline', '.contributor', '#authors'
            ]
            
            authors_list = []
            
            # Try meta tags first
            meta_authors = soup.find_all('meta', attrs={'name': 'citation_author'})
            if meta_authors:
                authors_list = [tag.get('content', '') for tag in meta_authors if tag.get('content')]
            
            if not authors_list:
                for selector in author_selectors:
                    if not selector.startswith('meta'):
                        author_elem = soup.select_one(selector)
                        if author_elem:
                            author_text = author_elem.get_text().strip()
                            if author_text:
                                # Handle different author formats
                                if ',' in author_text or ' and ' in author_text:
                                    authors_list = [author_text]
                                else:
                                    authors_list = [author_text]
                                break
            
            paper_data['authors'] = ', '.join(authors_list) if authors_list else 'Unknown Authors'
            
            # Extract year (multiple strategies)
            year_sources = [
                soup.find('meta', attrs={'name': 'citation_publication_date'}),
                soup.find('meta', attrs={'name': 'citation_date'}),
                soup.find('meta', attrs={'name': 'citation_year'})
            ]
            
            for source in year_sources:
                if source and source.get('content'):
                    date_content = source.get('content', '')
                    year_match = re.search(r'(\d{4})', date_content)
                    if year_match:
                        paper_data['year'] = year_match.group(1)
                        break
            
            if not paper_data['year']:
                # Look in page text
                page_text = soup.get_text()
                year_matches = re.findall(r'\b(20\d{2}|19[89]\d)\b', page_text)
                if year_matches:
                    # Get the most recent reasonable year
                    years = [int(y) for y in year_matches if 1990 <= int(y) <= 2025]
                    if years:
                        paper_data['year'] = str(max(set(years), key=years.count))
            
            # Extract journal
            journal_sources = [
                soup.find('meta', attrs={'name': 'citation_journal_title'}),
                soup.find('meta', attrs={'name': 'citation_conference_title'}),
                soup.select_one('.journal-title'),
                soup.select_one('.publication'),
                soup.select_one('[class*="journal"]')
            ]
            
            for source in journal_sources:
                if source:
                    if hasattr(source, 'get') and source.get('content'):
                        paper_data['journal'] = source.get('content', '').strip()
                        break
                    elif hasattr(source, 'get_text'):
                        journal_text = source.get_text().strip()
                        if journal_text:
                            paper_data['journal'] = journal_text
                            break
            
            # Extract DOI
            doi_sources = [
                soup.find('meta', attrs={'name': 'citation_doi'}),
                soup.select_one('a[href*="doi.org"]'),
                soup.select_one('[class*="doi"]')
            ]
            
            for source in doi_sources:
                if source:
                    if hasattr(source, 'get') and source.get('content'):
                        paper_data['doi'] = source.get('content', '').strip()
                        break
                    elif hasattr(source, 'get'):
                        doi_text = source.get('href', '') or source.get_text()
                        doi_match = re.search(r'10\.\d+/[^\s]+', doi_text)
                        if doi_match:
                            paper_data['doi'] = doi_match.group(0)
                            break
            
            # Extract abstract (multiple strategies)
            abstract_selectors = [
                '#abstract', '.abstract', '[data-test="abstract"]',
                '.abstract-content', '.summary', '[class*="abstract"]',
                '.paper-abstract', '#summary'
            ]
            
            for selector in abstract_selectors:
                abstract_elem = soup.select_one(selector)
                if abstract_elem:
                    abstract_text = abstract_elem.get_text().strip()
                    # Clean up abstract text
                    abstract_text = re.sub(r'\s+', ' ', abstract_text)
                    abstract_text = re.sub(r'^abstract:?\s*', '', abstract_text, flags=re.IGNORECASE)
                    if len(abstract_text) > 50:  # Reasonable abstract length
                        paper_data['abstract'] = abstract_text
                        break
            
            # If no abstract found, look for sections containing "abstract"
            if not paper_data['abstract']:
                for elem in soup.find_all(['div', 'section', 'p']):
                    if elem.get_text() and 'abstract' in elem.get_text().lower()[:100]:
                        text = elem.get_text().strip()
                        if len(text) > 100:
                            paper_data['abstract'] = re.sub(r'\s+', ' ', text)
                            break
            
            # Extract keywords
            keyword_sources = [
                soup.find('meta', attrs={'name': 'citation_keywords'}),
                soup.select_one('.keywords'),
                soup.select_one('[class*="keyword"]')
            ]
            
            for source in keyword_sources:
                if source:
                    if hasattr(source, 'get') and source.get('content'):
                        keywords = source.get('content', '').split(',')
                        paper_data['keywords'] = [k.strip() for k in keywords if k.strip()]
                        break
                    elif hasattr(source, 'get_text'):
                        keyword_text = source.get_text()
                        keywords = keyword_text.split(',')
                        paper_data['keywords'] = [k.strip() for k in keywords if k.strip()]
                        break
            
            # Determine study type from content
            full_text = f"{paper_data['title']} {paper_data['abstract']}".lower()
            study_types = {
                'meta-analysis': ['meta-analysis', 'systematic review', 'meta analysis'],
                'longitudinal': ['longitudinal', 'panel study', 'over time', 'longitudinal study'],
                'cross-sectional': ['cross-sectional', 'survey', 'cross sectional'],
                'experimental': ['experiment', 'randomized', 'controlled trial', 'rct'],
                'case study': ['case study', 'case report'],
                'qualitative': ['qualitative', 'interview', 'focus group'],
                'review': ['review', 'literature review']
            }
            
            for study_type, keywords in study_types.items():
                if any(keyword in full_text for keyword in keywords):
                    paper_data['study_type'] = study_type
                    break
            
            # Extract additional content sections
            content_selectors = [
                '.content', '.main-content', '.article-content',
                '#content', '.paper-content', '.text-content'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Extract paragraphs
                    paragraphs = content_elem.find_all('p')
                    for p in paragraphs[:10]:  # Limit to first 10 paragraphs
                        text = p.get_text().strip()
                        if len(text) > 50:
                            paper_data['full_text_sections'].append(text)
                    break
            
            # Save the extracted content
            content_filename = filename.replace('.pdf', '.txt')
            filepath = os.path.join(self.papers_dir, content_filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("EXTRACTED WEB CONTENT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"SOURCE URL: {paper_data['url']}\n")
                f.write(f"TITLE: {paper_data['title'] or 'Not found'}\n")
                f.write(f"AUTHORS: {paper_data['authors']}\n")
                f.write(f"YEAR: {paper_data['year'] or 'Not found'}\n")
                f.write(f"JOURNAL: {paper_data['journal'] or 'Not found'}\n")
                f.write(f"DOI: {paper_data['doi'] or 'Not found'}\n")
                f.write(f"STUDY TYPE: {paper_data['study_type'] or 'Not determined'}\n")
                f.write(f"KEYWORDS: {', '.join(paper_data['keywords']) if paper_data['keywords'] else 'Not found'}\n")
                f.write(f"\n{'='*50}\n")
                f.write("ABSTRACT:\n")
                f.write(paper_data['abstract'] or 'Abstract not found')
                f.write(f"\n\n{'='*50}\n")
                
                if paper_data['full_text_sections']:
                    f.write("ADDITIONAL CONTENT SECTIONS:\n")
                    f.write("-" * 30 + "\n")
                    for i, section in enumerate(paper_data['full_text_sections'], 1):
                        f.write(f"\nSection {i}:\n{section}\n")
                else:
                    f.write("No additional content sections extracted.\n")
            
            print(f"✅ Web content extracted and saved: {content_filename}")
            print(f"   - Title: {'✅' if paper_data['title'] else '❌'}")
            print(f"   - Authors: {'✅' if paper_data['authors'] != 'Unknown Authors' else '❌'}")
            print(f"   - Year: {'✅' if paper_data['year'] else '❌'}")
            print(f"   - Journal: {'✅' if paper_data['journal'] else '❌'}")
            print(f"   - Abstract: {'✅' if paper_data['abstract'] else '❌'}")
            print(f"   - DOI: {'✅' if paper_data['doi'] else '❌'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Web content extraction failed: {e}")
            return False
    
    def clean_filename(self, title: str) -> str:
        """Clean title for use as filename."""
        # Remove invalid filename characters
        cleaned = re.sub(r'[<>:"/\\|?*]', '', title)
        # Limit length
        cleaned = cleaned[:100]
        # Remove extra spaces
        cleaned = re.sub(r'\s+', '_', cleaned)
        return cleaned + '.pdf'
    
    def search_all_sources(self, query: str, max_per_source: int = 5) -> List[Dict]:
        """Search all available sources for a query."""
        print(f"\n🔍 COMPREHENSIVE SEARCH: {query}")
        print("=" * 60)
        
        all_papers = []
        
        # Search Google Scholar (most comprehensive)
        try:
            scholar_papers = self.search_google_scholar(query, max_per_source)
            all_papers.extend(scholar_papers)
            time.sleep(2)  # Rate limiting
        except Exception as e:
            print(f"⚠️ Google Scholar failed: {e}")
        
        # Search arXiv
        try:
            arxiv_papers = self.search_arxiv(query, max_per_source)
            all_papers.extend(arxiv_papers)
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ arXiv failed: {e}")
        
        # Search PubMed
        try:
            pubmed_papers = self.search_pubmed(query, max_per_source)
            all_papers.extend(pubmed_papers)
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ PubMed failed: {e}")
        
        # Search ResearchGate (last due to rate limiting)
        try:
            rg_papers = self.search_researchgate(query, max_per_source)
            all_papers.extend(rg_papers)
            time.sleep(3)  # Longer delay for ResearchGate
        except Exception as e:
            print(f"⚠️ ResearchGate failed: {e}")
        
        return all_papers
    
    def deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity."""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper['title'].lower().strip()
            # Simple deduplication based on first 50 characters
            title_key = title[:50]
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        return unique_papers
    
    def prioritize_papers(self, papers: List[Dict]) -> List[Dict]:
        """Prioritize papers based on relevance and quality indicators."""
        
        def calculate_score(paper):
            score = 0
            
            # Citation count (if available)
            score += min(paper.get('citations', 0) / 10, 10)  # Max 10 points
            
            # Recent papers get higher scores
            try:
                year = int(paper.get('year', 0))
                if year >= 2020:
                    score += 5
                elif year >= 2015:
                    score += 3
                elif year >= 2010:
                    score += 1
            except:
                pass
            
            # Papers with PDFs get higher scores
            if paper.get('pdf_links'):
                score += 3
            
            # Keyword relevance in title
            title_lower = paper['title'].lower()
            relevant_keywords = [
                'job performance', 'performance prediction', 'cognitive ability',
                'personality', 'correlation', 'validity', 'meta-analysis',
                'selection', 'assessment', 'predictor'
            ]
            
            for keyword in relevant_keywords:
                if keyword in title_lower:
                    score += 2
            
            return score
        
        # Sort by score (descending)
        return sorted(papers, key=calculate_score, reverse=True)

    def load_existing_papers(self) -> List[Dict]:
        """Load existing papers from the papers directory."""
        print("📁 LOADING EXISTING PAPERS FROM LOCAL DIRECTORY")
        print("=" * 80)
        
        existing_papers = []
        
        if not os.path.exists(self.papers_dir):
            print(f"❌ Papers directory '{self.papers_dir}' does not exist!")
            return existing_papers
        
        # Get all PDF files in the papers directory
        pdf_files = [f for f in os.listdir(self.papers_dir) if f.lower().endswith('.pdf')]
        
        print(f"📊 Found {len(pdf_files)} PDF files in '{self.papers_dir}'")
        
        for pdf_file in pdf_files:
            # Extract title from filename (removing .pdf extension and cleaning up)
            title = os.path.splitext(pdf_file)[0]
            title = title.replace('_', ' ').replace('-', ' ')
            
            paper_info = {
                'title': title,
                'authors_and_publication': 'Local File - Unknown',
                'year': 'Unknown',
                'source': 'Local Storage',
                'url': '',
                'journal': 'Unknown Journal',
                'citations': 0,
                'pdf_path': os.path.join(self.papers_dir, pdf_file),
                'search_query': 'existing_papers',
                'pdf_links': []  # Empty since we already have the file
            }
            
            existing_papers.append(paper_info)
            print(f"✅ Loaded: {title[:60]}...")
        
        print(f"\n📊 LOAD SUMMARY")
        print("=" * 40)
        print(f"Total papers loaded: {len(existing_papers)}")
        
        return existing_papers

    def run_comprehensive_search(self, max_papers_per_query: int = 5, max_total_papers: int = 50):
        """Run comprehensive search across all queries and sources."""
        print("🚀 STARTING COMPREHENSIVE RESEARCH PAPER SEARCH")
        print("=" * 80)
        
        all_papers = []
        
        for i, query in enumerate(self.search_queries, 1):
            print(f"\n[{i}/{len(self.search_queries)}] Processing query: {query}")
            
            try:
                papers = self.search_all_sources(query, max_papers_per_query)
                all_papers.extend(papers)
                
                print(f"✅ Found {len(papers)} papers for this query")
                
                # Break if we have enough papers
                if len(all_papers) >= max_total_papers:
                    print(f"📊 Reached maximum papers limit ({max_total_papers})")
                    break
                    
            except Exception as e:
                print(f"❌ Error with query '{query}': {e}")
                continue
        
        print(f"\n📊 SEARCH SUMMARY")
        print("=" * 40)
        print(f"Total papers found: {len(all_papers)}")
        
        # Deduplicate
        unique_papers = self.deduplicate_papers(all_papers)
        print(f"Unique papers: {len(unique_papers)}")
        
        # Prioritize
        prioritized_papers = self.prioritize_papers(unique_papers)
        
        # Limit to max total papers
        final_papers = prioritized_papers[:max_total_papers]
        
        print(f"Final papers to download: {len(final_papers)}")
        
        self.found_papers = final_papers
        return final_papers

    def download_and_process_papers(self, papers: List[Dict] = None, max_downloads: int = 20, skip_download: bool = False):
        """Download PDFs and process them with the extractor."""
        if papers is None:
            papers = self.found_papers
        
        action_text = "PROCESSING EXISTING" if skip_download else "DOWNLOADING AND PROCESSING"
        print(f"\n📥 {action_text} {min(len(papers), max_downloads)} PAPERS")
        print("=" * 80)
        
        downloaded_count = 0
        processed_count = 0
        
        # Create summary file
        summary_file = os.path.join(self.output_dir, "download_summary.json")
        summary_data = {
            'downloaded_papers': [],
            'failed_downloads': [],
            'processed_papers': [],
            'processing_errors': [],
            'web_content_extracted': []
        }
        
        for i, paper in enumerate(papers[:max_downloads], 1):
            print(f"\n[{i}/{min(len(papers), max_downloads)}] Processing: {paper['title'][:60]}...")
            
            pdf_path = None
            pdf_downloaded = False
            content_extracted = False
            
            if skip_download:
                # For existing papers, we already have the PDF path
                if paper.get('pdf_path') and os.path.exists(paper['pdf_path']):
                    pdf_path = paper['pdf_path']
                    pdf_downloaded = True
                    downloaded_count += 1
                    print(f"✅ Using existing file: {os.path.basename(pdf_path)}")
                    
                    # Add to summary
                    summary_data['downloaded_papers'].append({
                        'title': paper['title'],
                        'authors': paper.get('authors_and_publication', ''),
                        'year': paper.get('year', ''),
                        'source': paper.get('source', ''),
                        'url': paper.get('url', ''),
                        'pdf_path': pdf_path,
                        'search_query': paper.get('search_query', '')
                    })
                else:
                    print(f"❌ File not found: {paper.get('pdf_path', 'No path')}")
                    summary_data['failed_downloads'].append({
                        'title': paper['title'],
                        'reason': 'File not found locally',
                        'path': paper.get('pdf_path', '')
                    })
                    continue
            else:
                # Try to download or extract content
                if paper.get('pdf_links'):
                    for pdf_url in paper['pdf_links']:
                        try:
                            base_filename = self.clean_filename(paper['title'])
                            
                            # Try to download (this will now handle PDF vs web content automatically)
                            if self.download_pdf(pdf_url, base_filename + '.pdf'):
                                # Check if we got a PDF or text file
                                pdf_file = os.path.join(self.papers_dir, base_filename + '.pdf')
                                txt_file = os.path.join(self.papers_dir, base_filename + '.txt')
                                
                                if os.path.exists(pdf_file):
                                    pdf_path = pdf_file
                                    pdf_downloaded = True
                                    downloaded_count += 1
                                    
                                    summary_data['downloaded_papers'].append({
                                        'title': paper['title'],
                                        'authors': paper.get('authors_and_publication', ''),
                                        'year': paper.get('year', ''),
                                        'source': paper.get('source', ''),
                                        'url': paper.get('url', ''),
                                        'pdf_path': pdf_path,
                                        'search_query': paper.get('search_query', '')
                                    })
                                    
                                elif os.path.exists(txt_file):
                                    content_extracted = True
                                    downloaded_count += 1
                                    
                                    summary_data['web_content_extracted'].append({
                                        'title': paper['title'],
                                        'authors': paper.get('authors_and_publication', ''),
                                        'year': paper.get('year', ''),
                                        'source': paper.get('source', ''),
                                        'url': paper.get('url', ''),
                                        'content_path': txt_file,
                                        'search_query': paper.get('search_query', '')
                                    })
                                    
                                    print(f"✅ Web content extracted instead of PDF")
                                
                                break
                        except Exception as e:
                            print(f"❌ Processing failed: {e}")
                            continue
                
                if not pdf_downloaded and not content_extracted:
                    print(f"❌ No content could be obtained")
                    summary_data['failed_downloads'].append({
                        'title': paper['title'],
                        'reason': 'No downloadable PDF and no extractable content',
                        'url': paper.get('url', '')
                    })
                    continue
            
            # Process with extractor if we have a PDF
            if self.extractor and pdf_path and pdf_downloaded:
                try:
                    print(f"🔬 Processing PDF with extractor...")
                    # CREATE ENHANCED METADATA
                    enhanced_metadata = {
                        'web_title': paper['title'],
                        'web_authors': paper.get('authors_and_publication', ''),
                        'web_year': paper.get('year', ''),
                        'web_url': paper.get('url', ''),
                        'web_journal': paper.get('journal', 'Unknown Journal'),
                        'web_source': paper.get('source', ''),
                        'web_citations': paper.get('citations', 0),
                        'search_query': paper.get('search_query', '')
                    }
                    
                    # Pass enhanced metadata to extractor
                    result_path = self.extractor.process_research_paper(
                        pdf_path, 
                        self.output_dir, 
                        web_metadata=enhanced_metadata
                    )
                    
                    if result_path:
                        processed_count += 1
                        summary_data['processed_papers'].append({
                            'title': paper['title'],
                            'pdf_path': pdf_path,
                            'extracted_path': result_path
                        })
                        print(f"✅ PDF extraction complete: {os.path.basename(result_path)}")
                    else:
                        summary_data['processing_errors'].append({
                            'title': paper['title'],
                            'pdf_path': pdf_path,
                            'error': 'PDF extraction failed'
                        })
                                
                except Exception as e:
                    print(f"❌ PDF processing error: {e}")
                    summary_data['processing_errors'].append({
                        'title': paper['title'],
                        'pdf_path': pdf_path,
                        'error': str(e)
                    })
            
            # Rate limiting
            time.sleep(0.5 if skip_download else 2)
        
        # Save summary
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        action_text = "PROCESSING" if skip_download else "DOWNLOAD AND PROCESSING"
        print(f"\n🎉 {action_text} COMPLETE!")
        print("=" * 50)
        print(f"Total processed: {downloaded_count} items")
        print(f"PDFs downloaded: {len(summary_data['downloaded_papers'])}")
        print(f"Web content extracted: {len(summary_data['web_content_extracted'])}")
        print(f"PDFs processed by extractor: {processed_count}")
        print(f"Failed: {len(summary_data['failed_downloads'])}")
        print(f"Summary saved: {summary_file}")
        
        return summary_data

    def extract_abstract_from_url(self, url: str) -> Optional[Dict]:
        """Extract abstract and metadata from a paper URL."""
        print(f"🔍 Attempting to extract abstract from: {url[:60]}...")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            abstract_data = {
                'url': url,
                'title': '',
                'authors': '',
                'year': '',
                'journal': '',
                'abstract': '',
                'doi': '',
                'keywords': []
            }
            
            # Extract title
            title_selectors = [
                'h1.title', 'h1#title', '.title', 'h1',
                '[data-test="paper-title"]', '.paper-title',
                'meta[name="citation_title"]'
            ]
            
            for selector in title_selectors:
                if selector.startswith('meta'):
                    meta_tag = soup.find('meta', attrs={'name': 'citation_title'})
                    if meta_tag:
                        abstract_data['title'] = meta_tag.get('content', '').strip()
                        break
                else:
                    title_elem = soup.select_one(selector)
                    if title_elem:
                        abstract_data['title'] = title_elem.get_text().strip()
                        break
            
            # Extract authors
            author_selectors = [
                '[data-test="author-list"]', '.authors', '.author-list',
                'meta[name="citation_author"]', '.author', '[class*="author"]'
            ]
            
            authors_list = []
            for selector in author_selectors:
                if selector.startswith('meta'):
                    meta_tags = soup.find_all('meta', attrs={'name': 'citation_author'})
                    authors_list = [tag.get('content', '') for tag in meta_tags]
                    if authors_list:
                        break
                else:
                    author_elem = soup.select_one(selector)
                    if author_elem:
                        authors_list = [author_elem.get_text().strip()]
                        break
            
            abstract_data['authors'] = ', '.join(authors_list) if authors_list else 'Unknown Authors'
            
            # Extract year
            year_patterns = [
                r'(\d{4})',  # Any 4-digit number
                r'Published[:\s]*(\d{4})',
                r'Copyright[:\s]*(\d{4})'
            ]
            
            year_selectors = [
                'meta[name="citation_publication_date"]',
                'meta[name="citation_date"]',
                '.publication-date', '.date', '[class*="year"]'
            ]
            
            for selector in year_selectors:
                if selector.startswith('meta'):
                    meta_tag = soup.find('meta', attrs={'name': selector.split('[')[1].split('=')[1].replace('"', '').replace(']', '')})
                    if meta_tag:
                        date_content = meta_tag.get('content', '')
                        year_match = re.search(r'(\d{4})', date_content)
                        if year_match:
                            abstract_data['year'] = year_match.group(1)
                            break
                else:
                    date_elem = soup.select_one(selector)
                    if date_elem:
                        date_text = date_elem.get_text()
                        year_match = re.search(r'(\d{4})', date_text)
                        if year_match:
                            abstract_data['year'] = year_match.group(1)
                            break
            
            # Extract journal
            journal_selectors = [
                'meta[name="citation_journal_title"]',
                '.journal-title', '.publication', '[class*="journal"]'
            ]
            
            for selector in journal_selectors:
                if selector.startswith('meta'):
                    meta_tag = soup.find('meta', attrs={'name': 'citation_journal_title'})
                    if meta_tag:
                        abstract_data['journal'] = meta_tag.get('content', '').strip()
                        break
                else:
                    journal_elem = soup.select_one(selector)
                    if journal_elem:
                        abstract_data['journal'] = journal_elem.get_text().strip()
                        break
            
            # Extract abstract
            abstract_selectors = [
                '#abstract', '.abstract', '[data-test="abstract"]',
                '.abstract-content', '.summary', '[class*="abstract"]',
                'div:contains("Abstract")', 'section:contains("Abstract")'
            ]
            
            for selector in abstract_selectors:
                if ':contains(' in selector:
                    # Handle contains selectors differently
                    abstract_elem = soup.find(lambda tag: tag.name in ['div', 'section'] and 
                                            'abstract' in tag.get_text().lower())
                else:
                    abstract_elem = soup.select_one(selector)
                
                if abstract_elem:
                    abstract_text = abstract_elem.get_text().strip()
                    # Clean up abstract text
                    abstract_text = re.sub(r'\s+', ' ', abstract_text)
                    abstract_text = abstract_text.replace('Abstract', '').strip()
                    if len(abstract_text) > 50:  # Reasonable abstract length
                        abstract_data['abstract'] = abstract_text
                        break
            
            # Extract DOI
            doi_selectors = [
                'meta[name="citation_doi"]',
                '[class*="doi"]', 'a[href*="doi.org"]'
            ]
            
            for selector in doi_selectors:
                if selector.startswith('meta'):
                    meta_tag = soup.find('meta', attrs={'name': 'citation_doi'})
                    if meta_tag:
                        abstract_data['doi'] = meta_tag.get('content', '').strip()
                        break
                else:
                    doi_elem = soup.select_one(selector)
                    if doi_elem:
                        doi_text = doi_elem.get('href', '') or doi_elem.get_text()
                        doi_match = re.search(r'10\.\d+/[^\s]+', doi_text)
                        if doi_match:
                            abstract_data['doi'] = doi_match.group(0)
                            break
            
            # Extract keywords
            keyword_selectors = [
                'meta[name="citation_keywords"]',
                '.keywords', '[class*="keyword"]'
            ]
            
            for selector in keyword_selectors:
                if selector.startswith('meta'):
                    meta_tag = soup.find('meta', attrs={'name': 'citation_keywords'})
                    if meta_tag:
                        keywords = meta_tag.get('content', '').split(',')
                        abstract_data['keywords'] = [k.strip() for k in keywords]
                        break
                else:
                    keyword_elem = soup.select_one(selector)
                    if keyword_elem:
                        keywords = keyword_elem.get_text().split(',')
                        abstract_data['keywords'] = [k.strip() for k in keywords]
                        break
            
            print(f"✅ Abstract extracted successfully!")
            return abstract_data
            
        except Exception as e:
            print(f"❌ Failed to extract abstract: {e}")
            return None

    def analyze_abstract_for_job_performance(self, abstract_data: Dict) -> Dict:
        """Analyze abstract and metadata to extract job performance correlation data."""
        print(f"🔬 Analyzing abstract for job performance correlations...")
        
        full_text = f"{abstract_data.get('title', '')} {abstract_data.get('abstract', '')}"
        
        analysis_result = {
            'predictor_category': 'Unknown',
            'pearson_r': 'Not found',
            'p_value': 'Not found',
            'r_squared': 'Not found',
            'beta_weight': 'Not found',
            'odds_ratio': 'Not found',
            'job_domain': 'Not found',
            'sample_context': 'Not found',
            'study_type': 'Not found',
            'measurement_type': 'Not found',
            'sample_size_n': 'Not found',
            'year': abstract_data.get('year', 'Unknown'),
            'source_apa': f"{abstract_data.get('authors', 'Unknown')} ({abstract_data.get('year', 'Unknown')}). {abstract_data.get('title', 'Unknown')}. {abstract_data.get('journal', 'Unknown Journal')}.",
            'source_link': abstract_data.get('url', ''),
            'notes': []
        }
        
        # Extract predictor categories
        predictor_patterns = {
            'cognitive_ability': r'cognitive ability|intelligence|IQ|mental ability|reasoning|verbal ability|numerical ability',
            'personality': r'personality|big five|conscientiousness|extraversion|neuroticism|openness|agreeableness',
            'experience': r'experience|tenure|years of service|work experience',
            'education': r'education|degree|qualification|academic',
            'skills': r'skills|competenc|technical skills|soft skills',
            'assessment_center': r'assessment center|assessment centre|situational judgment',
            'interview': r'interview|structured interview|behavioral interview',
            'biodata': r'biodata|biographical data|background information'
        }
        
        for category, pattern in predictor_patterns.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                analysis_result['predictor_category'] = category.replace('_', ' ').title()
                break
        
        # Extract statistical values
        statistical_patterns = {
            'pearson_r': [
                r'r\s*=\s*([-]?\d+\.?\d*)',
                r'correlation[s]?\s*[=:]\s*([-]?\d+\.?\d*)',
                r'pearson[\'s]?\s*r\s*=\s*([-]?\d+\.?\d*)'
            ],
            'p_value': [
                r'p\s*[<>=]\s*(\d+\.?\d*)',
                r'significance\s*[=:]\s*(\d+\.?\d*)',
                r'p[-\s]*value\s*[=:]\s*(\d+\.?\d*)'
            ],
            'r_squared': [
                r'r[²2]\s*=\s*(\d+\.?\d*)',
                r'R[²2]\s*=\s*(\d+\.?\d*)',
                r'variance explained\s*[=:]\s*(\d+\.?\d*)%?'
            ],
            'beta_weight': [
                r'β\s*=\s*([-]?\d+\.?\d*)',
                r'beta\s*[=:]\s*([-]?\d+\.?\d*)',
                r'standardized coefficient\s*[=:]\s*([-]?\d+\.?\d*)'
            ],
            'sample_size_n': [
                r'N\s*=\s*(\d+)',
                r'n\s*=\s*(\d+)',
                r'sample size\s*[=:]\s*(\d+)',
                r'participants\s*[=:]?\s*(\d+)'
            ]
        }
        
        for key, patterns in statistical_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                if matches:
                    analysis_result[key] = matches[0]
                    break
        
        # Extract job domains
        job_domain_patterns = {
            'management': r'manag(er|ement)|supervisor|leadership|executive',
            'sales': r'sales|selling|sales performance',
            'customer_service': r'customer service|service quality|client relations',
            'technical': r'technical|engineering|IT|software|programming',
            'healthcare': r'healthcare|medical|nursing|clinical',
            'education': r'teaching|education|academic|school',
            'manufacturing': r'manufacturing|production|factory|assembly'
        }
        
        for domain, pattern in job_domain_patterns.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                analysis_result['job_domain'] = domain.replace('_', ' ').title()
                break
        
        # Extract study type
        study_type_patterns = {
            'meta-analysis': r'meta[-\s]?analysis|systematic review',
            'longitudinal': r'longitudinal|panel study|over time',
            'cross-sectional': r'cross[-\s]?sectional|survey',
            'experimental': r'experiment|randomized|controlled trial',
            'field_study': r'field study|organizational study'
        }
        
        for study_type, pattern in study_type_patterns.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                analysis_result['study_type'] = study_type.replace('_', ' ').title()
                break
        
        # Extract measurement type
        measurement_patterns = {
            'objective': r'objective measure|performance rating|supervisor rating',
            'subjective': r'self[-\s]?report|self[-\s]?assessment|survey',
            'behavioral': r'behavioral|observed|direct observation',
            'archival': r'archival|records|database|historical'
        }
        
        for measurement, pattern in measurement_patterns.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                analysis_result['measurement_type'] = measurement.title()
                break
        
        # Add notes about what was found
        if analysis_result['pearson_r'] != 'Not found':
            analysis_result['notes'].append(f"Correlation coefficient found: {analysis_result['pearson_r']}")
        
        if analysis_result['sample_size_n'] != 'Not found':
            analysis_result['notes'].append(f"Sample size identified: {analysis_result['sample_size_n']}")
        
        if len(abstract_data.get('abstract', '')) > 100:
            analysis_result['notes'].append("Full abstract available for analysis")
        else:
            analysis_result['notes'].append("Limited abstract content available")
        
        print(f"✅ Analysis complete. Found {len([v for v in analysis_result.values() if v != 'Not found' and v != 'Unknown'])} data points")
        
        return analysis_result

    def save_abstract_analysis(self, paper_title: str, abstract_data: Dict, analysis_result: Dict):
        """Save abstract content and analysis results."""
        
        # Clean filename
        clean_title = re.sub(r'[<>:"/\\|?*]', '', paper_title)[:100]
        
        # Save abstract as text file in papers folder
        abstract_filename = f"{clean_title}_abstract.txt"
        abstract_path = os.path.join(self.papers_dir, abstract_filename)
        
        try:
            with open(abstract_path, 'w', encoding='utf-8') as f:
                f.write(f"TITLE: {abstract_data.get('title', 'Unknown')}\n")
                f.write(f"AUTHORS: {abstract_data.get('authors', 'Unknown')}\n")
                f.write(f"YEAR: {abstract_data.get('year', 'Unknown')}\n")
                f.write(f"JOURNAL: {abstract_data.get('journal', 'Unknown')}\n")
                f.write(f"DOI: {abstract_data.get('doi', 'Unknown')}\n")
                f.write(f"URL: {abstract_data.get('url', '')}\n")
                f.write(f"KEYWORDS: {', '.join(abstract_data.get('keywords', []))}\n")
                f.write(f"\n{'='*50}\n")
                f.write(f"ABSTRACT:\n{abstract_data.get('abstract', 'No abstract available')}\n")
            
            print(f"💾 Abstract saved: {abstract_filename}")
        except Exception as e:
            print(f"❌ Failed to save abstract: {e}")
        
        # Save analysis results in output folder
        analysis_filename = f"{clean_title}_analysis.txt"
        analysis_path = os.path.join(self.output_dir, analysis_filename)
        
        try:
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write("JOB PERFORMANCE PREDICTOR ANALYSIS\n")
                f.write("="*50 + "\n\n")
                f.write(f"Paper Title: {abstract_data.get('title', 'Unknown')}\n")
                f.write(f"Authors: {abstract_data.get('authors', 'Unknown')}\n")
                f.write(f"Source URL: {abstract_data.get('url', '')}\n\n")
                
                f.write("EXTRACTED DATA:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Predictor Category: {analysis_result['predictor_category']}\n")
                f.write(f"Pearson r: {analysis_result['pearson_r']}\n")
                f.write(f"p-value: {analysis_result['p_value']}\n")
                f.write(f"R-squared: {analysis_result['r_squared']}\n")
                f.write(f"Beta Weight: {analysis_result['beta_weight']}\n")
                f.write(f"Odds Ratio: {analysis_result['odds_ratio']}\n")
                f.write(f"Job Domain: {analysis_result['job_domain']}\n")
                f.write(f"Sample Context: {analysis_result['sample_context']}\n")
                f.write(f"Study Type: {analysis_result['study_type']}\n")
                f.write(f"Measurement Type: {analysis_result['measurement_type']}\n")
                f.write(f"Sample Size (N): {analysis_result['sample_size_n']}\n")
                f.write(f"Year: {analysis_result['year']}\n")
                f.write(f"Source (APA): {analysis_result['source_apa']}\n")
                f.write(f"Source Link: {analysis_result['source_link']}\n")
                
                if analysis_result['notes']:
                    f.write(f"\nNotes:\n")
                    for note in analysis_result['notes']:
                        f.write(f"- {note}\n")
            
            print(f"💾 Analysis saved: {analysis_filename}")
        except Exception as e:
            print(f"❌ Failed to save analysis: {e}")
        
        return abstract_path, analysis_path

        
    def process_paper_links(self, paper_links: List[str]) -> List[Dict]:
        """Process a list of paper links and create paper objects for download."""
        print("🔗 PROCESSING PROVIDED PAPER LINKS")
        print("=" * 80)
        
        papers_from_links = []
        
        for i, link in enumerate(paper_links, 1):
            print(f"\n[{i}/{len(paper_links)}] Processing link: {link[:80]}...")
            
            # Determine if it's a direct PDF or a paper page
            is_direct_pdf = link.lower().endswith('.pdf')
            
            # Extract filename from URL or create generic one
            try:
                if is_direct_pdf:
                    # Direct PDF link
                    parsed_url = link.split('/')[-1]
                    title = parsed_url.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
                    pdf_links = [link]
                    print(f"🎯 Direct PDF detected")
                else:
                    # Paper page URL - need to find PDF link
                    title = f"Paper_{i}_from_page"
                    pdf_links = []
                    
                    # Try to convert common paper page URLs to PDF URLs
                    if 'arxiv.org/abs/' in link:
                        # Convert ArXiv abstract to PDF
                        arxiv_id = link.split('/')[-1]
                        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                        pdf_links = [pdf_url]
                        title = f"ArXiv_{arxiv_id}"
                        print(f"🔄 ArXiv abstract converted to PDF link")
                    elif 'doi.org/' in link:
                        # DOI link - will need to be handled by your existing scraper
                        pdf_links = [link]  # Let your scraper try to find the PDF
                        title = f"DOI_Paper_{i}"
                        print(f"🔗 DOI link - will attempt to find PDF")
                    else:
                        # Generic paper page - let your scraper handle it
                        pdf_links = [link]
                        print(f"🌐 Paper page - will attempt to find PDF")
                        
            except Exception as e:
                print(f"⚠️  Error processing link: {e}")
                title = f"Paper_{i}_from_link"
                pdf_links = [link]
            
            # Create paper object
            paper_info = {
                'title': title,
                'authors_and_publication': 'Direct Link - Unknown Authors',
                'year': 'Unknown',
                'source': 'Direct Link',
                'url': link,
                'journal': 'Unknown Journal',
                'citations': 0,
                'pdf_links': pdf_links,
                'search_query': 'direct_links',
                'is_direct_pdf': is_direct_pdf
            }
            
            papers_from_links.append(paper_info)
            print(f"✅ Added: {title}")
        
        print(f"\n📊 LINK PROCESSING SUMMARY")
        print("=" * 40)
        print(f"Total links processed: {len(papers_from_links)}")
        direct_pdfs = sum(1 for p in papers_from_links if p.get('is_direct_pdf', False))
        print(f"Direct PDF links: {direct_pdfs}")
        print(f"Paper page links: {len(papers_from_links) - direct_pdfs}")
        
        return papers_from_links

def main():
    """Main function to run the complete pipeline."""
    print("🔬 RESEARCH PAPER SCRAPER AND PROCESSOR")
    print("=" * 80)
    
    # ========================================
    # CONFIGURATION - CHANGE THESE AS NEEDED
    # ========================================
    
    # MODE SELECTION (choose one):
    # 1. FIND_NEW_PAPERS = True: Search online for new papers
    # 2. FIND_NEW_PAPERS = False and PAPER_LINKS = []: Process existing papers from folder
    # 3. FIND_NEW_PAPERS = False and PAPER_LINKS = [...]: Download from provided links
    
    FIND_NEW_PAPERS = False  # Set to True to search online
    
    # If FIND_NEW_PAPERS is False, you can provide direct links here:
    PAPER_LINKS = []
    
    # Configuration
    papers_dir = "papers"
    output_dir = "outputs"
    max_papers_total = 50  # Adjust based on your needs
    max_downloads = 30     # Adjust based on your needs
    
    # Initialize scraper
    scraper = ResearchPaperScraper(papers_dir, output_dir)
    
    try:
        if FIND_NEW_PAPERS:
            # Mode 1: Search online for new papers
            print("\n🔍 MODE: FINDING NEW RESEARCH PAPERS ONLINE")
            print("🔍 STEP 1: SEARCHING FOR RESEARCH PAPERS")
            found_papers = scraper.run_comprehensive_search(
                max_papers_per_query=3,
                max_total_papers=max_papers_total
            )
            
            if not found_papers:
                print("❌ No papers found!")
                return
            
            # Step 2: Download and process
            print("\n📥 STEP 2: DOWNLOADING AND PROCESSING PAPERS")
            summary = scraper.download_and_process_papers(
                found_papers,
                max_downloads=max_downloads,
                skip_download=False
            )
        
        elif PAPER_LINKS:
            # Mode 2: Download from provided links
            print("\n🔗 MODE: DOWNLOADING FROM PROVIDED LINKS")
            print("🔗 STEP 1: PROCESSING PROVIDED LINKS")
            papers_from_links = scraper.process_paper_links(PAPER_LINKS)
            
            if not papers_from_links:
                print("❌ No valid links processed!")
                return
            
            # Step 2: Download and process from links
            print("\n📥 STEP 2: DOWNLOADING AND PROCESSING FROM LINKS")
            summary = scraper.download_and_process_papers(
                papers_from_links,
                max_downloads=max_downloads,
                skip_download=False
            )
        
        else:  
            # Mode 3: Process existing papers from folder
            print("\n📁 MODE: PROCESSING EXISTING PAPERS FROM LOCAL FOLDER")
            print("📁 STEP 1: LOADING EXISTING PAPERS")
            existing_papers = scraper.load_existing_papers()
            
            if not existing_papers:
                print("❌ No existing papers found in the 'papers' folder!")
                return
            
            # Step 2: Process existing papers (skip download)
            print("\n🔬 STEP 2: PROCESSING EXISTING PAPERS")
            summary = scraper.download_and_process_papers(
                existing_papers,
                max_downloads=max_downloads,
                skip_download=True
            )
        
        print(f"\n🎉 PIPELINE COMPLETE!")
        print(f"Check '{output_dir}' for extracted data files")
        if FIND_NEW_PAPERS or PAPER_LINKS:
            print(f"Check '{papers_dir}' for downloaded PDFs")
        
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()