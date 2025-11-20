"""
Advanced Resume Parser with Free AI Semantic Search

IMPROVEMENTS IN THIS VERSION:
1. FIXED: Empty search now returns all resumes (for dashboard view)
2. ADDED: Free AI semantic search using Hugging Face or Ollama
3. ADDED: Embedding cache for fast repeated searches
4. ADDED: use_ai parameter in search API

QUICK START:
1. Basic usage (keyword search only):
   python parse.py --web

2. With AI search enabled:
   - Set AI_SEARCH_ENABLED = True below
   - Get free HF token: https://huggingface.co/settings/tokens
   - Set environment variable: export HF_TOKEN=your_token
   - Run: python parse.py --web

3. For unlimited AI search:
   - Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
   - Download model: ollama pull nomic-embed-text
   - Start server: ollama serve
   - Set AI_PROVIDER = 'ollama' below
   - Run: python parse.py --web

API USAGE:
- Keyword search: POST /api/search with {"query": "python developer"}
- AI search: POST /api/search with {"query": "experienced cloud engineer", "use_ai": true}

"""

import os
import re
import json
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

mods: Dict[str, Any] = {}

def _try_imports():
    try:
        import fitz
        mods['fitz'] = fitz
    except Exception:
        pass

    try:
        import pdfplumber
        mods['pdfplumber'] = pdfplumber
    except Exception:
        pass

    try:
        from docx import Document as DocxDocument
        mods['docx'] = DocxDocument
    except Exception:
        pass

    try:
        from PIL import Image
        mods['PIL'] = Image
    except Exception:
        pass

    try:
        import pytesseract
        mods['tesseract'] = pytesseract
    except Exception:
        pass

    try:
        import numpy as np
        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
        if numpy_version[0] < 2:
            import easyocr
            mods['easyocr'] = easyocr
    except Exception:
        pass

    try:
        import numpy as np
        mods['numpy'] = np
    except Exception:
        pass

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        mods['spacy'] = nlp
    except Exception:
        mods['spacy'] = None

_try_imports()

# ============ FREE AI SEARCH CONFIGURATION ============
AI_SEARCH_ENABLED = False  # Set to True to enable AI semantic search

# Free AI Provider Options:
# 1. 'huggingface' - Free cloud API (requires token for better limits)
#    Get free token at: https://huggingface.co/settings/tokens
# 2. 'ollama' - Free local API (unlimited, requires ollama installed)
#    Install: curl -fsSL https://ollama.com/install.sh | sh
#    Then run: ollama pull nomic-embed-text
AI_PROVIDER = 'huggingface'  # or 'ollama'

# Hugging Face settings (if using 'huggingface' provider)
HF_API_KEY = os.environ.get('HF_TOKEN', '')  # Optional, improves rate limits
HF_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'  # Fast & free
HF_ENDPOINT = 'https://api-inference.huggingface.co/pipeline/feature-extraction'

# Ollama settings (if using 'ollama' provider)
OLLAMA_MODEL = 'nomic-embed-text'
OLLAMA_ENDPOINT = 'http://localhost:11434'

# Cache settings
ENABLE_EMBEDDING_CACHE = True
EMBEDDING_CACHE_FILE = 'embeddings_cache.json'
# ====================================================

WHITESPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.I)
PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{4})")
LINKEDIN_RE = re.compile(r"(?:linkedin\.com/in/|linkedin\.com/pub/)([a-zA-Z0-9\-]+)", re.I)
GITHUB_RE = re.compile(r"(?:github\.com/)([a-zA-Z0-9\-]+)", re.I)

SECTION_HEADERS = {
    'experience': ['experience', 'work history', 'employment', 'professional experience', 'work experience', 'career history'],
    'education': ['education', 'academic', 'qualification', 'degree', 'university', 'college'],
    'skills': ['skills', 'technical skills', 'competencies', 'expertise', 'technologies', 'proficiencies'],
    'projects': ['projects', 'personal projects', 'key projects'],
    'certifications': ['certifications', 'certificates', 'licenses'],
    'summary': ['summary', 'profile', 'objective', 'about', 'professional summary'],
}

COMPREHENSIVE_SKILLS = {
    'programming': {
        'python': ['python', 'python3', 'py', 'django', 'flask', 'fastapi', 'pandas', 'numpy'],
        'java': ['java', 'spring', 'spring boot', 'hibernate', 'maven', 'gradle'],
        'javascript': ['javascript', 'js', 'typescript', 'ts', 'node.js', 'nodejs', 'node'],
        'csharp': ['c#', 'csharp', '.net', 'dotnet', 'asp.net'],
        'cpp': ['c++', 'cpp'],
        'c': ['c programming', r'\bc\b'],
        'ruby': ['ruby', 'rails', 'ruby on rails'],
        'go': ['golang', r'\bgo\b', 'go lang'],
        'rust': ['rust'],
        'php': ['php', 'laravel', 'symfony'],
        'swift': ['swift', 'swiftui'],
        'kotlin': ['kotlin'],
        'scala': ['scala'],
        'r': [r'\br\b', 'r programming', 'rstudio'],
        'matlab': ['matlab'],
        'perl': ['perl'],
        'shell': ['bash', 'shell', 'powershell', 'shell scripting'],
    },
    'web': {
        'frontend': ['html', 'html5', 'css', 'css3', 'sass', 'scss', 'less'],
        'react': ['react', 'reactjs', 'react.js', 'redux', 'react native'],
        'angular': ['angular', 'angularjs'],
        'vue': ['vue', 'vuejs', 'vue.js', 'vuex', 'nuxt'],
        'jquery': ['jquery'],
        'bootstrap': ['bootstrap'],
        'tailwind': ['tailwind', 'tailwindcss'],
        'webpack': ['webpack', 'vite', 'parcel'],
    },
    'backend': {
        'nodejs': ['node.js', 'nodejs', 'express', 'expressjs', 'nestjs'],
        'django': ['django', 'django rest'],
        'flask': ['flask'],
        'fastapi': ['fastapi'],
        'spring': ['spring', 'spring boot', 'spring mvc'],
        'rails': ['rails', 'ruby on rails'],
        'asp': ['asp.net', 'asp.net core'],
    },
    'database': {
        'sql': ['sql', 'mysql', 'postgresql', 'postgres', 'sqlite', 'mariadb', 'mssql', 'oracle', 't-sql', 'pl/sql'],
        'nosql': ['mongodb', 'cassandra', 'couchdb', 'dynamodb'],
        'redis': ['redis', 'memcached'],
        'elasticsearch': ['elasticsearch', 'elastic search', 'elk'],
    },
    'cloud': {
        'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda', 'cloudformation', 'cloudfront', 'rds', 'dynamodb'],
        'azure': ['azure', 'microsoft azure', 'azure devops'],
        'gcp': ['gcp', 'google cloud', 'google cloud platform'],
        'docker': ['docker', 'containerization'],
        'kubernetes': ['kubernetes', 'k8s', 'helm'],
        'terraform': ['terraform', 'infrastructure as code', 'iac'],
        'jenkins': ['jenkins', 'ci/cd', 'continuous integration'],
        'ansible': ['ansible'],
        'chef': ['chef'],
        'puppet': ['puppet'],
    },
    'data_science': {
        'machine_learning': ['machine learning', 'ml', 'deep learning', 'neural networks', 'cnn', 'rnn', 'lstm'],
        'tensorflow': ['tensorflow', 'tf', 'keras'],
        'pytorch': ['pytorch', 'torch'],
        'scikit': ['scikit-learn', 'sklearn', 'scikit'],
        'pandas': ['pandas', 'data analysis'],
        'numpy': ['numpy'],
        'spark': ['spark', 'pyspark', 'apache spark'],
        'hadoop': ['hadoop', 'mapreduce', 'hdfs'],
        'data_viz': ['matplotlib', 'seaborn', 'plotly', 'tableau', 'power bi', 'powerbi', 'd3.js'],
        'nlp': ['nlp', 'natural language processing', 'spacy', 'nltk', 'bert', 'gpt'],
        'computer_vision': ['opencv', 'computer vision', 'image processing'],
    },
    'mobile': {
        'ios': ['ios', 'swift', 'swiftui', 'objective-c', 'xcode'],
        'android': ['android', 'kotlin', 'java android', 'android studio'],
        'flutter': ['flutter', 'dart'],
        'react_native': ['react native', 'react-native'],
    },
    'devops': {
        'git': ['git', 'github', 'gitlab', 'bitbucket', 'version control'],
        'cicd': ['ci/cd', 'jenkins', 'github actions', 'gitlab ci', 'circleci', 'travis ci'],
        'monitoring': ['prometheus', 'grafana', 'datadog', 'new relic', 'splunk'],
        'linux': ['linux', 'unix', 'ubuntu', 'centos', 'rhel'],
    },
    'testing': {
        'unit': ['junit', 'pytest', 'jest', 'mocha', 'chai', 'unit testing'],
        'automation': ['selenium', 'cypress', 'playwright', 'test automation'],
        'api': ['postman', 'rest api', 'api testing'],
    },
    'other': {
        'agile': ['agile', 'scrum', 'kanban', 'jira', 'sprint'],
        'rest': ['rest', 'restful', 'rest api', 'graphql'],
        'microservices': ['microservices', 'micro services'],
        'api': ['api', 'rest api', 'graphql', 'soap'],
        'security': ['security', 'cybersecurity', 'penetration testing', 'owasp'],
    }
}

DEGREE_PATTERNS = [
    r'\b(?:bachelor|bachelors|b\.?s\.?|b\.?a\.?|b\.?tech|b\.?e\.?)\b',
    r'\b(?:master|masters|m\.?s\.?|m\.?a\.?|m\.?tech|m\.?e\.?|mba)\b',
    r'\b(?:phd|ph\.?d\.?|doctorate|doctoral)\b',
    r'\b(?:associate|diploma|certification)\b',
]

def norm_ws(s: str) -> str:
    if not s:
        return ""
    return WHITESPACE_RE.sub(" ", s).strip()

def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def only_digits(s: Optional[str]) -> str:
    return re.sub(r"\D", "", s or "")

def normalize_email(e: str) -> str:
    if not e: return ""
    e = e.strip().lower()
    # collapse gmail dots (common duplicate)
    local, _, dom = e.partition("@")
    if dom in {"gmail.com", "googlemail.com"}:
        local = local.replace(".", "")
    return f"{local}@{dom}"

def normalize_name(n: str) -> str:
    return re.sub(r"[^a-z ]", "", (n or "").lower()).strip()

def content_fingerprint(text: str, k: int = 7) -> str:
    """Lightweight shingle hash to match same CV content across formats."""
    t = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if len(t) < k: 
        return sha1_hex(t)
    shingles = [t[i:i+k] for i in range(0, len(t)-k+1, k//2 or 1)]
    return sha1_hex("|".join(shingles[:1200]))  # cap for speed

def similarity(a: str, b: str) -> float:
    if not a or not b: return 0.0
    return SequenceMatcher(None, a, b).ratio()

def pick_better(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Choose the stronger profile field-by-field.
    Heuristics: higher extraction_confidence, longer text, has email/phone/urls,
    more skills/positions/education entries.
    """
    def score(p: Dict[str, Any]) -> Tuple:
        return (
            round(p.get("extraction_confidence", 0.0), 3),
            1 if p.get("email") else 0,
            1 if p.get("phone") else 0,
            len(p.get("skills") or []),
            len(p.get("education") or []),
            len(p.get("positions") or []),
            len((p.get("full_text") or "")),
        )
    return a if score(a) >= score(b) else b

def extract_text_from_pdf(path: Path) -> Tuple[str, float]:
    text = ""
    
    if 'fitz' in mods:
        try:
            with mods['fitz'].open(str(path)) as doc:
                parts = [p.get_text("text") or "" for p in doc]
                text = "\n".join(parts)
                if len(text.strip()) > 50:
                    return text, 0.95
        except Exception:
            pass
    
    if not text and 'pdfplumber' in mods:
        try:
            with mods['pdfplumber'].open(str(path)) as pdf:
                parts = [p.extract_text() or "" for p in pdf.pages]
                text = "\n".join(parts)
                if len(text.strip()) > 50:
                    return text, 0.90
        except Exception:
            pass
    
    confidence = 0.8 if len(text) > 50 else 0.3
    return text, confidence

def extract_text_from_doc(path: Path) -> Tuple[str, float]:
    """Extract text from old .doc format (Word 97-2003)"""
    text = ""
    
    # Method 1: Try antiword if available
    try:
        import subprocess
        result = subprocess.run(['antiword', str(path)], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout:
            text = result.stdout
            if len(text.strip()) > 50:
                return text, 0.90
    except Exception:
        pass
    
    # Method 2: Try textract if available
    try:
        import textract
        text = textract.process(str(path)).decode('utf-8', errors='ignore')
        if len(text.strip()) > 50:
            return text, 0.85
    except Exception:
        pass
    
    # Method 3: Try python-docx (sometimes works with .doc)
    if 'docx' in mods:
        try:
            doc = mods['docx'](str(path))
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            if len(text.strip()) > 50:
                return text, 0.80
        except Exception:
            pass
    
    # Method 4: Try converting with LibreOffice if available
    try:
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run([
                'soffice', '--headless', '--convert-to', 'txt:Text',
                '--outdir', tmpdir, str(path)
            ], capture_output=True, timeout=15)
            txt_file = Path(tmpdir) / f"{path.stem}.txt"
            if txt_file.exists():
                text = txt_file.read_text(encoding='utf-8', errors='ignore')
                if len(text.strip()) > 50:
                    return text, 0.85
    except Exception:
        pass
    
    # Method 5: Raw binary extraction (last resort)
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        # Try to decode as latin-1 and extract printable text
        text = raw.decode('latin-1', errors='ignore')
        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable() or c in '\n\r\t ')
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        if len(text.strip()) > 100:
            return text, 0.60
    except Exception:
        pass
    
    return "", 0.0

def extract_text_from_docx(path: Path) -> Tuple[str, float]:
    if 'docx' in mods:
        try:
            doc = mods['docx'](str(path))
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            if len(text.strip()) > 50:
                return text, 0.95
        except Exception:
            pass
    
    try:
        import zipfile
        with zipfile.ZipFile(str(path), 'r') as zf:
            with zf.open('word/document.xml') as f:
                xml = f.read().decode('utf-8', errors='ignore')
        xml = re.sub(r'</w:p>', '\n', xml, flags=re.I)
        text = re.sub(r'<[^>]+>', ' ', xml)
        text = re.sub(r'\s+', ' ', text)
        if len(text.strip()) > 50:
            return text.strip(), 0.85
    except Exception:
        pass
    
    return "", 0.0

def extract_text_from_file(path: Path) -> Tuple[str, float]:
    ext = path.suffix.lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(path)
    elif ext == '.doc':
        return extract_text_from_doc(path)
    elif ext == '.docx':
        return extract_text_from_docx(path)
    elif ext == '.txt':
        try:
            return path.read_text(encoding='utf-8'), 0.95
        except Exception:
            return "", 0.0
    
    return "", 0.0

def extract_email(text: str) -> List[str]:
    emails = EMAIL_RE.findall(text)
    return list(dict.fromkeys(emails))[:3]

def extract_phone(text: str) -> List[str]:
    phones = PHONE_RE.findall(text)
    return list(dict.fromkeys(phones))[:3]

def extract_urls(text: str) -> Dict[str, str]:
    urls = {}
    
    linkedin = LINKEDIN_RE.search(text)
    if linkedin:
        urls['linkedin'] = f"linkedin.com/in/{linkedin.group(1)}"
    
    github = GITHUB_RE.search(text)
    if github:
        urls['github'] = f"github.com/{github.group(1)}"
    
    return urls

def extract_name(text: str, filename: str) -> str:
    # Clean the text first
    text = re.sub(r'[^\w\s\-\.\,]', ' ', text)  # Remove special chars except basic ones
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    nlp = mods.get('spacy')
    if nlp:
        try:
            # Look at first 5 lines for name
            first_lines = '\n'.join(text.split('\n')[:5])
            doc = nlp(first_lines)
            persons = []
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name_candidate = norm_ws(ent.text)
                    # Better filtering for names
                    words = name_candidate.split()
                    if (2 <= len(words) <= 4 and 
                        5 <= len(name_candidate) <= 50 and
                        not any(word.lower() in ['resume', 'cv', 'curriculum', 'vitae'] for word in words) and
                        not re.search(r'\d{4}', name_candidate) and  # No years
                        not re.search(r'@|\.|com|org|net', name_candidate.lower()) and  # No email parts
                        all(len(word) >= 2 for word in words)):  # All words at least 2 chars
                        persons.append(name_candidate)
            
            if persons:
                return persons[0]
        except Exception:
            pass
    
    # Fallback: look for name-like patterns in first few lines
    for line in text.split('\n')[:5]:
        line = norm_ws(line)
        words = line.split()
        
        if (2 <= len(words) <= 3 and 
            5 <= len(line) <= 40 and
            not any(kw in line.lower() for kw in ['resume', 'curriculum', 'cv', 'vitae', 'phone', 'email', '@', 'address']) and
            not re.search(r'\d', line) and  # No numbers
            not re.search(r'[^a-zA-Z\s\-\.]', line) and  # Only letters, spaces, hyphens, dots
            all(word[0].isupper() for word in words if len(word) > 1)):  # Capitalized words
            return line
    
    # Last resort: clean filename
    clean_name = Path(filename).stem.replace('_', ' ').replace('-', ' ')
    clean_name = re.sub(r'\d+', '', clean_name)  # Remove numbers
    clean_name = ' '.join(word.title() for word in clean_name.split() if len(word) > 1)
    return clean_name if clean_name else "Unnamed Candidate"

def detect_sections(text: str) -> Dict[str, str]:
    lines = text.split('\n')
    sections = {}
    current_section = 'header'
    section_content = []
    
    for line in lines:
        line_lower = line.lower().strip()
        matched_section = None
        
        for section_type, keywords in SECTION_HEADERS.items():
            for keyword in keywords:
                if line_lower == keyword or (len(line_lower) < 30 and keyword in line_lower):
                    matched_section = section_type
                    break
            if matched_section:
                break
        
        if matched_section:
            if current_section and section_content:
                sections[current_section] = '\n'.join(section_content)
            current_section = matched_section
            section_content = []
        else:
            section_content.append(line)
    
    if current_section and section_content:
        sections[current_section] = '\n'.join(section_content)
    
    return sections

def extract_skills_advanced(text: str, sections: Dict[str, str]) -> Dict[str, Any]:
    text_lower = text.lower()
    skills_section = sections.get('skills', '').lower()
    
    found_skills = defaultdict(list)
    all_skills = []
    
    for category, subcategories in COMPREHENSIVE_SKILLS.items():
        for skill_name, patterns in subcategories.items():
            for pattern in patterns:
                regex = re.compile(r'\b' + re.escape(pattern) + r'\b', re.I) if '\\b' not in pattern else re.compile(pattern, re.I)
                
                if regex.search(skills_section):
                    found_skills[category].append(skill_name)
                    all_skills.append(skill_name)
                    break
                elif regex.search(text_lower):
                    found_skills[category].append(skill_name)
                    all_skills.append(skill_name)
                    break
    
    all_skills = list(dict.fromkeys(all_skills))
    
    return {
        'all_skills': all_skills,
        'by_category': dict(found_skills),
        'total_count': len(all_skills)
    }

def extract_education(text: str, sections: Dict[str, str]) -> List[Dict[str, Any]]:
    education_text = sections.get('education', '')
    if not education_text:
        education_text = text
    
    education = []
    lines = education_text.split('\n')
    
    current_edu = {}
    for i, line in enumerate(lines):
        line = norm_ws(line)
        if not line or len(line) < 5:
            continue
        
        for pattern in DEGREE_PATTERNS:
            if re.search(pattern, line, re.I):
                if current_edu:
                    education.append(current_edu)
                current_edu = {'degree': line, 'institution': '', 'year': '', 'details': ''}
                
                next_lines = lines[i+1:i+3]
                for next_line in next_lines:
                    next_line = norm_ws(next_line)
                    if next_line and len(next_line) > 5:
                        year_match = re.search(r'(19|20)\d{2}', next_line)
                        if year_match and not current_edu['year']:
                            current_edu['year'] = year_match.group(0)
                        if not current_edu['institution'] and not year_match:
                            current_edu['institution'] = next_line
                break
    
    if current_edu:
        education.append(current_edu)
    
    return education

def extract_experience_advanced(text: str, sections: Dict[str, str]) -> Dict[str, Any]:
    experience_text = sections.get('experience', '')
    if not experience_text:
        experience_text = text
    
    years = re.findall(r'\b(19|20)\d{2}\b', experience_text)
    year_numbers = [int(y) for y in years]
    
    positions = []
    companies = []
    locations = []
    
    title_pattern = re.compile(
        r'\b(?:(?:senior|sr\.?|junior|jr\.?|lead|principal|staff|chief|head|associate|assistant)\s+)?'
        r'(?:software|data|machine learning|ml|ai|product|project|program|security|platform|cloud|'
        r'devops|site\s+reliability|sre|marketing|sales|business|qa|quality|frontend|backend|'
        r'full\s*stack|mobile|ios|android|web|research|systems?|network|database|infrastructure)\s+'
        r'(?:engineer|developer|scientist|analyst|manager|director|architect|consultant|specialist|'
        r'designer|administrator|lead|coordinator|intern|trainee)',
        re.I
    )
    
    for match in title_pattern.finditer(experience_text):
        position = norm_ws(match.group(0))
        if position and position not in positions:
            positions.append(position)
    
    nlp = mods.get('spacy')
    if nlp:
        try:
            doc = nlp(experience_text[:5000])
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    company = norm_ws(ent.text)
                    if 3 <= len(company) <= 50 and company not in companies:
                        companies.append(company)
                elif ent.label_ == "GPE":
                    location = norm_ws(ent.text)
                    if location and location not in locations:
                        locations.append(location)
        except Exception:
            pass
    
    location_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b')
    for match in location_pattern.finditer(experience_text):
        location = norm_ws(match.group(0))
        if location not in locations:
            locations.append(location)
    
    total_years = 0
    if year_numbers:
        min_year = min(year_numbers)
        max_year = max(year_numbers)
        total_years = max(0, min(50, datetime.now().year - min_year))
    
    date_ranges = re.findall(r'(\d{4})\s*[-–—]\s*(?:(\d{4})|present|current)', experience_text, re.I)
    calculated_years = 0
    for start, end in date_ranges:
        end_year = int(end) if end else datetime.now().year
        calculated_years += max(0, end_year - int(start))
    
    if calculated_years > 0:
        total_years = min(calculated_years, total_years) if total_years > 0 else calculated_years
    
    return {
        'total_years': total_years,
        'positions': positions[:10],
        'companies': companies[:10],
        'locations': locations[:10],
        'current_title': positions[0] if positions else None,
        'years_range': (min(year_numbers), max(year_numbers)) if year_numbers else (None, None)
    }

def extract_certifications(text: str, sections: Dict[str, str]) -> List[str]:
    cert_text = sections.get('certifications', '')
    if not cert_text:
        cert_text = text
    
    cert_keywords = ['certified', 'certification', 'certificate', 'professional']
    certifications = []
    
    for line in cert_text.split('\n'):
        line = norm_ws(line)
        if any(kw in line.lower() for kw in cert_keywords):
            if 5 < len(line) < 100:
                certifications.append(line)
    
    return certifications[:10]

class ResumeParser:
    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        
        if not path.exists():
            return self._error_profile(str(path), path.name, "File not found")
        
        text, confidence = extract_text_from_file(path)
        
        if not text or len(text.strip()) < 50:
            return self._error_profile(str(path), path.name, "Insufficient text")
        
        try:
            sections = detect_sections(text)
            
            emails = extract_email(text)
            phones = extract_phone(text)
            urls = extract_urls(text)
            name = extract_name(text, path.name)
            
            skills_data = extract_skills_advanced(text, sections)
            education = extract_education(text, sections)
            experience = extract_experience_advanced(text, sections)
            certifications = extract_certifications(text, sections)
            
            profile = {
                'file_path': str(path),
                'filename': path.name,
                'candidate_id': sha1_hex(path.name + text[:100])[:12],  # provisional; may be replaced by deduper
                'extraction_confidence': confidence,
                'extracted_at': datetime.now().isoformat(),

                'name': name,
                'email': emails[0] if emails else None,
                'emails': [normalize_email(e) for e in emails],
                'phone': phones[0] if phones else None,
                'phones': phones,
                'urls': urls,

                'skills': skills_data['all_skills'],
                'skills_by_category': skills_data['by_category'],
                'total_skills': skills_data['total_count'],

                'education': education,
                'certifications': certifications,

                'total_years_experience': experience['total_years'],
                'positions': experience['positions'],
                'companies': experience['companies'],
                'locations': experience['locations'],
                'current_title': experience['current_title'],
                'years_range': experience['years_range'],

                'sections': list(sections.keys()),
                'summary': sections.get('summary', text[:500])[:500],
                'full_text': text[:15000],

                # NEW: dedup signals
                'fp_content': content_fingerprint(text),
                'name_norm': normalize_name(name),
                'primary_email': normalize_email(emails[0]) if emails else "",
                'sources': [{
                    'path': str(path),
                    'ext': path.suffix.lower().lstrip('.'),
                    'confidence': confidence,
                    'size': path.stat().st_size if path.exists() else None,
                    'mtime': datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
                }],
            }

            
            return profile
            
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return self._error_profile(str(path), path.name, str(e))
    
    def _error_profile(self, file_path: str, filename: str, error: str) -> Dict[str, Any]:
        return {
            'file_path': file_path,
            'filename': filename,
            'candidate_id': sha1_hex(filename)[:12],
            'error': error,
            'extraction_confidence': 0.0,
            'extracted_at': datetime.now().isoformat(),
            'name': 'Parse Failed',
            'email': None,
            'emails': [],
            'phone': None,
            'phones': [],
            'urls': {},
            'skills': [],
            'skills_by_category': {},
            'total_skills': 0,
            'education': [],
            'certifications': [],
            'total_years_experience': 0,
            'positions': [],
            'companies': [],
            'locations': [],
            'current_title': None,
            'years_range': (None, None),
            'sections': [],
            'summary': f'Error: {error}',
            'full_text': '',
        }

# ============ AI SEMANTIC SEARCH ENGINE ============
class AISearchEngine:
    """Free AI-powered semantic search using embeddings"""
    
    def __init__(self):
        self.cache = {}
        self.load_cache()
        
    def load_cache(self):
        """Load embedding cache from disk"""
        if ENABLE_EMBEDDING_CACHE and os.path.exists(EMBEDDING_CACHE_FILE):
            try:
                with open(EMBEDDING_CACHE_FILE, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def save_cache(self):
        """Save embedding cache to disk"""
        if ENABLE_EMBEDDING_CACHE:
            try:
                with open(EMBEDDING_CACHE_FILE, 'w') as f:
                    json.dump(self.cache, f)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text with caching"""
        if not text or 'requests' not in mods:
            return None
        
        # Check cache
        cache_key = sha1_hex(text[:1000])
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get fresh embedding
        embedding = None
        if AI_PROVIDER == 'huggingface':
            embedding = self._get_hf_embedding(text)
        elif AI_PROVIDER == 'ollama':
            embedding = self._get_ollama_embedding(text)
        
        # Cache it
        if embedding:
            self.cache[cache_key] = embedding
            if len(self.cache) % 10 == 0:
                self.save_cache()
        
        return embedding
    
    def _get_hf_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from Hugging Face API"""
        try:
            requests = mods['requests']
            headers = {'Content-Type': 'application/json'}
            if HF_API_KEY:
                headers['Authorization'] = f'Bearer {HF_API_KEY}'
            
            response = requests.post(
                f'{HF_ENDPOINT}/{HF_MODEL}',
                headers=headers,
                json={'inputs': text[:512], 'options': {'wait_for_model': True}},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0] if isinstance(result[0], list) else result
            return None
        except Exception as e:
            logger.warning(f"HF API error: {e}")
            return None
    
    def _get_ollama_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from local Ollama"""
        try:
            requests = mods['requests']
            response = requests.post(
                f'{OLLAMA_ENDPOINT}/api/embeddings',
                json={'model': OLLAMA_MODEL, 'prompt': text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get('embedding')
            return None
        except Exception as e:
            logger.warning(f"Ollama error: {e}")
            return None
    
    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity"""
        if 'numpy' in mods:
            np = mods['numpy']
            v1, v2 = np.array(v1), np.array(v2)
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        else:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = sum(a * a for a in v1) ** 0.5
            norm2 = sum(b * b for b in v2) ** 0.5
            return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
    
    def search(self, query: str, resumes: List[Dict], top_n: int = 10) -> List[Dict]:
        """Semantic search using AI embeddings"""
        query_emb = self.get_embedding(query)
        if not query_emb:
            return []
        
        results = []
        for resume in resumes:
            # Create searchable text
            parts = [
                resume.get('name', ''),
                resume.get('current_title', ''),
                resume.get('summary', '')[:500],
                ' '.join(resume.get('skills', [])[:20]),
                ' '.join(resume.get('companies', [])[:5]),
            ]
            doc_text = ' | '.join(filter(None, parts))
            
            doc_emb = self.get_embedding(doc_text)
            if doc_emb:
                similarity = self.cosine_similarity(query_emb, doc_emb)
                results.append({'resume': resume, 'score': similarity * 100})  # Scale to 0-100
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]

# ===================================================

class ResumeSearchSystem:
    def __init__(self, storage_path: str = "resume_database.json"):
        self.storage_path = storage_path
        self.parser = ResumeParser()
        self.resumes = self.load_database()
        self._build_key_index()
        
        # Initialize AI search engine if enabled
        self.ai_engine = None
        if AI_SEARCH_ENABLED:
            try:
                self.ai_engine = AISearchEngine()
                logger.info("✓ AI semantic search enabled")
            except Exception as e:
                logger.warning(f"AI search unavailable: {e}")
    
    def _build_key_index(self):
        self.key_index = {}  # key -> canonical_candidate_id
        for r in self.resumes:
            for k in self._candidate_keys_from_profile(r):
                self.key_index.setdefault(k, r['candidate_id'])

    def _candidate_keys_from_profile(self, p: Dict[str, Any]) -> List[str]:
        keys = set()
        # strongest: stable email(s)
        for e in (p.get('emails') or []):
            if e:
                keys.add(f"email:{normalize_email(e)}")
        if p.get('primary_email'):
            keys.add(f"email:{normalize_email(p['primary_email'])}")

        # name + phone
        if p.get('name') and p.get('phone'):
            phone10 = only_digits(p['phone'])[:10]
            if phone10:
                keys.add(f"namephone:{normalize_name(p['name'])}|{phone10}")

        # content fingerprint
        if p.get('fp_content'):
            keys.add(f"fp:{p['fp_content']}")

        # name-only weak key (used only as tie-breaker)
        if p.get('name'):
            keys.add(f"name:{normalize_name(p['name'])}")

        return list(keys)

    def _find_existing_id(self, keys: List[str], incoming: Dict[str, Any]) -> Optional[str]:
        # direct hit
        for k in keys:
            if k in self.key_index:
                return self.key_index[k]
        # soft match: same name + high similarity of text or overlapping companies/skills
        name = normalize_name(incoming.get('name',''))
        if not name: return None
        best_id, best_sim = None, 0.0
        for r in self.resumes:
            if normalize_name(r.get('name','')) != name: 
                continue
            sim = max(
                similarity(r.get('full_text',''), incoming.get('full_text','')),
                similarity(" ".join(r.get('companies') or []), " ".join(incoming.get('companies') or [])),
                similarity(" ".join(r.get('skills') or []), " ".join(incoming.get('skills') or [])),
            )
            if sim >= 0.85 and sim > best_sim:
                best_id, best_sim = r['candidate_id'], sim
        return best_id

    def _merge_profiles(self, base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        # choose best "overall", then fill missing bits from the other
        winner = pick_better(base, incoming)
        loser  = incoming if winner is base else base

        merged = dict(winner)  # shallow copy is fine for our primitives
        # union fields
        merged['emails'] = sorted(set((winner.get('emails') or []) + (loser.get('emails') or [])))
        merged['skills'] = sorted(set((winner.get('skills') or []) + (loser.get('skills') or [])))
        merged['skills_by_category'] = {
            **(loser.get('skills_by_category') or {}),
            **(winner.get('skills_by_category') or {})
        }
        merged['education'] = (winner.get('education') or []) + [e for e in (loser.get('education') or []) if e not in (winner.get('education') or [])]
        merged['certifications'] = sorted(set((winner.get('certifications') or []) + (loser.get('certifications') or [])))
        merged['positions'] = list(dict.fromkeys((winner.get('positions') or []) + (loser.get('positions') or [])))[:15]
        merged['companies'] = list(dict.fromkeys((winner.get('companies') or []) + (loser.get('companies') or [])))[:15]
        merged['locations'] = list(dict.fromkeys((winner.get('locations') or []) + (loser.get('locations') or [])))[:15]
        merged['sources'] = (winner.get('sources') or []) + (loser.get('sources') or [])
        # prefer non-empty summary/full_text from the winner, but keep a longer text if winner is very short
        if len(merged.get('full_text','')) < 200 and len(loser.get('full_text','')) > len(merged.get('full_text','')):
            merged['full_text'] = loser.get('full_text','')[:15000]

        # stable keys
        merged['primary_email'] = merged.get('primary_email') or loser.get('primary_email') or (merged['emails'][0] if merged['emails'] else "")
        merged['fp_content'] = merged.get('fp_content') or loser.get('fp_content')
        merged['name_norm'] = normalize_name(merged.get('name') or loser.get('name') or "")

        return merged

    def load_database(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception:
                return []
        return []

    
    def save_database(self):
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.resumes, f, indent=2, ensure_ascii=False)
    
    def upload_resume(self, file_path: str) -> Optional[Dict[str, Any]]:
        incoming = self.parser.parse_resume(file_path)
        if not incoming or incoming.get('extraction_confidence', 0) <= 0:
            return None

        keys = self._candidate_keys_from_profile(incoming)
        existing_id = self._find_existing_id(keys, incoming)

        if existing_id:
            # merge into existing canonical record
            for i, r in enumerate(self.resumes):
                if r['candidate_id'] == existing_id:
                    merged = self._merge_profiles(r, incoming)
                    merged['candidate_id'] = existing_id  # keep canonical id
                    self.resumes[i] = merged
                    # refresh index
                    for k in keys + self._candidate_keys_from_profile(merged):
                        self.key_index[k] = existing_id
                    self.save_database()
                    return merged
        else:
            # new canonical record; generate stable id (prefer email-based)
            stable_key = None
            if incoming.get('primary_email'):
                stable_key = f"email:{incoming['primary_email']}"
            elif incoming.get('emails'):
                stable_key = f"email:{incoming['emails'][0]}"
            elif incoming.get('fp_content'):
                stable_key = f"fp:{incoming['fp_content']}"
            else:
                stable_key = f"name:{normalize_name(incoming.get('name',''))}"
            canonical_id = sha1_hex(stable_key)[:12]
            incoming['candidate_id'] = canonical_id

            self.resumes.append(incoming)
            for k in keys:
                self.key_index[k] = canonical_id
            self.save_database()
            return incoming
    
    def upload_directory(self, directory_path: str):
        directory = Path(directory_path)
        if not directory.exists():
            return
        
        files = (list(directory.glob('*.pdf')) + 
                list(directory.glob('*.docx')) + 
                list(directory.glob('*.doc')) +
                list(directory.glob('*.txt')))
        
        success = 0
        for file_path in files:
            if self.upload_resume(str(file_path)):
                success += 1
    
    def search(self, query: str, top_n: int = 10, use_ai: bool = False) -> List[Dict[str, Any]]:
        """
        Enhanced search with keyword and AI semantic search options.
        
        Args:
            query: Search query string
            top_n: Number of results to return
            use_ai: Use AI semantic search if available (default: False for speed)
        """
        query_lower = query.lower().strip()
        
        # FIXED: Return all resumes when no query provided (for dashboard view)
        if not query_lower:
            logger.info(f"No query provided, returning all {len(self.resumes)} resumes")
            return [{'resume': r, 'score': 0} for r in self.resumes]
        
        # Try AI search if enabled and requested
        if use_ai and self.ai_engine and AI_SEARCH_ENABLED:
            logger.info(f"Using AI semantic search for: '{query}'")
            try:
                return self.ai_engine.search(query, self.resumes, top_n)
            except Exception as e:
                logger.warning(f"AI search failed, falling back to keyword: {e}")
        
        # Keyword-based search (fast and reliable)
        query_terms = query_lower.split()
        results = []
        
        logger.info(f"Searching {len(self.resumes)} resumes for: '{query}' (terms: {query_terms})")
        
        for resume in self.resumes:
            score = 0
            full_text = (resume.get('full_text') or '').lower()
            skills = [s.lower() for s in (resume.get('skills') or [])]
            positions = [p.lower() for p in (resume.get('positions') or [])]
            companies = [c.lower() for c in (resume.get('companies') or [])]
            matched_items = {'skills': [], 'positions': [], 'companies': []}
            
            # Score each query term
            for term in query_terms:
                # Exact skill match
                if term in skills:
                    score += 20
                    matched_items['skills'].append(term)
                # Partial skill match
                elif any(term in skill for skill in skills):
                    score += 12
                    for skill in skills:
                        if term in skill:
                            matched_items['skills'].append(skill)
                            break
                
                # Position matches
                for position in positions:
                    if term in position:
                        score += 10
                        matched_items['positions'].append(position)
                        break
                
                # Company matches
                for company in companies:
                    if term in company:
                        score += 8
                        matched_items['companies'].append(company)
                        break
                
                # Current title bonus
                current_title = resume.get('current_title') or ''
                if current_title and term in current_title.lower():
                    score += 15
                
                # Full text frequency
                score += full_text.count(term) * 2
            
            # Multi-word query exact match bonus
            if len(query_terms) > 1 and query_lower in full_text:
                score += 30
            
            # Experience bonus
            years_exp = resume.get('total_years_experience', 0)
            if years_exp > 0:
                score += min(years_exp, 10)
            
            # Add to results if any match found
            if score > 0:
                results.append({
                    'resume': resume,
                    'score': score,
                    'matched_skills': list(set(matched_items['skills'])),
                    'matched_positions': list(set(matched_items['positions']))[:3],
                    'matched_companies': list(set(matched_items['companies']))[:3],
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"Found {len(results)} matching resumes")
        return results[:top_n]
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self.resumes:
            return {}
        
        total = len(self.resumes)
        with_email = sum(1 for r in self.resumes if r.get('email'))
        with_phone = sum(1 for r in self.resumes if r.get('phone'))
        
        all_skills = defaultdict(int)
        for resume in self.resumes:
            for skill in resume.get('skills', []):
                all_skills[skill.lower()] += 1
        
        top_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:15]
        
        years_list = [r.get('total_years_experience', 0) for r in self.resumes 
                     if r.get('total_years_experience', 0) > 0]
        avg_years = sum(years_list) / len(years_list) if years_list else 0
        
        all_companies = defaultdict(int)
        for resume in self.resumes:
            for company in resume.get('companies', []):
                all_companies[company] += 1
        
        top_companies = sorted(all_companies.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total': total,
            'with_email': with_email,
            'with_phone': with_phone,
            'top_skills': top_skills,
            'top_companies': top_companies,
            'avg_years': avg_years,
            'years_range': (min(years_list), max(years_list)) if years_list else (0, 0)
        }

def _parse_one_file_global(fp: str):
    """Global function for multiprocessing - must be at module level for pickling"""
    parser = ResumeParser()
    return parser.parse_resume(fp)

def create_web_ui():
    try:
        from flask import Flask, make_response, request, jsonify, send_file, send_from_directory
        from werkzeug.exceptions import RequestEntityTooLarge, BadRequest
        from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
        import multiprocessing

        from queue import Queue
        import threading, time
    except ImportError:
        return None

    app = Flask(__name__, static_folder='build/static', static_url_path='/static')
    system = ResumeSearchSystem()

    try:
        from flask_cors import CORS
        CORS(app, resources={r"/api/*": {"origins": "*"}})
    except Exception:
        pass  # CORS is optional when serving same-origin


    # ---------------- Upload policy & limits ----------------
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB request cap
    PER_FILE_MAX_BYTES = 25 * 1024 * 1024                 # soft cap per file
    ALLOWED_EXTS = {".pdf", ".docx", ".doc", ".txt"}

    # ---------------- Background worker state ---------------
    incoming_dir = Path("incoming_uploads"); incoming_dir.mkdir(exist_ok=True)
    job_q: Queue[Path] = Queue()
    progress = {
        "queued": 0,       # total enqueued (since server start)
        "processing": 0,   # currently being parsed
        "processed": 0,    # finished (success + skipped)
        "success": 0,
        "skipped": 0,
        "last_error": ""
    }
    progress_lock = threading.Lock()
    worker_started = {"flag": False}

    def _has_active_filters_request(req):
        """Detect whether request JSON has any active search/filter fields."""
        payload = (req.get_json(silent=True) or {})
        q   = (payload.get("query") or "").strip()
        loc = (payload.get("location") or "").strip()
        jd  = (payload.get("jobDescription") or "").strip()
        f   = payload.get("filters") or {}

        def _nonempty(x):
            return bool(x and (isinstance(x, (list, tuple, set, dict)) and len(x) or str(x).strip()))

        has_exp = f.get("experience") not in ([0, 15], None)
        return any([
            _nonempty(q), _nonempty(loc), _nonempty(jd),
            _nonempty(f.get("skills")), _nonempty(f.get("workAuth")),
            _nonempty(f.get("remote")), _nonempty(f.get("education")),
            _nonempty(f.get("experienceLevel")), has_exp
        ])


    def _has_active_filters_request(req):
        return _has_active_filters(req)


    def start_worker_once():
        if worker_started["flag"]:
            return
        worker_started["flag"] = True

        # Use a small process pool to avoid GIL and allow hard timeouts.
        max_workers = max(2, min(6, (multiprocessing.cpu_count() or 2)))  # Increased workers
        pool = ProcessPoolExecutor(max_workers=max_workers)

        def worker():
            batch_save_interval = 5  # Save every 5 successful parses
            success_count = 0
            
            while True:
                path: Path = job_q.get()  # blocks
                with progress_lock:
                    progress["processing"] += 1
                    progress["current"] = str(path.name)

                try:
                    fut = pool.submit(_parse_one_file_global, str(path))
                    # Reduced timeout for faster processing
                    result = fut.result(timeout=25)

                    if result and result.get('extraction_confidence', 0) > 0:
                        # Direct insertion with dedup - faster than calling upload_resume
                        keys = system._candidate_keys_from_profile(result)
                        existing_id = system._find_existing_id(keys, result)
                        
                        if existing_id:
                            # Merge into existing
                            for i, r in enumerate(system.resumes):
                                if r['candidate_id'] == existing_id:
                                    merged = system._merge_profiles(r, result)
                                    merged['candidate_id'] = existing_id
                                    system.resumes[i] = merged
                                    system._build_key_index()  # Rebuild index
                                    break
                        else:
                            # New candidate
                            stable_key = None
                            if result.get('primary_email'):
                                stable_key = f"email:{result['primary_email']}"
                            elif result.get('emails'):
                                stable_key = f"email:{result['emails'][0]}"
                            elif result.get('fp_content'):
                                stable_key = f"fp:{result['fp_content']}"
                            else:
                                stable_key = f"name:{normalize_name(result.get('name',''))}"
                            canonical_id = sha1_hex(stable_key)[:12]
                            result['candidate_id'] = canonical_id
                            system.resumes.append(result)
                            for k in keys:
                                system.key_index[k] = canonical_id
                        
                        success_count += 1
                        
                        # Batch save - only save every N successful parses
                        if success_count % batch_save_interval == 0:
                            system.save_database()
                            refresh_suggestion_cache()  # Refresh cache after batch
                            logger.info(f"Batch saved: {success_count} resumes processed")
                        
                        with progress_lock:
                            progress["success"] += 1
                        logger.debug(f"Successfully parsed: {path.name}")
                    else:
                        with progress_lock:
                            progress["skipped"] += 1
                        logger.warning(f"Failed to extract from: {path.name}")

                except FuturesTimeout:
                    with progress_lock:
                        progress["skipped"] += 1
                        progress["last_error"] = f"Timeout parsing: {path.name}"
                    logger.error(f"Timeout parsing: {path.name}")

                except Exception as e:
                    with progress_lock:
                        progress["skipped"] += 1
                        progress["last_error"] = f"{type(e).__name__}: {str(e)[:300]}"
                    logger.error(f"Error parsing {path.name}: {e}")

                finally:
                    with progress_lock:
                        progress["processed"] += 1
                        progress["processing"] = max(0, progress["processing"] - 1)
                        progress["current"] = ""
                    
                    # Final save if queue is empty
                    if job_q.empty() and success_count % batch_save_interval != 0:
                        system.save_database()
                        refresh_suggestion_cache()
                        logger.info(f"Final save: {len(system.resumes)} total resumes")
                    
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    job_q.task_done()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        logger.info(f"Worker thread started with {max_workers} process workers")

    # ---------------- JSON error handlers -------------------
    @app.errorhandler(RequestEntityTooLarge)
    def handle_413(e):
        return jsonify({
            "ok": False,
            "error": "Payload too large",
            "hint": "Upload fewer files per batch or increase MAX_CONTENT_LENGTH."
        }), 413

    @app.errorhandler(BadRequest)
    def handle_400(e):
        return jsonify({
            "ok": False,
            "error": "Bad request",
            "hint": "The upload request was malformed or empty. Try again with smaller batches."
        }), 400

    HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TalentHub Pro - Advanced Recruitment Platform</title>
    
    <!-- React & Dependencies -->
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    
    <!-- Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #22d3ee;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #1f2937;
            --light: #f9fafb;
            --border: #e5e7eb;
            --text: #374151;
            --text-light: #6b7280;
            --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --radius: 8px;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--light);
            color: var(--text);
            line-height: 1.6;
        }

        .header {
            background: white;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: var(--shadow);
        }

        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
        }

        .logo i { font-size: 2rem; }

        .nav-items {
            display: flex;
            gap: 2rem;
            align-items: center;
        }

        .nav-item {
            color: var(--text);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s;
            cursor: pointer;
        }

        .nav-item:hover { color: var(--primary); }

        .btn {
            padding: 0.5rem 1.25rem;
            border-radius: var(--radius);
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            border: none;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            text-decoration: none;
            font-size: 0.95rem;
        }

        .btn-primary {
            background: var(--primary);
            color: white;
        }

        .btn-primary:hover {
            background: var(--primary-dark);
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }

        .btn-secondary {
            background: white;
            color: var(--primary);
            border: 1px solid var(--primary);
        }

        .btn-secondary:hover {
            background: var(--primary);
            color: white;
        }

        .btn-outline {
            background: transparent;
            color: var(--text);
            border: 1px solid var(--border);
        }

        .btn-outline:hover { background: var(--light); }
        .btn-sm { padding: 0.35rem 0.75rem; font-size: 0.85rem; }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        .search-hero {
            background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 100%);
            padding: 3rem 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            color: white;
        }

        .search-hero h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        .search-hero p {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 2rem;
        }

        .search-box {
            display: grid;
            grid-template-columns: 1fr 1fr auto;
            gap: 1rem;
            background: white;
            padding: 0.75rem;
            border-radius: 12px;
            box-shadow: var(--shadow-lg);
        }

        .search-input-group {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0 1rem;
        }

        .search-input-group i {
            color: var(--text-light);
            font-size: 1.25rem;
        }

        .search-input {
            flex: 1;
            border: none;
            outline: none;
            font-size: 1rem;
            color: var(--text);
        }

        .search-input::placeholder { color: var(--text-light); }

        .search-btn {
            padding: 0.75rem 2rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s;
        }

        .search-btn:hover {
            background: var(--primary-dark);
            transform: scale(1.02);
        }

        .job-desc-box {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow);
        }

        .job-desc-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .job-desc-textarea {
            width: 100%;
            min-height: 150px;
            padding: 1rem;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            font-family: inherit;
            font-size: 0.95rem;
            resize: vertical;
        }

        .job-desc-actions {
            display: flex;
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: var(--shadow);
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .stat-icon {
            width: 50px;
            height: 50px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }

        .stat-icon.primary { background: #eef2ff; color: var(--primary); }
        .stat-icon.success { background: #d1fae5; color: var(--success); }
        .stat-icon.warning { background: #fef3c7; color: var(--warning); }
        .stat-icon.secondary { background: #cffafe; color: var(--secondary); }

        .stat-content h3 {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--dark);
        }

        .stat-content p {
            color: var(--text-light);
            font-size: 0.9rem;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 2rem;
            margin-top: 2rem;
        }

        .filters-sidebar {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            height: fit-content;
            box-shadow: var(--shadow);
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }

        .filters-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            position: sticky;
            top: 0;
            background: white;
            padding-bottom: 0.5rem;
        }

        .filters-header h3 {
            font-size: 1.25rem;
            color: var(--dark);
        }

        .clear-filters {
            color: var(--primary);
            font-size: 0.9rem;
            cursor: pointer;
            text-decoration: none;
        }

        .clear-filters:hover { text-decoration: underline; }

        .filter-section {
            margin-bottom: 1.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }

        .filter-section:last-child { border-bottom: none; }

        .filter-label {
            font-weight: 600;
            color: var(--dark);
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .filter-label i {
            color: var(--primary);
            font-size: 0.9rem;
        }

        .range-value {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: var(--text-light);
            margin-bottom: 0.5rem;
        }

        .range-slider {
            width: 100%;
            margin: 0.5rem 0;
        }

        .skill-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }

        .skill-tag {
            background: var(--light);
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .skill-tag button {
            background: none;
            border: none;
            color: var(--text-light);
            cursor: pointer;
            padding: 0;
            display: flex;
            align-items: center;
        }

        .skill-tag button:hover { color: var(--danger); }

        .tag-input {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }

        .filter-checkbox {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            padding: 0.25rem 0;
        }

        .filter-checkbox input {
            cursor: pointer;
            width: 16px;
            height: 16px;
        }

        .filter-checkbox label {
            cursor: pointer;
            font-size: 0.9rem;
        }

        .results-section {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
        }

        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .results-info {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .results-count {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--dark);
        }

        .view-toggle {
            display: flex;
            gap: 0.5rem;
        }

        .view-btn {
            padding: 0.5rem 0.75rem;
            border: 1px solid var(--border);
            background: white;
            border-radius: var(--radius);
            cursor: pointer;
            color: var(--text);
            transition: all 0.2s;
        }

        .view-btn.active {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }

        .results-controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .sort-select {
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: white;
            cursor: pointer;
            font-size: 0.9rem;
        }

        .candidates-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 1.5rem;
        }

        .candidates-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .candidate-card {
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            cursor: pointer;
            transition: all 0.2s;
            background: white;
        }

        .candidate-card:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
            border-color: var(--primary);
        }

        .candidate-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 1rem;
        }

        .candidate-info h3 {
            font-size: 1.25rem;
            color: var(--dark);
            margin-bottom: 0.25rem;
        }

        .candidate-title {
            color: var(--text-light);
            font-size: 0.95rem;
        }

        .match-badge {
            background: var(--success);
            color: white;
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .candidate-details {
            display: grid;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .detail-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
            color: var(--text);
        }

        .detail-row i {
            width: 20px;
            color: var(--text-light);
        }

        .candidate-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .badge.remote { background: #dbeafe; color: #1e40af; }
        .badge.verified { background: #d1fae5; color: #065f46; }
        .badge.urgent { background: #fee2e2; color: #991b1b; }
        .badge.featured { background: #fef3c7; color: #92400e; }

        .candidate-skills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .skill-badge {
            background: var(--light);
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            color: var(--text);
        }

        .pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
            margin-top: 2rem;
        }

        .page-btn {
            padding: 0.5rem 0.75rem;
            border: 1px solid var(--border);
            background: white;
            border-radius: var(--radius);
            cursor: pointer;
            transition: all 0.2s;
        }

        .page-btn:hover { background: var(--light); }

        .page-btn.active {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }

        .page-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            padding: 2rem;
        }

        .modal {
            background: white;
            border-radius: 16px;
            max-width: 900px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: var(--shadow-lg);
        }

        .modal-header {
            padding: 2rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: start;
        }

        .modal-close {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: var(--text-light);
        }

        .modal-close:hover { color: var(--dark); }

        .modal-body { padding: 2rem; }
        .modal-section { margin-bottom: 2rem; }

        .modal-section h4 {
            font-size: 1.1rem;
            color: var(--dark);
            margin-bottom: 1rem;
        }

        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 2000;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid var(--border);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .upload-section {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow);
        }

        .upload-area {
            border: 2px dashed var(--border);
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }

        .upload-area:hover {
            border-color: var(--primary);
            background: var(--light);
        }

        .upload-area.dragging {
            border-color: var(--primary);
            background: #eef2ff;
        }

        .upload-icon {
            font-size: 3rem;
            color: var(--primary);
            margin-bottom: 1rem;
        }

        .upload-text h3 {
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
            color: var(--dark);
        }

        .upload-text p { color: var(--text-light); }
        .file-input { display: none; }

        .upload-progress { margin-top: 1rem; }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: var(--primary);
            transition: width 0.3s;
        }

        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
        }

        .empty-state i {
            font-size: 4rem;
            color: var(--border);
            margin-bottom: 1rem;
        }

        .empty-state h3 {
            font-size: 1.5rem;
            color: var(--dark);
            margin-bottom: 0.5rem;
        }

        .empty-state p { color: var(--text-light); }

        .error-message {
            background: #fee;
            color: var(--danger);
            padding: 1rem;
            border-radius: var(--radius);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .success-message {
            background: #d1fae5;
            color: var(--success);
            padding: 1rem;
            border-radius: var(--radius);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        @media (max-width: 1024px) {
            .main-grid { grid-template-columns: 1fr; }
            .search-box { grid-template-columns: 1fr; }
        }

        @media (max-width: 768px) {
            .candidates-grid { grid-template-columns: 1fr; }
            .results-header {
                flex-direction: column;
                align-items: flex-start;
            }
            .stats-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect, useMemo } = React;

        const API_BASE_URL = 'http://localhost:5001';

        function RecruitmentApp() {
            const [candidates, setCandidates] = useState([]);
            const [filteredCandidates, setFilteredCandidates] = useState([]);
            const [loading, setLoading] = useState(false);
            const [error, setError] = useState(null);
            const [successMessage, setSuccessMessage] = useState(null);
            const [searchQuery, setSearchQuery] = useState('');
            const [locationQuery, setLocationQuery] = useState('');
            const [jobDescription, setJobDescription] = useState('');
            const [showJobDesc, setShowJobDesc] = useState(false);
            const [selectedCandidate, setSelectedCandidate] = useState(null);
            const [showModal, setShowModal] = useState(false);
            const [showUpload, setShowUpload] = useState(false);
            const [currentPage, setCurrentPage] = useState(1);
            const [sortBy, setSortBy] = useState('matchScore');
            const [viewType, setViewType] = useState('grid');
            
            const [filters, setFilters] = useState({
                experience: [0, 15],
                experienceLevel: [],
                workAuth: [],
                remote: [],
                education: [],
                availability: '',
                salaryMin: 0,
                salaryMax: 300000,
            });

            const [skillTags, setSkillTags] = useState([]);
            const [tagInput, setTagInput] = useState('');
            const [uploadProgress, setUploadProgress] = useState(0);
            const [uploading, setUploading] = useState(false);

            const [stats, setStats] = useState({
                totalCandidates: 0,
                avgExperience: 0,
                activeFilters: 0,
                matchedCandidates: 0,
                topSkills: []
            });

            const itemsPerPage = 12;

            const experienceLevels = ['Entry Level (0-2 yrs)', 'Mid Level (3-5 yrs)', 'Senior (6-10 yrs)', 'Lead/Principal (10+ yrs)'];
            const workAuthOptions = ['US Citizen', 'Green Card', 'H1B Visa', 'OPT/CPT', 'Need Sponsorship', 'Any'];
            const workTypeOptions = ['Remote', 'Hybrid', 'On-site'];
            const educationOptions = ['High School', 'Associate', "Bachelor's", "Master's", 'PhD', 'Any'];

            const fetchCandidates = async () => {
                setLoading(true);
                setError(null);
                
                try {
                    const response = await fetch(`${API_BASE_URL}/api/search`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            query: searchQuery,
                            location: locationQuery,
                            jobDescription: jobDescription,
                            filters: {
                                experience: filters.experience,
                                skills: skillTags,
                                workAuth: filters.workAuth,
                                remote: filters.remote,
                                education: filters.education,
                                experienceLevel: filters.experienceLevel
                            }
                        })
                    });
                    
                    if (!response.ok) throw new Error('Failed to fetch candidates');
                    
                    const data = await response.json();
                    setCandidates(data);
                    setFilteredCandidates(data);
                } catch (err) {
                    setError(err.message);
                    console.error('Error fetching candidates:', err);
                    setCandidates([]);
                    setFilteredCandidates([]);
                } finally {
                    setLoading(false);
                }
            };

            const fetchStats = async () => {
                try {
                    const response = await fetch(`${API_BASE_URL}/api/stats`);
                    const data = await response.json();
                    
                    setStats({
                        totalCandidates: data.total || 0,
                        avgExperience: Math.round(data.avg_years || 0),
                        activeFilters: calculateActiveFilters(),
                        matchedCandidates: filteredCandidates.length,
                        topSkills: data.top_skills ? data.top_skills.slice(0, 8) : []
                    });
                } catch (err) {
                    console.error('Error fetching stats:', err);
                }
            };

            const calculateActiveFilters = () => {
                let count = 0;
                if (skillTags.length > 0) count += skillTags.length;
                if (filters.experienceLevel.length > 0) count += filters.experienceLevel.length;
                if (filters.workAuth.length > 0) count += filters.workAuth.length;
                if (filters.remote.length > 0) count += filters.remote.length;
                if (filters.education.length > 0) count += filters.education.length;
                if (locationQuery) count++;
                if (jobDescription) count++;
                return count;
            };

            const handleFileUpload = async (files) => {
                if (!files || files.length === 0) return;

                setUploading(true);
                setUploadProgress(0);

                const formData = new FormData();
                for (let i = 0; i < files.length; i++) {
                    formData.append('files', files[i]);
                }

                try {
                    const response = await fetch(`${API_BASE_URL}/api/upload`, {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    
                    if (result.ok) {
                        setUploadProgress(100);
                        setSuccessMessage(`Successfully uploaded ${result.queued} resume(s)`);
                        setTimeout(() => {
                            setUploading(false);
                            setShowUpload(false);
                            setSuccessMessage(null);
                            fetchCandidates();
                            fetchStats();
                        }, 2000);
                    } else {
                        throw new Error(result.message || 'Upload failed');
                    }
                } catch (err) {
                    setError('Upload failed: ' + err.message);
                    setUploading(false);
                }
            };

            useEffect(() => {
                fetchCandidates();
                fetchStats();
            }, []);

            useEffect(() => {
                const debounceTimer = setTimeout(() => {
                    fetchCandidates();
                }, 500);
                
                return () => clearTimeout(debounceTimer);
            }, [searchQuery, locationQuery, jobDescription, filters.experience, skillTags, 
                filters.workAuth, filters.remote, filters.education, filters.experienceLevel]);

            useEffect(() => {
                fetchStats();
            }, [filteredCandidates, filters, skillTags, locationQuery, jobDescription]);

            useEffect(() => {
                let result = [...candidates];

                result.sort((a, b) => {
                    switch(sortBy) {
                        case 'matchScore':
                            return (b.matchScore || 0) - (a.matchScore || 0);
                        case 'experience':
                            return (b.experience || 0) - (a.experience || 0);
                        case 'name':
                            return (a.name || '').localeCompare(b.name || '');
                        default:
                            return 0;
                    }
                });

                setFilteredCandidates(result);
                setCurrentPage(1);
            }, [candidates, sortBy]);

            const paginatedCandidates = useMemo(() => {
                const startIndex = (currentPage - 1) * itemsPerPage;
                return filteredCandidates.slice(startIndex, startIndex + itemsPerPage);
            }, [filteredCandidates, currentPage]);

            const totalPages = Math.ceil(filteredCandidates.length / itemsPerPage);

            const handleAddSkillTag = (e) => {
                if (e.key === 'Enter' && tagInput.trim()) {
                    if (!skillTags.includes(tagInput.trim())) {
                        setSkillTags([...skillTags, tagInput.trim()]);
                    }
                    setTagInput('');
                }
            };

            const handleRemoveSkillTag = (index) => {
                setSkillTags(skillTags.filter((_, i) => i !== index));
            };

            const toggleFilter = (filterName, value) => {
                setFilters(prev => {
                    const current = prev[filterName];
                    if (current.includes(value)) {
                        return { ...prev, [filterName]: current.filter(v => v !== value) };
                    } else {
                        return { ...prev, [filterName]: [...current, value] };
                    }
                });
            };

            const clearFilters = () => {
                setFilters({
                    experience: [0, 15],
                    experienceLevel: [],
                    workAuth: [],
                    remote: [],
                    education: [],
                    availability: '',
                    salaryMin: 0,
                    salaryMax: 300000,
                });
                setSkillTags([]);
                setSearchQuery('');
                setLocationQuery('');
                setJobDescription('');
            };

            const openCandidateModal = (candidate) => {
                setSelectedCandidate(candidate);
                setShowModal(true);
            };

            const handleJobDescriptionMatch = () => {
                if (jobDescription.trim()) {
                    fetchCandidates();
                    setShowJobDesc(false);
                }
            };

            return (
                <div>
                    <header className="header">
                        <div className="header-content">
                            <div className="logo">
                                <i className="fas fa-users-cog"></i>
                                <span>TalentHub Pro</span>
                            </div>
                            <nav className="nav-items">
                                <a href="#" className="nav-item">Dashboard</a>
                                <a href="#" className="nav-item">Candidates</a>
                                <a href="#" className="nav-item">Analytics</a>
                                <button 
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => setShowJobDesc(!showJobDesc)}
                                >
                                    <i className="fas fa-file-alt"></i>
                                    {showJobDesc ? 'Hide' : 'Job Description'}
                                </button>
                                <button 
                                    className="btn btn-primary"
                                    onClick={() => setShowUpload(!showUpload)}
                                >
                                    <i className="fas fa-upload"></i>
                                    Upload Resumes
                                </button>
                            </nav>
                        </div>
                    </header>

                    <div className="container">
                        {successMessage && (
                            <div className="success-message">
                                <i className="fas fa-check-circle"></i>
                                {successMessage}
                                <button 
                                    onClick={() => setSuccessMessage(null)}
                                    style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer' }}
                                >
                                    <i className="fas fa-times"></i>
                                </button>
                            </div>
                        )}

                        {error && (
                            <div className="error-message">
                                <i className="fas fa-exclamation-circle"></i>
                                {error}
                                <button 
                                    onClick={() => setError(null)}
                                    style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer' }}
                                >
                                    <i className="fas fa-times"></i>
                                </button>
                            </div>
                        )}

                        {showUpload && (
                            <div className="upload-section">
                                <h2 style={{ marginBottom: '1rem' }}>Upload Resumes</h2>
                                <div 
                                    className="upload-area"
                                    onClick={() => document.getElementById('fileInput').click()}
                                    onDragOver={(e) => {
                                        e.preventDefault();
                                        e.currentTarget.classList.add('dragging');
                                    }}
                                    onDragLeave={(e) => {
                                        e.currentTarget.classList.remove('dragging');
                                    }}
                                    onDrop={(e) => {
                                        e.preventDefault();
                                        e.currentTarget.classList.remove('dragging');
                                        handleFileUpload(e.dataTransfer.files);
                                    }}
                                >
                                    <div className="upload-icon">
                                        <i className="fas fa-cloud-upload-alt"></i>
                                    </div>
                                    <div className="upload-text">
                                        <h3>Drop files here or click to upload</h3>
                                        <p>Supports PDF, DOCX, DOC, TXT, PNG, JPG</p>
                                    </div>
                                    <input 
                                        id="fileInput"
                                        type="file" 
                                        className="file-input"
                                        multiple
                                        accept=".pdf,.docx,.doc,.txt,.png,.jpg,.jpeg"
                                        onChange={(e) => handleFileUpload(e.target.files)}
                                    />
                                </div>
                                {uploading && (
                                    <div className="upload-progress">
                                        <p>Uploading... {uploadProgress}%</p>
                                        <div className="progress-bar">
                                            <div 
                                                className="progress-fill" 
                                                style={{ width: `${uploadProgress}%` }}
                                            ></div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {showJobDesc && (
                            <div className="job-desc-box">
                                <div className="job-desc-header">
                                    <h3><i className="fas fa-file-alt"></i> Job Description Matching</h3>
                                    <button 
                                        className="btn btn-outline btn-sm"
                                        onClick={() => {
                                            setJobDescription('');
                                            setShowJobDesc(false);
                                        }}
                                    >
                                        Clear
                                    </button>
                                </div>
                                <textarea
                                    className="job-desc-textarea"
                                    placeholder="Paste the job description here and we'll find the best matching candidates based on skills, experience, and requirements..."
                                    value={jobDescription}
                                    onChange={(e) => setJobDescription(e.target.value)}
                                />
                                <div className="job-desc-actions">
                                    <button 
                                        className="btn btn-primary"
                                        onClick={handleJobDescriptionMatch}
                                        disabled={!jobDescription.trim()}
                                    >
                                        <i className="fas fa-search"></i>
                                        Find Matching Candidates
                                    </button>
                                    <button 
                                        className="btn btn-outline"
                                        onClick={() => setShowJobDesc(false)}
                                    >
                                        Cancel
                                    </button>
                                </div>
                            </div>
                        )}

                        <div className="search-hero">
                            <h1>Find Your Perfect Candidate</h1>
                            <p>Search through {stats.totalCandidates} qualified professionals with advanced filters</p>
                            <div className="search-box">
                                <div className="search-input-group">
                                    <i className="fas fa-search"></i>
                                    <input 
                                        type="text" 
                                        className="search-input"
                                        placeholder="Job title, keywords, or skills"
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                    />
                                </div>
                                <div className="search-input-group">
                                    <i className="fas fa-map-marker-alt"></i>
                                    <input 
                                        type="text" 
                                        className="search-input"
                                        placeholder="City, state, or zip code"
                                        value={locationQuery}
                                        onChange={(e) => setLocationQuery(e.target.value)}
                                    />
                                </div>
                                <button 
                                    className="search-btn"
                                    onClick={fetchCandidates}
                                >
                                    <i className="fas fa-search"></i>
                                    Search
                                </button>
                            </div>
                        </div>

                        <div className="stats-grid">
                            <div className="stat-card">
                                <div className="stat-icon primary">
                                    <i className="fas fa-database"></i>
                                </div>
                                <div className="stat-content">
                                    <h3>{stats.totalCandidates}</h3>
                                    <p>Total in Database</p>
                                </div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-icon success">
                                    <i className="fas fa-check-circle"></i>
                                </div>
                                <div className="stat-content">
                                    <h3>{stats.matchedCandidates}</h3>
                                    <p>Matched Results</p>
                                </div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-icon warning">
                                    <i className="fas fa-briefcase"></i>
                                </div>
                                <div className="stat-content">
                                    <h3>{stats.avgExperience} yrs</h3>
                                    <p>Avg Experience</p>
                                </div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-icon secondary">
                                    <i className="fas fa-filter"></i>
                                </div>
                                <div className="stat-content">
                                    <h3>{stats.activeFilters}</h3>
                                    <p>Active Filters</p>
                                </div>
                            </div>
                        </div>

                        <div className="main-grid">
                            <aside className="filters-sidebar">
                                <div className="filters-header">
                                    <h3><i className="fas fa-sliders-h"></i> Filters</h3>
                                    <a 
                                        className="clear-filters"
                                        onClick={clearFilters}
                                    >
                                        Clear All
                                    </a>
                                </div>

                                <div className="filter-section">
                                    <label className="filter-label">
                                        <i className="fas fa-clock"></i>
                                        Years of Experience
                                    </label>
                                    <div className="range-value">
                                        <span>{filters.experience[0]} years</span>
                                        <span>{filters.experience[1] === 15 ? '15+' : filters.experience[1]} years</span>
                                    </div>
                                    <input 
                                        type="range"
                                        className="range-slider"
                                        min="0"
                                        max="15"
                                        value={filters.experience[0]}
                                        onChange={(e) => setFilters({
                                            ...filters,
                                            experience: [parseInt(e.target.value), filters.experience[1]]
                                        })}
                                    />
                                    <input 
                                        type="range"
                                        className="range-slider"
                                        min="0"
                                        max="15"
                                        value={filters.experience[1]}
                                        onChange={(e) => setFilters({
                                            ...filters,
                                            experience: [filters.experience[0], parseInt(e.target.value)]
                                        })}
                                    />
                                </div>

                                <div className="filter-section">
                                    <label className="filter-label">
                                        <i className="fas fa-layer-group"></i>
                                        Experience Level
                                    </label>
                                    {experienceLevels.map(level => (
                                        <div key={level} className="filter-checkbox">
                                            <input 
                                                type="checkbox"
                                                checked={filters.experienceLevel.includes(level)}
                                                onChange={() => toggleFilter('experienceLevel', level)}
                                            />
                                            <label>{level}</label>
                                        </div>
                                    ))}
                                </div>

                                <div className="filter-section">
                                    <label className="filter-label">
                                        <i className="fas fa-code"></i>
                                        Required Skills
                                    </label>
                                    <input 
                                        type="text"
                                        className="tag-input"
                                        placeholder="Type skill and press Enter"
                                        value={tagInput}
                                        onChange={(e) => setTagInput(e.target.value)}
                                        onKeyPress={handleAddSkillTag}
                                    />
                                    <div className="skill-tags">
                                        {skillTags.map((tag, index) => (
                                            <div key={index} className="skill-tag">
                                                {tag}
                                                <button onClick={() => handleRemoveSkillTag(index)}>
                                                    <i className="fas fa-times"></i>
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div className="filter-section">
                                    <label className="filter-label">
                                        <i className="fas fa-passport"></i>
                                        Work Authorization
                                    </label>
                                    {workAuthOptions.map(auth => (
                                        <div key={auth} className="filter-checkbox">
                                            <input 
                                                type="checkbox"
                                                checked={filters.workAuth.includes(auth)}
                                                onChange={() => toggleFilter('workAuth', auth)}
                                            />
                                            <label>{auth}</label>
                                        </div>
                                    ))}
                                </div>

                                <div className="filter-section">
                                    <label className="filter-label">
                                        <i className="fas fa-home"></i>
                                        Work Type
                                    </label>
                                    {workTypeOptions.map(type => (
                                        <div key={type} className="filter-checkbox">
                                            <input 
                                                type="checkbox"
                                                checked={filters.remote.includes(type)}
                                                onChange={() => toggleFilter('remote', type)}
                                            />
                                            <label>{type}</label>
                                        </div>
                                    ))}
                                </div>

                                <div className="filter-section">
                                    <label className="filter-label">
                                        <i className="fas fa-graduation-cap"></i>
                                        Education Level
                                    </label>
                                    {educationOptions.map(edu => (
                                        <div key={edu} className="filter-checkbox">
                                            <input 
                                                type="checkbox"
                                                checked={filters.education.includes(edu)}
                                                onChange={() => toggleFilter('education', edu)}
                                            />
                                            <label>{edu}</label>
                                        </div>
                                    ))}
                                </div>

                                {stats.topSkills && stats.topSkills.length > 0 && (
                                    <div className="filter-section">
                                        <label className="filter-label">
                                            <i className="fas fa-star"></i>
                                            Popular Skills
                                        </label>
                                        <div style={{ fontSize: '0.85rem' }}>
                                            {stats.topSkills.map(([skill, count], idx) => (
                                                <div 
                                                    key={idx}
                                                    style={{ 
                                                        padding: '0.5rem',
                                                        cursor: 'pointer',
                                                        borderRadius: '4px',
                                                        marginBottom: '0.25rem',
                                                        background: skillTags.includes(skill) ? '#eef2ff' : 'transparent'
                                                    }}
                                                    onClick={() => {
                                                        if (!skillTags.includes(skill)) {
                                                            setSkillTags([...skillTags, skill]);
                                                        }
                                                    }}
                                                >
                                                    <strong>{skill}</strong> <span style={{ color: 'var(--text-light)' }}>({count})</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </aside>

                            <section className="results-section">
                                <div className="results-header">
                                    <div className="results-info">
                                        <span className="results-count">
                                            {filteredCandidates.length} Candidates
                                        </span>
                                        {stats.activeFilters > 0 && (
                                            <span style={{ fontSize: '0.9rem', color: 'var(--text-light)' }}>
                                                with {stats.activeFilters} filter{stats.activeFilters > 1 ? 's' : ''}
                                            </span>
                                        )}
                                    </div>
                                    <div className="results-controls">
                                        <div className="view-toggle">
                                            <button 
                                                className={`view-btn ${viewType === 'grid' ? 'active' : ''}`}
                                                onClick={() => setViewType('grid')}
                                            >
                                                <i className="fas fa-th"></i>
                                            </button>
                                            <button 
                                                className={`view-btn ${viewType === 'list' ? 'active' : ''}`}
                                                onClick={() => setViewType('list')}
                                            >
                                                <i className="fas fa-list"></i>
                                            </button>
                                        </div>
                                        <select 
                                            className="sort-select"
                                            value={sortBy}
                                            onChange={(e) => setSortBy(e.target.value)}
                                        >
                                            <option value="matchScore">Best Match</option>
                                            <option value="experience">Most Experience</option>
                                            <option value="name">Name (A-Z)</option>
                                        </select>
                                    </div>
                                </div>

                                {loading ? (
                                    <div style={{ textAlign: 'center', padding: '3rem' }}>
                                        <div className="spinner" style={{ margin: '0 auto' }}></div>
                                        <p style={{ marginTop: '1rem' }}>Loading candidates...</p>
                                    </div>
                                ) : filteredCandidates.length === 0 ? (
                                    <div className="empty-state">
                                        <i className="fas fa-search"></i>
                                        <h3>No candidates found</h3>
                                        <p>Try adjusting your search criteria or filters</p>
                                        <button 
                                            className="btn btn-primary"
                                            onClick={clearFilters}
                                            style={{ marginTop: '1rem' }}
                                        >
                                            Clear All Filters
                                        </button>
                                    </div>
                                ) : (
                                    <>
                                        <div className={viewType === 'grid' ? 'candidates-grid' : 'candidates-list'}>
                                            {paginatedCandidates.map((candidate, idx) => (
                                                <div 
                                                    key={candidate.id || idx} 
                                                    className="candidate-card"
                                                    onClick={() => openCandidateModal(candidate)}
                                                >
                                                    <div className="candidate-header">
                                                        <div className="candidate-info">
                                                            <h3>{candidate.name || 'Unnamed Candidate'}</h3>
                                                            <p className="candidate-title">{candidate.title || 'Professional'}</p>
                                                        </div>
                                                        {candidate.matchScore > 0 && (
                                                            <div className="match-badge">
                                                                {candidate.matchScore}% Match
                                                            </div>
                                                        )}
                                                    </div>

                                                    <div className="candidate-badges">
                                                        {candidate.remote && <span className="badge remote"><i className="fas fa-home"></i> Remote</span>}
                                                        {candidate.experience >= 10 && <span className="badge featured"><i className="fas fa-star"></i> Senior</span>}
                                                    </div>
                                                    
                                                    <div className="candidate-details">
                                                        {candidate.location && (
                                                            <div className="detail-row">
                                                                <i className="fas fa-map-marker-alt"></i>
                                                                <span>{candidate.location}</span>
                                                            </div>
                                                        )}
                                                        <div className="detail-row">
                                                            <i className="fas fa-briefcase"></i>
                                                            <span>{candidate.experience || 0} years experience</span>
                                                        </div>
                                                        {candidate.email && (
                                                            <div className="detail-row">
                                                                <i className="fas fa-envelope"></i>
                                                                <span>{candidate.email}</span>
                                                            </div>
                                                        )}
                                                        {candidate.phone && (
                                                            <div className="detail-row">
                                                                <i className="fas fa-phone"></i>
                                                                <span>{candidate.phone}</span>
                                                            </div>
                                                        )}
                                                    </div>

                                                    {candidate.skills && candidate.skills.length > 0 && (
                                                        <div className="candidate-skills">
                                                            {candidate.skills.slice(0, 6).map((skill, idx) => (
                                                                <span key={idx} className="skill-badge">{skill}</span>
                                                            ))}
                                                            {candidate.skills.length > 6 && (
                                                                <span className="skill-badge">+{candidate.skills.length - 6} more</span>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>

                                        {totalPages > 1 && (
                                            <div className="pagination">
                                                <button 
                                                    className="page-btn"
                                                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                                    disabled={currentPage === 1}
                                                >
                                                    <i className="fas fa-chevron-left"></i>
                                                </button>
                                                {[...Array(Math.min(totalPages, 10))].map((_, i) => {
                                                    const pageNum = i + 1;
                                                    if (totalPages <= 10 || pageNum <= 3 || pageNum > totalPages - 3 || Math.abs(currentPage - pageNum) <= 1) {
                                                        return (
                                                            <button 
                                                                key={i}
                                                                className={`page-btn ${currentPage === pageNum ? 'active' : ''}`}
                                                                onClick={() => setCurrentPage(pageNum)}
                                                            >
                                                                {pageNum}
                                                            </button>
                                                        );
                                                    } else if (pageNum === 4 || pageNum === totalPages - 3) {
                                                        return <span key={i}>...</span>;
                                                    }
                                                    return null;
                                                })}
                                                <button 
                                                    className="page-btn"
                                                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                                                    disabled={currentPage === totalPages}
                                                >
                                                    <i className="fas fa-chevron-right"></i>
                                                </button>
                                            </div>
                                        )}
                                    </>
                                )}
                            </section>
                        </div>
                    </div>

                    {showModal && selectedCandidate && (
                        <div className="modal-overlay" onClick={() => setShowModal(false)}>
                            <div className="modal" onClick={(e) => e.stopPropagation()}>
                                <div className="modal-header">
                                    <div>
                                        <h2>{selectedCandidate.name}</h2>
                                        <p style={{ color: 'var(--text-light)', marginTop: '0.25rem' }}>
                                            {selectedCandidate.title}
                                        </p>
                                        {selectedCandidate.matchScore > 0 && (
                                            <div className="match-badge" style={{ marginTop: '0.5rem', display: 'inline-block' }}>
                                                {selectedCandidate.matchScore}% Match
                                            </div>
                                        )}
                                    </div>
                                    <button 
                                        className="modal-close"
                                        onClick={() => setShowModal(false)}
                                    >
                                        <i className="fas fa-times"></i>
                                    </button>
                                </div>
                                <div className="modal-body">
                                    <div className="modal-section">
                                        <h4><i className="fas fa-address-card"></i> Contact Information</h4>
                                        <div className="candidate-details">
                                            {selectedCandidate.email && (
                                                <div className="detail-row">
                                                    <i className="fas fa-envelope"></i>
                                                    <span>{selectedCandidate.email}</span>
                                                </div>
                                            )}
                                            {selectedCandidate.phone && (
                                                <div className="detail-row">
                                                    <i className="fas fa-phone"></i>
                                                    <span>{selectedCandidate.phone}</span>
                                                </div>
                                            )}
                                            {selectedCandidate.location && (
                                                <div className="detail-row">
                                                    <i className="fas fa-map-marker-alt"></i>
                                                    <span>{selectedCandidate.location}</span>
                                                </div>
                                            )}
                                            <div className="detail-row">
                                                <i className="fas fa-briefcase"></i>
                                                <span>{selectedCandidate.experience || 0} years of experience</span>
                                            </div>
                                        </div>
                                    </div>

                                    {selectedCandidate.summary && (
                                        <div className="modal-section">
                                            <h4><i className="fas fa-user"></i> Professional Summary</h4>
                                            <p style={{ color: 'var(--text)', lineHeight: '1.8' }}>
                                                {selectedCandidate.summary}
                                            </p>
                                        </div>
                                    )}

                                    {selectedCandidate.skills && selectedCandidate.skills.length > 0 && (
                                        <div className="modal-section">
                                            <h4><i className="fas fa-code"></i> Skills & Technologies</h4>
                                            <div className="candidate-skills">
                                                {selectedCandidate.skills.map((skill, idx) => (
                                                    <span key={idx} className="skill-badge">{skill}</span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {selectedCandidate.education && (
                                        <div className="modal-section">
                                            <h4><i className="fas fa-graduation-cap"></i> Education</h4>
                                            <p style={{ color: 'var(--text)' }}>{selectedCandidate.education}</p>
                                        </div>
                                    )}

                                    <div style={{ display: 'flex', gap: '1rem', marginTop: '2rem' }}>
                                        <button className="btn btn-primary" style={{ flex: 1 }}>
                                            <i className="fas fa-user-plus"></i>
                                            Add to Pipeline
                                        </button>
                                        <button className="btn btn-secondary" style={{ flex: 1 }}>
                                            <i className="fas fa-envelope"></i>
                                            Send Message
                                        </button>
                                        <button className="btn btn-outline" style={{ flex: 1 }}>
                                            <i className="fas fa-download"></i>
                                            Download Resume
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {loading && (
                        <div className="loading-overlay">
                            <div style={{ textAlign: 'center' }}>
                                <div className="spinner"></div>
                                <p style={{ marginTop: '1rem', color: 'var(--text)' }}>
                                    Searching candidates...
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            );
        }

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<RecruitmentApp />);
    </script>
</body>
</html>
'''


    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_react(path):
        start_worker_once()
        if path.startswith('api/'):
            return 'Not found', 404
        if path and os.path.exists(os.path.join('build', path)):
            return send_from_directory('build', path)
        if os.path.exists('build/index.html'):
            return send_from_directory('build', 'index.html')
        return '<h1>Run: npm run build</h1>', 404

    @app.route('/api/stats')
    def api_stats():
        """
        Lightweight global stats for the dashboard cards.
        - total: number of resumes in DB
        - avg_years: rounded average years of experience
        - top_skills: [(skill, count), ...] sorted by frequency
        (Matched results and active filters are computed on the client:
        your UI already sets matchedCandidates from the latest /api/search.)
        """
        try:
            db = system.resumes or []
            total = len(db)

            # average years of experience
            if total:
                years = []
                for r in db:
                    try:
                        years.append(int(r.get("total_years_experience") or 0))
                    except Exception:
                        years.append(0)
                avg_years = round(sum(years) / max(len(years), 1))
            else:
                avg_years = 0

            # top skills
            freq = {}
            for r in db:
                for s in (r.get("skills") or []):
                    key = (s or "").strip()
                    if not key:
                        continue
                    freq[key] = freq.get(key, 0) + 1

            # sort by count desc then alpha
            top_skills = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:50]

            return jsonify({
                "ok": True,
                "total": total,
                "avg_years": avg_years,
                "top_skills": top_skills
            }), 200

        except Exception as e:
            logger.error("stats failed", exc_info=True)
            return jsonify({"ok": False, "error": str(e)}), 500


    @app.route('/api/debug')
    def api_debug():
        """Debug endpoint to inspect database state"""
        return jsonify({
            'total_resumes': len(system.resumes),
            'database_path': system.storage_path,
            'database_exists': os.path.exists(system.storage_path),
            'sample_resume': system.resumes[0] if system.resumes else None,
            'all_candidate_ids': [r.get('candidate_id') for r in system.resumes[:10]]
        })
    
    @app.route('/api/clear_database', methods=['POST'])
    def api_clear_database():
        """Clear all resumes from the database"""
        try:
            system.resumes = []
            system.key_index = {}
            system.save_database()
            return jsonify({'ok': True, 'message': 'Database cleared successfully', 'total': 0})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    # ==================== NEW: CANDIDATE MANAGEMENT ====================
    
    @app.route('/api/candidates', methods=['GET'])
    def api_get_all_candidates():
        """Get all candidates without any filtering - for Candidates page"""
        try:
            candidates = []
            for resume in system.resumes:
                edu_items = resume.get('education') or []
                edu_str = "; ".join([e.get('degree', '') for e in edu_items if e.get('degree')])
                
                # Better location handling
                loc = None
                if resume.get('locations') and isinstance(resume['locations'], list):
                    # Filter out invalid locations
                    valid_locations = []
                    for location in resume['locations']:
                        if (isinstance(location, str) and 
                            len(location) > 2 and 
                            not re.match(r'^\d+(\.\d+)*$', location.strip()) and  # Not just numbers/version numbers
                            not location.lower() in ['client', 'pulmonary', 'consultant', 'n/a', 'na'] and
                            len(location) < 100):
                            valid_locations.append(location)
                    loc = valid_locations[0] if valid_locations else None
                elif resume.get('location') and isinstance(resume['location'], str):
                    location = resume['location']
                    if (len(location) > 2 and 
                        not re.match(r'^\d+(\.\d+)*$', location.strip()) and
                        not location.lower() in ['client', 'pulmonary', 'consultant', 'n/a', 'na'] and
                        len(location) < 100):
                        loc = location
                
                # Clean and validate name
                name = resume.get('name') or 'Unnamed'
                if isinstance(name, str):
                    # Remove obvious parsing errors from name
                    name = re.sub(r'\([^)]*\)', '', name)  # Remove parentheses content
                    name = re.sub(r'\d{4}', '', name)  # Remove years
                    name = re.sub(r'[^\w\s\-\.]', ' ', name)  # Remove special chars
                    name = re.sub(r'\s+', ' ', name).strip()  # Normalize spaces
                    
                    # If name is too long or has obvious issues, use filename
                    if len(name) > 50 or len(name.split()) > 4:
                        filename = resume.get('filename', 'unknown')
                        name = Path(filename).stem.replace('_', ' ').replace('-', ' ').title()
                    
                    if not name or name.lower() in ['unnamed', 'resume', 'cv']:
                        name = 'Unnamed Candidate'
                
                # Clean title
                title = resume.get('current_title') or 'Candidate'
                if isinstance(title, str) and len(title) > 100:
                    title = title[:100] + '...'
                
                # Validate experience
                experience = resume.get('total_years_experience', 0)
                try:
                    experience = max(0, min(50, int(experience)))  # Cap at reasonable range
                except (ValueError, TypeError):
                    experience = 0
                
                # Clean skills list
                skills = resume.get('skills') or []
                if isinstance(skills, list):
                    clean_skills = []
                    for skill in skills:
                        if isinstance(skill, str) and 2 <= len(skill) <= 30 and skill.strip():
                            clean_skills.append(skill.strip())
                    skills = clean_skills[:20]  # Limit to 20 skills
                
                candidates.append({
                    'id': resume.get('candidate_id'),
                    'name': name,
                    'title': title,
                    'location': loc,
                    'experience': experience,
                    'email': resume.get('email') or resume.get('primary_email'),
                    'phone': resume.get('phone'),
                    'skills': skills,
                    'summary': resume.get('summary', '')[:500] if resume.get('summary') else '',  # Limit summary length
                    'education': edu_str,
                    'companies': resume.get('companies', [])[:10] if isinstance(resume.get('companies'), list) else [],
                    'positions': resume.get('positions', [])[:10] if isinstance(resume.get('positions'), list) else [],
                    'status': resume.get('status', 'new'),
                    'stage': resume.get('stage', 'applied'),
                    'tags': resume.get('tags', []),
                    'notes': resume.get('notes', ''),
                    'rating': resume.get('rating', 0),
                    'applied_date': resume.get('applied_date', resume.get('extracted_at')),
                    'resume_url': resume.get('file_path', ''),
                })
            
            return jsonify({'ok': True, 'candidates': candidates, 'total': len(candidates)})
        except Exception as e:
            logger.error(f"Error getting candidates: {e}", exc_info=True)
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    @app.route('/api/candidate/<candidate_id>', methods=['GET'])
    def api_get_candidate(candidate_id):
        """Get detailed info for a single candidate"""
        try:
            for resume in system.resumes:
                if resume.get('candidate_id') == candidate_id:
                    return jsonify({'ok': True, 'candidate': resume})
            return jsonify({'ok': False, 'error': 'Candidate not found'}), 404
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    @app.route('/api/candidate/<candidate_id>/update', methods=['POST'])
    def api_update_candidate(candidate_id):
        """Update candidate status, stage, notes, rating, tags"""
        try:
            data = request.get_json() or {}
            
            for i, resume in enumerate(system.resumes):
                if resume.get('candidate_id') == candidate_id:
                    # Update fields
                    if 'status' in data:
                        resume['status'] = data['status']
                    if 'stage' in data:
                        resume['stage'] = data['stage']
                    if 'notes' in data:
                        resume['notes'] = data['notes']
                    if 'rating' in data:
                        resume['rating'] = data['rating']
                    if 'tags' in data:
                        resume['tags'] = data['tags']
                    if 'position_id' in data:
                        resume['position_id'] = data['position_id']
                    
                    system.resumes[i] = resume
                    system.save_database()
                    return jsonify({'ok': True, 'candidate': resume})
            
            return jsonify({'ok': False, 'error': 'Candidate not found'}), 404
        except Exception as e:
            logger.error(f"Error updating candidate: {e}", exc_info=True)
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    @app.route('/api/candidate/<candidate_id>/download', methods=['GET'])
    def api_download_resume(candidate_id):
        """Download the original resume file"""
        try:
            for resume in system.resumes:
                if resume.get('candidate_id') == candidate_id:
                    file_path = resume.get('file_path')
                    if file_path and os.path.exists(file_path):
                        return send_file(file_path, as_attachment=True)
                    return jsonify({'ok': False, 'error': 'Resume file not found'}), 404
            return jsonify({'ok': False, 'error': 'Candidate not found'}), 404
        except Exception as e:
            logger.error(f"Error downloading resume: {e}", exc_info=True)
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    # ==================== NEW: POSITION/JOB MANAGEMENT ====================
    
    positions_file = Path("positions.json")
    
    def load_positions():
        if positions_file.exists():
            try:
                with open(positions_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_positions(positions):
        with open(positions_file, 'w') as f:
            json.dump(positions, f, indent=2)
    
    @app.route('/api/positions', methods=['GET'])
    def api_get_positions():
        """Get all job positions"""
        try:
            positions = load_positions()
            return jsonify({'ok': True, 'positions': positions, 'total': len(positions)})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    @app.route('/api/positions', methods=['POST'])
    def api_create_position():
        """Create a new job position"""
        try:
            data = request.get_json() or {}
            positions = load_positions()
            
            new_position = {
                'id': sha1_hex(f"{data.get('title')}{datetime.now().isoformat()}")[:12],
                'title': data.get('title', 'Untitled Position'),
                'department': data.get('department', ''),
                'location': data.get('location', ''),
                'employment_type': data.get('employment_type', 'Full-time'),
                'experience_required': data.get('experience_required', ''),
                'skills_required': data.get('skills_required', []),
                'description': data.get('description', ''),
                'status': data.get('status', 'open'),  # open, closed, on-hold
                'openings': data.get('openings', 1),
                'filled': 0,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            }
            
            positions.append(new_position)
            save_positions(positions)
            
            return jsonify({'ok': True, 'position': new_position})
        except Exception as e:
            logger.error(f"Error creating position: {e}", exc_info=True)
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    @app.route('/api/positions/<position_id>', methods=['PUT'])
    def api_update_position(position_id):
        """Update a job position"""
        try:
            data = request.get_json() or {}
            positions = load_positions()
            
            for i, pos in enumerate(positions):
                if pos['id'] == position_id:
                    positions[i].update(data)
                    positions[i]['updated_at'] = datetime.now().isoformat()
                    save_positions(positions)
                    return jsonify({'ok': True, 'position': positions[i]})
            
            return jsonify({'ok': False, 'error': 'Position not found'}), 404
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    @app.route('/api/positions/<position_id>', methods=['DELETE'])
    def api_delete_position(position_id):
        """Delete a job position"""
        try:
            positions = load_positions()
            positions = [p for p in positions if p['id'] != position_id]
            save_positions(positions)
            return jsonify({'ok': True})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    # ==================== NEW: KANBAN BOARD ====================
    
    @app.route('/api/kanban/candidates', methods=['GET'])
    def api_kanban_candidates():
        """Get candidates organized by stage for kanban board"""
        try:
            position_id = request.args.get('position_id')
            
            stages = {
                'applied': [],
                'screening': [],
                'interview': [],
                'offer': [],
                'hired': [],
                'rejected': []
            }
            
            for resume in system.resumes:
                # Filter by position if specified
                if position_id and resume.get('position_id') != position_id:
                    continue
                
                stage = resume.get('stage', 'applied')
                if stage not in stages:
                    stage = 'applied'
                
                candidate = {
                    'id': resume.get('candidate_id'),
                    'name': resume.get('name') or 'Unnamed',
                    'title': resume.get('current_title') or 'Candidate',
                    'email': resume.get('email') or resume.get('primary_email'),
                    'experience': int(resume.get('total_years_experience') or 0),
                    'skills': resume.get('skills', [])[:5],  # Show top 5 skills
                    'rating': resume.get('rating', 0),
                    'status': resume.get('status', 'new'),
                    'stage': stage,
                }
                
                stages[stage].append(candidate)
            
            return jsonify({'ok': True, 'stages': stages})
        except Exception as e:
            logger.error(f"Error getting kanban data: {e}", exc_info=True)
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    @app.route('/api/kanban/move', methods=['POST'])
    def api_kanban_move():
        """Move candidate to a different stage"""
        try:
            data = request.get_json() or {}
            candidate_id = data.get('candidate_id')
            new_stage = data.get('stage')
            
            for i, resume in enumerate(system.resumes):
                if resume.get('candidate_id') == candidate_id:
                    resume['stage'] = new_stage
                    resume['updated_at'] = datetime.now().isoformat()
                    system.resumes[i] = resume
                    system.save_database()
                    return jsonify({'ok': True})
            
            return jsonify({'ok': False, 'error': 'Candidate not found'}), 404
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    # ==================== ANALYTICS ====================
    
    @app.route('/api/remove_duplicates', methods=['POST'])
    def api_remove_duplicates():
        """Remove duplicate candidates based on email or name similarity"""
        try:
            seen_emails = {}
            seen_names = {}
            duplicates_removed = 0
            unique_resumes = []
            
            for resume in system.resumes:
                email = (resume.get('email') or resume.get('primary_email') or '').lower().strip()
                name = (resume.get('name') or '').lower().strip()
                
                is_duplicate = False
                
                # Check email duplicate
                if email and email in seen_emails:
                    is_duplicate = True
                    duplicates_removed += 1
                    logger.info(f"Duplicate found (email): {name} - {email}")
                
                # Check name duplicate
                elif name and name in seen_names:
                    is_duplicate = True
                    duplicates_removed += 1
                    logger.info(f"Duplicate found (name): {name}")
                
                if not is_duplicate:
                    unique_resumes.append(resume)
                    if email:
                        seen_emails[email] = True
                    if name:
                        seen_names[name] = True
            
            system.resumes = unique_resumes
            system._build_key_index()
            system.save_database()
            
            return jsonify({
                'ok': True,
                'duplicates_removed': duplicates_removed,
                'remaining': len(unique_resumes),
                'message': f'Removed {duplicates_removed} duplicates. {len(unique_resumes)} unique candidates remaining.'
            })
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}", exc_info=True)
            return jsonify({'ok': False, 'error': str(e)}), 500
    
    @app.route('/api/analytics', methods=['GET'])
    def api_analytics():
        """Get analytics data"""
        try:
            total = len(system.resumes)
            
            # Status breakdown
            status_counts = defaultdict(int)
            stage_counts = defaultdict(int)
            
            for resume in system.resumes:
                status_counts[resume.get('status', 'new')] += 1
                stage_counts[resume.get('stage', 'applied')] += 1
            
            # Experience distribution
            exp_ranges = {'0-2': 0, '3-5': 0, '6-10': 0, '10+': 0}
            for resume in system.resumes:
                years = resume.get('total_years_experience', 0)
                if years <= 2:
                    exp_ranges['0-2'] += 1
                elif years <= 5:
                    exp_ranges['3-5'] += 1
                elif years <= 10:
                    exp_ranges['6-10'] += 1
                else:
                    exp_ranges['10+'] += 1
            
            # Top skills
            skills_freq = defaultdict(int)
            for resume in system.resumes:
                for skill in resume.get('skills', []):
                    skills_freq[skill] += 1
            
            top_skills = sorted(skills_freq.items(), key=lambda x: x[1], reverse=True)[:15]
            
            # Applications over time (last 30 days)
            from datetime import datetime, timedelta
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_applications = []
            
            for resume in system.resumes:
                applied_date = resume.get('applied_date', resume.get('extracted_at'))
                if applied_date:
                    try:
                        date_obj = datetime.fromisoformat(applied_date.replace('Z', '+00:00'))
                        if date_obj >= thirty_days_ago:
                            recent_applications.append(date_obj.date().isoformat())
                    except:
                        pass
            
            # Count by date
            date_counts = defaultdict(int)
            for date in recent_applications:
                date_counts[date] += 1
            
            timeline = sorted([{'date': k, 'count': v} for k, v in date_counts.items()], key=lambda x: x['date'])
            
            return jsonify({
                'ok': True,
                'total_candidates': total,
                'status_breakdown': dict(status_counts),
                'stage_breakdown': dict(stage_counts),
                'experience_distribution': exp_ranges,
                'top_skills': [{'skill': s, 'count': c} for s, c in top_skills],
                'applications_timeline': timeline,
                'positions': len(load_positions()),
            })
        except Exception as e:
            logger.error(f"Analytics error: {e}", exc_info=True)
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.route('/api/search', methods=['POST'])
    def api_search():
        """
        Main search endpoint used by the frontend.
        Accepts: query, location, jobDescription, filters object, and optional limit/offset
        Returns: Array of candidate objects matching the frontend structure
        
        NEW: Supports use_ai parameter for semantic search
        """
        data = request.get_json(silent=True) or {}
        
        logger.warning(f"========== API SEARCH CALLED ==========")
        logger.warning(f"Total resumes in database: {len(system.resumes)}")

        # --- inputs ---
        query           = (data.get('query') or '').strip()
        location        = (data.get('location') or '').strip().lower()
        job_description = (data.get('jobDescription') or '').strip()
        use_ai          = data.get('use_ai', False)  # NEW: AI search toggle

        filters         = data.get('filters') or {}
        exp_range       = filters.get('experience', [0, 100]) or [0, 100]
        skill_tags      = filters.get('skills') or []
        work_auth_sel   = set((filters.get('workAuth') or []))
        remote_sel      = set((filters.get('remote') or []))   # {'Remote','Hybrid','On-site'}
        education_sel   = set((filters.get('education') or []))
        level_sel       = set((filters.get('experienceLevel') or []))

        # pagination / size controls (client can pass these)
        try:
            limit  = int(data.get('limit', 100000))   # effectively "no cap" unless client asks
            offset = int(data.get('offset', 0))
        except Exception:
            limit, offset = 100000, 0

        req_skills = set(s.strip().lower() for s in skill_tags if s and s.strip())

        # --- build combined query for semantic search (optional) ---
        search_terms = []
        if query:
            search_terms.append(query)
        if job_description:
            search_terms.append(job_description)
        combined_query = ' '.join(search_terms).strip()

        # --- fetch candidates ---
        # If there's a query, use the ranked search; otherwise start from full DB
        if combined_query:
            # pull a bit extra to allow server-side filtering then windowing
            base = system.search(combined_query, top_n=max(limit + offset, 1000), use_ai=use_ai)
        else:
            # no ranking when no query; wrap to match downstream format
            base = [{'resume': r, 'score': 0} for r in system.resumes]
        
        logger.warning(f"========== BASE RESULTS: {len(base)} resumes ==========")

        # --- mapping helper ---
        def to_candidate(rwrap):
            res = rwrap.get('resume', {}) or {}
            edu_items = res.get('education') or []
            edu_str = "; ".join([e.get('degree', '') for e in edu_items if e.get('degree')])

            # location: first known
            loc = None
            if res.get('locations'):
                loc = res['locations'][0]
            elif res.get('location'):
                loc = res.get('location')

            # work type normalization: one of {'Remote','Hybrid','On-site'} when possible
            work_type = None
            rt = (res.get('work_type') or res.get('work_mode') or '').strip().lower()
            if rt:
                if 'remote' in rt:
                    work_type = 'Remote'
                elif 'hybrid' in rt:
                    work_type = 'Hybrid'
                elif 'on' in rt or 'office' in rt or 'onsite' in rt.replace('-', ''):
                    work_type = 'On-site'
            elif res.get('remote') is True:
                work_type = 'Remote'

            cand = {
                'id':         res.get('candidate_id'),
                'name':       res.get('name') or 'Unnamed',
                'title':      res.get('current_title') or 'Candidate',
                'location':   loc,
                'experience': int(res.get('total_years_experience') or 0),
                'email':      res.get('email') or res.get('primary_email'),
                'phone':      res.get('phone'),
                'skills':     res.get('skills') or [],
                'matchScore': int(rwrap.get('score', 0)),
                'summary':    res.get('summary') or '',
                'education':  edu_str,
                'availability': res.get('availability'),
                'salary':       res.get('salary'),
                'remote':       work_type == 'Remote',
                'workType':     work_type,  # keep the string too
                'workAuth':     res.get('work_authorization'),
            }
            return cand

        mapped = [to_candidate(x) for x in base]
        logger.warning(f"========== MAPPED: {len(mapped)} candidates ==========")
        
        # --- filtering helper ---
        def in_experience_level(years: int) -> bool:
            if not level_sel:
                return True
            y = years or 0
            ok = False
            # match the UI labels
            if 'Entry Level (0-2 yrs)' in level_sel and y <= 2:
                ok = True
            if 'Mid Level (3-5 yrs)' in level_sel and 3 <= y <= 5:
                ok = True
            if 'Senior (6-10 yrs)' in level_sel and 6 <= y <= 10:
                ok = True
            if 'Lead/Principal (10+ yrs)' in level_sel and y >= 10:
                ok = True
            return ok

        def passes_filters(c):
            # experience range slider - only filter if not default range
            years = c['experience'] or 0
            # Only apply if the range is actually restrictive (not [0, 15] which is frontend default)
            if exp_range not in ([0, 100], [0, 15]) and not (exp_range[0] <= years <= exp_range[1]):
                return False

            # skills (any overlap)
            if req_skills:
                have = set(s.lower() for s in (c.get('skills') or []))
                if not (have & req_skills):
                    return False

            # location substring match
            if location:
                loc = (c.get('location') or '').lower()
                if not loc or location not in loc:
                    return False

            # experience level buckets
            if level_sel and not in_experience_level(years):
                return False

            # education substring (any)
            if education_sel and c.get('education'):
                edu_low = c['education'].lower()
                if not any(e.lower() in edu_low for e in education_sel):
                    return False

            # work authorization (any)
            if work_auth_sel:
                wa = c.get('workAuth') or ''
                if not any(sel.lower() in str(wa).lower() for sel in work_auth_sel):
                    return False

            # work type (Remote/Hybrid/On-site) – any
            if remote_sel:
                wt = c.get('workType')
                if not wt or wt not in remote_sel:
                    return False

            return True

        filtered = [c for c in mapped if passes_filters(c)]
        
        logger.warning(f"========== FILTERED: {len(filtered)} (from {len(mapped)}) ==========")
        logger.warning(f"FILTERS - exp_range: {exp_range}, req_skills: {req_skills}, location: {location}, education_sel: {education_sel}, work_auth_sel: {work_auth_sel}, remote_sel: {remote_sel}, level_sel: {level_sel}")

        # --- window & return ---
        sliced = filtered[offset: offset + limit]
        logger.warning(f"!!!! RETURNING {len(sliced)} RESULTS (filtered={len(filtered)}, offset={offset}, limit={limit}) !!!!")
        return jsonify(sliced), 200


        def to_candidate(r):
            res = r.get("resume", {}) or {}
            edu_items = res.get("education") or []
            edu_str = "; ".join([e.get("degree","") for e in edu_items if e.get("degree")])

            # Infer location display (first known)
            loc = (res.get("locations") or [None])[0]

            cand = {
                "id": res.get("candidate_id"),
                "name": res.get("name") or "Unnamed",
                "title": res.get("current_title") or "Candidate",
                "location": loc,
                "experience": int(res.get("total_years_experience") or 0),
                "email": res.get("email"),
                "phone": res.get("phone"),
                "skills": res.get("skills") or [],
                "matchScore": int(r.get("score", 0)),
                "summary": res.get("summary") or "",
                "education": edu_str,
                "availability": None,
                "salary": None,
                "remote": None,
                "workAuth": res.get("work_authorization"),
            }
            return cand

        # Map and apply lightweight client-specified filters server-side
        mapped = [to_candidate(x) for x in results]

        def passes_filters(c):
            # Experience range
            if not (exp_range[0] <= (c["experience"] or 0) <= exp_range[1]):
                return False
            
            # Skills contain any/all? (here: any)
            if req_skills:
                have = set([s.lower() for s in (c["skills"] or [])])
                if not (have & req_skills):
                    return False
            
            # Location contains (substring match)
            if location:
                if not (c["location"] and location in c["location"].lower()):
                    return False
            
            # Experience level filter
            if experience_level:
                exp_years = c["experience"] or 0
                level_match = False
                for level in experience_level:
                    if level == 'Entry' and exp_years <= 2:
                        level_match = True
                    elif level == 'Mid' and 3 <= exp_years <= 7:
                        level_match = True
                    elif level == 'Senior' and 8 <= exp_years <= 12:
                        level_match = True
                    elif level == 'Lead' and exp_years >= 13:
                        level_match = True
                if not level_match:
                    return False
            
            # Education filter
            if education and c["education"]:
                edu_lower = c["education"].lower()
                edu_match = False
                for edu in education:
                    if edu.lower() in edu_lower:
                        edu_match = True
                        break
                if not edu_match:
                    return False
            
            return True

        filtered = [c for c in mapped if passes_filters(c)]
        # Keep top N (UI paginates)
        return jsonify(filtered[:200]), 200

    @app.route('/api/autocomplete', methods=['GET'])
    def api_autocomplete():
        """Real-time autocomplete for search suggestions"""
        try:
            query = (request.args.get('q') or '').strip().lower()
            field = request.args.get('field', 'all')
            
            if not query or len(query) < 2:
                return jsonify([])
            
            suggestions = set()
            
            for resume in system.resumes:
                if field in ['all', 'skills']:
                    for skill in (resume.get('skills') or []):
                        if query in skill.lower():
                            suggestions.add(skill)
                
                if field in ['all', 'titles']:
                    title = resume.get('current_title') or ''
                    if query in title.lower():
                        suggestions.add(title)
                
                if field in ['all', 'companies']:
                    for company in (resume.get('companies') or []):
                        if query in company.lower():
                            suggestions.add(company)
                
                if field in ['all', 'locations']:
                    locations = resume.get('locations') or []
                    if isinstance(locations, list):
                        for loc in locations:
                            if isinstance(loc, str) and query in loc.lower():
                                suggestions.add(loc)
            
            results = []
            starts_with = [s for s in suggestions if s.lower().startswith(query)]
            contains = [s for s in suggestions if not s.lower().startswith(query)]
            
            results = sorted(starts_with)[:10] + sorted(contains)[:10]
            return jsonify(results[:15])
            
        except Exception as e:
            logger.error(f"Autocomplete error: {e}", exc_info=True)
            return jsonify([])

    @app.route('/api/candidate/<candidate_id>/details', methods=['GET'])
    def api_candidate_details(candidate_id):
        """Get full candidate details for modal/detailed view"""
        try:
            for resume in system.resumes:
                if resume.get('candidate_id') == candidate_id:
                    edu_items = resume.get('education') or []
                    edu_list = []
                    for e in edu_items:
                        if isinstance(e, dict):
                            edu_list.append({
                                'degree': e.get('degree', ''),
                                'institution': e.get('institution', ''),
                                'year': e.get('year', ''),
                                'details': e.get('details', '')
                            })
                    
                    loc = None
                    if resume.get('locations') and isinstance(resume['locations'], list):
                        valid_locations = []
                        for location in resume['locations']:
                            if (isinstance(location, str) and 
                                len(location) > 2 and 
                                not re.match(r'^\d+(\.\d+)*$', location.strip()) and
                                len(location) < 100):
                                valid_locations.append(location)
                        loc = valid_locations[0] if valid_locations else None
                    
                    candidate = {
                        'id': resume.get('candidate_id'),
                        'name': resume.get('name') or 'Unnamed',
                        'title': resume.get('current_title') or 'Candidate',
                        'location': loc,
                        'experience': int(resume.get('total_years_experience') or 0),
                        'email': resume.get('email') or resume.get('primary_email'),
                        'phone': resume.get('phone'),
                        'skills': resume.get('skills') or [],
                        'skills_by_category': resume.get('skills_by_category') or {},
                        'summary': resume.get('summary', ''),
                        'education': edu_list,
                        'certifications': resume.get('certifications') or [],
                        'positions': resume.get('positions') or [],
                        'companies': resume.get('companies') or [],
                        'locations': resume.get('locations') or [],
                        'status': resume.get('status', 'new'),
                        'stage': resume.get('stage', 'applied'),
                        'tags': resume.get('tags', []),
                        'notes': resume.get('notes', ''),
                        'rating': resume.get('rating', 0),
                        'applied_date': resume.get('applied_date', resume.get('extracted_at')),
                        'file_path': resume.get('file_path', ''),
                        'urls': resume.get('urls', {}),
                    }
                    
                    return jsonify({'ok': True, 'candidate': candidate})
            
            return jsonify({'ok': False, 'error': 'Candidate not found'}), 404
        except Exception as e:
            logger.error(f"Error getting candidate details: {e}", exc_info=True)
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.route('/api/filters/options', methods=['GET'])
    def api_filter_options():
        """Get all available filter options dynamically from database"""
        try:
            skills = set()
            locations = set()
            companies = set()
            
            for resume in system.resumes:
                for skill in (resume.get('skills') or []):
                    if skill and len(skill) > 1:
                        skills.add(skill)
                
                locs = resume.get('locations') or []
                if isinstance(locs, list):
                    for loc in locs:
                        if isinstance(loc, str) and 2 < len(loc) < 100:
                            locations.add(loc)
                
                for company in (resume.get('companies') or []):
                    if company and len(company) > 1:
                        companies.add(company)
            
            skill_counts = {}
            for resume in system.resumes:
                for skill in (resume.get('skills') or []):
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
            
            top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:50]
            
            return jsonify({
                'ok': True,
                'skills': [{'value': s, 'count': c} for s, c in top_skills],
                'locations': sorted(list(locations))[:100],
                'companies': sorted(list(companies))[:100]
            })
            
        except Exception as e:
            logger.error(f"Error getting filter options: {e}", exc_info=True)
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.post('/api/advanced_search')
    def api_advanced_search():
        """
        Adapter that converts internal search results ({'resume':..., 'score':...})
        into flat candidate objects used by the React UI.
        Supports jobDescription and advanced filters.
        """
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        location = (payload.get("location") or "").strip().lower()
        job_description = (payload.get("jobDescription") or "").strip()
        
        filters = payload.get("filters") or {}
        exp_range = filters.get("experience") or [0, 100]
        req_skills = set([s.strip().lower() for s in (filters.get("skills") or []) if s.strip()])
        work_auth = filters.get("workAuth", [])
        remote = filters.get("remote", [])
        education = filters.get("education", [])
        experience_level = filters.get("experienceLevel", [])

        # Build combined search query
        search_terms = []
        if query:
            search_terms.append(query)
        if job_description:
            search_terms.append(job_description)
        
        combined_query = ' '.join(search_terms).strip()
        
        # Run the search (top 200)
        results = system.search(combined_query, top_n=200) if combined_query else [{'resume': r, 'score': 0} for r in system.resumes[:200]]

        def to_candidate(r):
            res = r.get("resume", {}) or {}
            edu_items = res.get("education") or []
            edu_str = "; ".join([e.get("degree","") for e in edu_items if e.get("degree")])

            # Infer location display (first known)
            loc = (res.get("locations") or [None])[0]

            cand = {
                "id": res.get("candidate_id"),
                "name": res.get("name") or "Unnamed",
                "title": res.get("current_title") or "Candidate",
                "location": loc,
                "experience": int(res.get("total_years_experience") or 0),
                "email": res.get("email"),
                "phone": res.get("phone"),
                "skills": res.get("skills") or [],
                "matchScore": int(r.get("score", 0)),
                "summary": res.get("summary") or "",
                "education": edu_str,
                "availability": None,
                "salary": None,
                "remote": None,
                "workAuth": res.get("work_authorization"),
            }
            return cand

        # Map and apply lightweight client-specified filters server-side
        mapped = [to_candidate(x) for x in results]

        def passes_filters(c):
            # Experience range
            if not (exp_range[0] <= (c["experience"] or 0) <= exp_range[1]):
                return False
            
            # Skills contain any/all? (here: any)
            if req_skills:
                have = set([s.lower() for s in (c["skills"] or [])])
                if not (have & req_skills):
                    return False
            
            # Location contains (substring match)
            if location:
                if not (c["location"] and location in c["location"].lower()):
                    return False
            
            # Experience level filter
            if experience_level:
                exp_years = c["experience"] or 0
                level_match = False
                for level in experience_level:
                    if level == 'Entry' and exp_years <= 2:
                        level_match = True
                    elif level == 'Mid' and 3 <= exp_years <= 7:
                        level_match = True
                    elif level == 'Senior' and 8 <= exp_years <= 12:
                        level_match = True
                    elif level == 'Lead' and exp_years >= 13:
                        level_match = True
                if not level_match:
                    return False
            
            # Education filter
            if education and c["education"]:
                edu_lower = c["education"].lower()
                edu_match = False
                for edu in education:
                    if edu.lower() in edu_lower:
                        edu_match = True
                        break
                if not edu_match:
                    return False
            
            return True

        filtered = [c for c in mapped if passes_filters(c)]
        # Keep top N (UI paginates)
        return jsonify(filtered[:200]), 200

    @app.route('/api/list')
    def api_list():
        results = [{'resume': r, 'score': 0} for r in system.resumes]
        return jsonify(results)

    @app.route('/api/progress')
    def api_progress():
        with progress_lock:
            out = {
                "queued":      progress.get("queued", 0),
                "processing":  progress.get("processing", 0),
                "processed":   progress.get("processed", 0),
                "success":     progress.get("success", 0),
                "skipped":     progress.get("skipped", 0),
                "current":     progress.get("current", ""),
                "last_error":  progress.get("last_error", ""),
                "db_total":    len(system.resumes),
                "queue_len":   job_q.qsize()
            }
        return jsonify(out)

    # Cache for suggestions to improve performance
    suggestion_cache = {'skills': {}, 'companies': {}, 'positions': {}, 'last_update': 0}
    
    def refresh_suggestion_cache():
        """Rebuild suggestion cache from current resumes"""
        import time
        skills_agg = defaultdict(int)
        companies_agg = defaultdict(int)
        positions_agg = defaultdict(int)
        
        for r in system.resumes:
            for s in (r.get('skills') or []): 
                skills_agg[s.lower()] += 1
            for c in (r.get('companies') or []): 
                companies_agg[c.lower()] += 1
            for p in (r.get('positions') or []): 
                positions_agg[p.lower()] += 1
        
        suggestion_cache['skills'] = dict(skills_agg)
        suggestion_cache['companies'] = dict(companies_agg)
        suggestion_cache['positions'] = dict(positions_agg)
        suggestion_cache['last_update'] = time.time()
        logger.info(f"Suggestion cache refreshed: {len(skills_agg)} skills, {len(companies_agg)} companies, {len(positions_agg)} positions")
    
    @app.route('/api/suggest')
    def api_suggest():
        import time
        q = (request.args.get('q') or '').strip().lower()
        bucket = request.args.get('bucket', 'skills')
        
        # Refresh cache if stale (older than 30 seconds) or empty
        if (time.time() - suggestion_cache['last_update'] > 30 or 
            not suggestion_cache.get(bucket)):
            refresh_suggestion_cache()
        
        agg = suggestion_cache.get(bucket, {})
        
        # Filter and sort
        if q:
            # Fast filtering with comprehension
            items = [k for k in agg.keys() if q in k]
        else:
            items = list(agg.keys())
        
        # Sort by count (descending) then alphabetically
        items.sort(key=lambda x: (-agg[x], x))
        
        return jsonify([{'value': it, 'count': agg[it]} for it in items[:20]])

    @app.route('/api/upload', methods=['POST'])
    def api_upload():
        start_worker_once()

        logger.info(f"Upload request received - Content-Type: {request.content_type}")
        logger.info(f"Request files keys: {list(request.files.keys())}")
        logger.info(f"Request form keys: {list(request.form.keys())}")
        
        files = request.files.getlist('files')
        logger.info(f"Files received via getlist('files'): {len(files)}")
        
        if not files:
            # Try alternative ways to get files
            all_files = []
            for key in request.files.keys():
                all_files.extend(request.files.getlist(key))
            
            if all_files:
                logger.info(f"Found {len(all_files)} files via alternative method")
                files = all_files
            else:
                logger.warning("No files received in upload request")
                return jsonify({'ok': False, 'total': 0, 'queued': 0, 'message': 'No files received'}), 200

        total = len(files)
        queued_now = 0
        skipped = 0

        logger.info(f"Upload request received with {total} files")

        for file in files:
            filename = file.filename or ""
            ext = Path(filename).suffix.lower()

            # filter by extension
            if ext not in ALLOWED_EXTS:
                logger.debug(f"Skipping {filename} - invalid extension {ext}")
                skipped += 1
                continue

            # soft per-file size check
            try:
                file.stream.seek(0, os.SEEK_END)
                size = file.stream.tell()
                file.stream.seek(0)
            except Exception:
                size = 0
            if PER_FILE_MAX_BYTES and size and size > PER_FILE_MAX_BYTES:
                logger.warning(f"Skipping {filename} - file too large ({size} bytes)")
                skipped += 1
                continue

            # save quickly, enqueue for background parsing
            try:
                dest = incoming_dir / filename
                i = 1
                while dest.exists():
                    dest = incoming_dir / f"{Path(filename).stem}__{i}{ext}"
                    i += 1
                file.save(str(dest))
                job_q.put(dest)
                queued_now += 1
                logger.debug(f"Queued {filename} for parsing")
            except Exception as e:
                logger.error(f"Error saving {filename}: {e}")
                with progress_lock:
                    progress["last_error"] = str(e)[:500]
                skipped += 1

        with progress_lock:
            progress["queued"] += queued_now

        logger.info(f"Upload complete: queued={queued_now}, skipped={skipped}, total={total}")
        return jsonify({
            'ok': True, 
            'total': total, 
            'queued': queued_now, 
            'skipped': skipped,
            'message': f'Queued {queued_now} files for processing'
        }), 200

    return app

def run_cli():
    system = ResumeSearchSystem()

    print("\nAdvanced Resume Search System - CLI Mode")
    print("Tip: Run with --web flag for web UI\n")

    while True:
        print("\nOptions:")
        print("  1. Upload resumes from directory")
        print("  2. Search resumes")
        print("  3. List all resumes")
        print("  4. Show statistics")
        print("  0. Exit")

        choice = input("\nChoice: ").strip()

        if choice == '1':
            path = input("Directory path: ").strip()
            if path:
                system.upload_directory(path)
                print(f"Upload complete. Total resumes: {len(system.resumes)}")

        elif choice == '2':
            query = input("Search query: ").strip()
            if query:
                results = system.search(query, top_n=10)
                print(f"\nFound {len(results)} results:\n")
                for i, r in enumerate(results, 1):
                    res = r['resume']
                    print(f"{i}. {res['name']} (Score: {r['score']})")
                    print(f"   Title: {res.get('current_title', 'N/A')}")
                    print(f"   Email: {res.get('email', 'N/A')}")
                    print(f"   Experience: {res.get('total_years_experience', 0)} years")
                    print(f"   Skills: {', '.join(res.get('skills', [])[:5])}")
                    if res.get('companies'):
                        print(f"   Companies: {', '.join(res['companies'][:3])}")
                    print()

        elif choice == '3':
            print(f"\nTotal: {len(system.resumes)} resumes\n")
            for i, r in enumerate(system.resumes[:20], 1):
                print(f"{i}. {r['name']} - {r.get('total_skills', 0)} skills, {r.get('total_years_experience', 0)} years exp")

        elif choice == '4':
            stats = system.get_statistics()
            print(f"\nStatistics:")
            print(f"   Total resumes: {stats.get('total', 0)}")
            print(f"   With email: {stats.get('with_email', 0)}")
            print(f"   Avg experience: {stats.get('avg_years', 0):.1f} years")
            print(f"\n   Top skills:")
            for skill, count in stats.get('top_skills', [])[:10]:
                print(f"   - {skill}: {count}")
            if stats.get('top_companies'):
                print(f"\n   Top companies:")
                for company, count in stats.get('top_companies', [])[:5]:
                    print(f"   - {company}: {count}")

        elif choice == '0':
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    import sys
    if '--web' in sys.argv or '-w' in sys.argv:
        app = create_web_ui()
        if app:
            print("\nStarting Advanced Resume Search System...")
            print("Open your browser to: http://localhost:5001\n")
            app.run(debug=False, port=5001, host='0.0.0.0')
        else:
            print("\nFlask not installed. Install with: pip install flask")
            print("Or run without --web flag for CLI mode\n")
    else:
        run_cli()