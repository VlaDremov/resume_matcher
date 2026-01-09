# Resume Keyword Matcher v2.0

A Python tool that generates keyword-optimized resume variants and matches them to job descriptions using GPT-5 powered analysis and a modern web interface.

## Features

### GPT-5 Powered Analysis
- **Semantic Matching**: Uses OpenAI embeddings for understanding synonyms and context
- **Intelligent Bullet Rewriting**: GPT-5 rewrites experience bullets to emphasize job-relevant keywords
- **Structured Scoring**: Returns relevancy score (0-100) with detailed analysis
- **Vacancy-Aware Keyword Base**: Saved vacancy descriptions are used to bias keyword extraction for future analyses

### Web Interface (FastAPI + React)
- Modern dark-themed UI
- Paste job descriptions and get instant analysis
- PDF resume preview
- Save vacancies to database for future reference

### 3 Resume Variants
- **Research & Advanced ML** - Experiments, statistical rigor, model optimization
- **Applied ML & Production Systems** - MLOps, deployment, pipelines, monitoring
- **Generative AI & LLM Engineering** - LLM apps, RAG, agents, prompt engineering

## Installation

```bash
# Clone or navigate to the project
cd resume_matcher

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Set OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"
```

### Frontend Setup (Optional)
```bash
cd frontend
npm install
```

### LaTeX Compilation (Optional)
To compile resume variants to PDF:
- **macOS**: `brew install --cask mactex-no-gui`
- **Ubuntu**: `sudo apt-get install texlive-latex-base texlive-latex-extra`

## Usage

### Web Interface
```bash
# Start the backend server
python main.py serve

# In another terminal, start the frontend (if using npm)
cd frontend && npm run dev

# Or just use the API at http://localhost:8000
```

### CLI Commands

```bash
# Generate all 3 resume variants
python main.py generate

# Tailor resume with GPT-5 analysis (V2)
python main.py tailor vacancies/google.txt

# Preview GPT-5 bullet rewrites
python main.py tailor vacancies/google.txt --preview

# Cluster vacancies into keyword groups
python main.py cluster-vacancies

# Analyze keywords from vacancies
python main.py analyze --top 30

# Start web server
python main.py serve --port 8000
```

## Project Structure

```
resume_matcher/
├── resume.tex              # Your original resume
├── Profile.pdf             # LinkedIn export (optional)
├── vacancies/              # Job description files
├── output/                 # Generated resume variants
├── src/
│   ├── llm_client.py       # OpenAI GPT-5 wrapper
│   ├── semantic_matcher.py # Embedding-based matching
│   ├── bullet_rewriter.py  # GPT-5 bullet optimization
│   ├── keyword_engine.py   # Keyword extraction
│   ├── resume_generator.py # LaTeX variant generator
│   ├── vacancy_clustering.py # Vacancy clustering pipeline
│   └── ...
├── deprecated/
│   ├── linkedin_scraper.py # Deprecated LinkedIn scraper
├── backend/
│   ├── api.py              # FastAPI endpoints
│   ├── schemas.py          # Pydantic models
│   └── services.py         # Business logic
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Main React component
│   │   ├── api.ts          # API client
│   │   └── components/     # UI components
│   └── ...
├── main.py                 # CLI entry point
└── requirements.txt
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Analyze job description with GPT-5 |
| `/api/save-vacancy` | POST | Save job description to database |
| `/api/vacancies` | GET | List saved vacancies |
| `/api/variants` | GET | List resume variants |
| `/api/resume/{variant}/pdf` | GET | Download PDF |
| `/api/resume/{variant}/tex` | GET | Download LaTeX |

## Cost Estimation (GPT-5)

Per analysis:
- Embeddings: ~$0.0001 (negligible)
- GPT-5 input (~2K tokens): ~$0.0025
- GPT-5 output (~1K tokens): ~$0.01
- **Total: ~$0.01-0.02 per analysis**

## Target Positions

Optimized for:
- ML Engineer
- Applied Scientist  
- Data Scientist
- AI Engineer

## Configuration

Set the OpenAI API key:
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

## Dependencies

### Python
- `openai` - GPT-5 API
- `fastapi`, `uvicorn` - Web backend
- `pdfplumber` - PDF extraction
- `spacy`, `scikit-learn` - NLP
- `click`, `tqdm` - CLI

### Frontend
- React 18
- TypeScript
- Vite

## License

MIT License
