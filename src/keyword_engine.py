"""
Keyword Extraction and Clustering Engine.

Extracts keywords from job descriptions and clusters them into
categories for resume variant generation.
"""

import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_extraction import load_vacancy_files

# * Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("! spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None


# * Predefined technology taxonomy for better keyword recognition
TECH_TAXONOMY = {
    "research_ml": [
        "deep learning", "neural network", "neural networks", "pytorch", "tensorflow",
        "statistical analysis", "statistical", "optimization", "research", "experiments",
        "xgboost", "catboost", "model performance", "algorithms", "classification",
        "regression", "forecasting", "prophet", "bayesian", "time series",
        "feature selection", "model evaluation", "hyperparameter tuning",
    ],
    "applied_production": [
        "production", "pipeline", "deployment", "mlops", "docker", "kubernetes",
        "ci/cd", "cicd", "airflow", "mlflow", "sagemaker", "a/b testing",
        "ab testing", "monitoring", "feature engineering", "sql", "clickhouse",
        "infrastructure", "scaling", "automation", "retraining",
        "data pipeline", "containerization", "orchestration",
    ],
    "genai_llm": [
        "llm", "langchain", "langgraph", "rag", "embeddings", "prompt engineering",
        "gpt", "vector database", "agents", "fastapi", "rest api", "retrieval",
        "generation", "telegram bot", "automated", "chatbot", "openai",
        "huggingface", "transformer",
    ],
}

# * Flatten taxonomy for quick lookup
ALL_TECH_KEYWORDS = set()
for keywords in TECH_TAXONOMY.values():
    ALL_TECH_KEYWORDS.update(keywords)

DEFAULT_VACANCIES_DIR = Path(__file__).resolve().parents[1] / "vacancies"
VACANCIES_DIR = Path(os.getenv("VACANCIES_DIR", str(DEFAULT_VACANCIES_DIR)))
USE_VACANCY_KEYWORD_BASE = os.getenv("USE_VACANCY_KEYWORD_BASE", "true").lower() == "true"
VACANCY_BASE_TOP_N = int(os.getenv("VACANCY_BASE_TOP_N", "200"))
VACANCY_BASE_MAX_BOOST = float(os.getenv("VACANCY_BASE_MAX_BOOST", "5.0"))

_VACANCY_BASE_CACHE = None
_VACANCY_BASE_SIGNATURE = None


def _vacancy_files_signature(vacancies_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """Build a lightweight signature for vacancy files to detect changes."""
    if not vacancies_dir.exists():
        return ()

    signature = []
    for file_path in sorted(vacancies_dir.glob("*.txt")):
        try:
            stat = file_path.stat()
        except OSError:
            continue
        signature.append((file_path.name, stat.st_mtime_ns, stat.st_size))

    return tuple(signature)


def _get_vacancy_keyword_base(
    vacancies_dir: Optional[str | Path] = None,
) -> dict[str, float]:
    """Load or compute the aggregated vacancy keyword base."""
    if not USE_VACANCY_KEYWORD_BASE:
        return {}

    base_dir = Path(vacancies_dir) if vacancies_dir else VACANCIES_DIR
    if not base_dir.exists():
        return {}

    signature = _vacancy_files_signature(base_dir)
    global _VACANCY_BASE_CACHE, _VACANCY_BASE_SIGNATURE
    if signature == _VACANCY_BASE_SIGNATURE and _VACANCY_BASE_CACHE is not None:
        return _VACANCY_BASE_CACHE

    try:
        report = analyze_vacancies(
            base_dir,
            top_n=VACANCY_BASE_TOP_N,
            use_vacancy_base=False,
        )
    except Exception:
        return {}

    keywords = report.get("keywords", [])
    base = {keyword.lower(): float(score) for keyword, score in keywords}

    _VACANCY_BASE_CACHE = base
    _VACANCY_BASE_SIGNATURE = signature
    return base


def extract_keywords_from_text(
    text: str,
    top_n: int = 50,
    use_vacancy_base: bool = True,
) -> list[tuple[str, float]]:
    """
    Extract keywords from text using TF-IDF and NLP.

    Args:
        text: Input text to extract keywords from.
        top_n: Number of top keywords to return.
        use_vacancy_base: Whether to boost keywords using saved vacancies.

    Returns:
        List of (keyword, score) tuples sorted by score descending.
    """
    if not text.strip():
        return []

    keywords = Counter()

    # * 1. Extract using spaCy NER and noun phrases
    if nlp is not None:
        doc = nlp(text.lower())

        # * Extract noun phrases
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if len(chunk_text) > 2 and not chunk_text.isdigit():
                keywords[chunk_text] += 1

        # * Extract named entities (ORG, PRODUCT, etc.)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "GPE", "WORK_OF_ART"):
                keywords[ent.text.lower()] += 1

    # * 2. Match against technology taxonomy
    text_lower = text.lower()
    for keyword in ALL_TECH_KEYWORDS:
        if keyword in text_lower:
            # * Count occurrences
            count = len(re.findall(re.escape(keyword), text_lower))
            keywords[keyword] += count * 2  # * Boost taxonomy matches

    # * 3. Extract technical terms using patterns
    technical_patterns = [
        r"\b[A-Z][a-zA-Z]*(?:AI|ML|DB|API)\b",  # * Acronym suffixes
        r"\b(?:Python|Java|Scala|Go|R|SQL)\b",  # * Programming languages
        r"\b\d+\+?\s*years?\b",  # * Experience requirements
    ]

    for pattern in technical_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            keywords[match.lower()] += 1

    # * 4. Boost keywords that are common across saved vacancies
    if use_vacancy_base:
        vacancy_base = _get_vacancy_keyword_base()
        if vacancy_base:
            for keyword, base_score in vacancy_base.items():
                if keyword in text_lower:
                    count = text_lower.count(keyword)
                    if count:
                        boost = min(
                            VACANCY_BASE_MAX_BOOST,
                            1.0 + math.log1p(base_score),
                        )
                        keywords[keyword] += count * boost

    # * Return top N keywords
    return keywords.most_common(top_n)


def get_technology_patterns() -> list[re.Pattern]:
    """Regex patterns for detecting specific technologies."""
    return [
        re.compile(
            r"\b(?:pytorch|tensorflow|catboost|xgboost|lightgbm|scikit[- ]learn|hugging\s*face"
            r"|langchain|langgraph|fastapi|mlflow|airflow|spark|kafka)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:aws|sagemaker|ec2|s3|lambda|gcp|google\s*cloud|bigquery|azure|databricks)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:docker|kubernetes|k8s|terraform|helm|ci\s*/\s*cd)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:python|sql|go|golang|rust|java|scala|r)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:vector\s+db|vector\s+database|pinecone|faiss|chromadb|weaviate)\b",
            re.IGNORECASE,
        ),
    ]


def extract_keywords_hybrid(text: str) -> list[tuple[str, float]]:
    """Combine TF-IDF + taxonomy + patterns with weighted scoring."""
    if not text.strip():
        return []

    keywords = Counter()

    # * Base extraction (NER + taxonomy + patterns)
    base_keywords = extract_keywords_from_text(
        text,
        top_n=80,
        use_vacancy_base=False,
    )
    for keyword, score in base_keywords:
        normalized = re.sub(r"\s+", " ", keyword.strip().lower())
        keywords[normalized] += float(score)

    # * Lightweight TF-IDF for the document
    tfidf_keywords = extract_keywords_tfidf([text], top_n_per_doc=30).get(0, [])
    for keyword, score in tfidf_keywords:
        normalized = re.sub(r"\s+", " ", keyword.strip().lower())
        keywords[normalized] += float(score) * 8

    # * Explicit technology pattern matches
    for pattern in get_technology_patterns():
        for match in pattern.findall(text):
            normalized = re.sub(r"\s+", " ", match.strip().lower())
            keywords[normalized] += 3

    return keywords.most_common(60)


def extract_keywords_tfidf(
    documents: list[str],
    top_n_per_doc: int = 30,
) -> dict[int, list[tuple[str, float]]]:
    """
    Extract keywords from multiple documents using TF-IDF.

    Args:
        documents: List of text documents.
        top_n_per_doc: Number of top keywords per document.

    Returns:
        Dictionary mapping document index to list of (keyword, score) tuples.
    """
    if not documents:
        return {}

    # * Configure TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 3),  # * Unigrams, bigrams, trigrams
        min_df=1,
        max_df=0.95,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.-]*\b",  # * Allow tech terms
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return {}

    feature_names = vectorizer.get_feature_names_out()

    result = {}

    for doc_idx in range(len(documents)):
        # * Get TF-IDF scores for this document
        doc_vector = tfidf_matrix[doc_idx].toarray().flatten()

        # * Get top N indices
        top_indices = doc_vector.argsort()[-top_n_per_doc:][::-1]

        # * Build keyword list
        keywords = []
        for idx in top_indices:
            score = doc_vector[idx]
            if score > 0:
                keywords.append((feature_names[idx], float(score)))

        result[doc_idx] = keywords

    return result


def cluster_keywords(
    keywords: list[str],
) -> dict[str, list[str]]:
    """
    Cluster keywords into predefined categories.

    Args:
        keywords: List of keywords to cluster.

    Returns:
        Dictionary mapping category name to list of keywords.
    """
    clustered = {category: [] for category in TECH_TAXONOMY}
    clustered["general"] = []

    for keyword in keywords:
        keyword_lower = keyword.lower()
        assigned = False

        for category, category_keywords in TECH_TAXONOMY.items():
            # * Check if keyword matches any category keyword
            for cat_kw in category_keywords:
                if cat_kw in keyword_lower or keyword_lower in cat_kw:
                    if keyword not in clustered[category]:
                        clustered[category].append(keyword)
                    assigned = True
                    break

            if assigned:
                break

        if not assigned:
            clustered["general"].append(keyword)

    return clustered


def analyze_vacancies(
    vacancies_dir: str | Path,
    top_n: int = 100,
    use_vacancy_base: bool = False,
) -> dict:
    """
    Analyze all vacancy files and extract keyword statistics.

    Args:
        vacancies_dir: Path to directory containing vacancy files.
        top_n: Number of top keywords to return.
        use_vacancy_base: Whether to include the vacancy base during extraction.

    Returns:
        Dictionary containing:
        - keywords: List of (keyword, frequency) tuples
        - by_category: Keywords grouped by category
        - by_vacancy: Keywords per vacancy file
    """
    vacancies = load_vacancy_files(vacancies_dir)

    if not vacancies:
        return {"keywords": [], "by_category": {}, "by_vacancy": {}}

    # * Aggregate all keywords
    all_keywords = Counter()
    by_vacancy = {}

    for name, content in vacancies.items():
        doc_keywords = extract_keywords_from_text(
            content,
            top_n=50,
            use_vacancy_base=use_vacancy_base,
        )
        by_vacancy[name] = doc_keywords

        for keyword, score in doc_keywords:
            all_keywords[keyword] += score

    # * Get top keywords overall
    top_keywords = all_keywords.most_common(top_n)

    # * Cluster keywords
    keyword_list = [kw for kw, _ in top_keywords]
    by_category = cluster_keywords(keyword_list)

    # * Also run TF-IDF analysis
    documents = list(vacancies.values())
    tfidf_keywords = extract_keywords_tfidf(documents, top_n_per_doc=30)

    # * Merge TF-IDF keywords into results
    for doc_idx, kw_list in tfidf_keywords.items():
        for keyword, score in kw_list:
            all_keywords[keyword] += score * 10  # * Boost TF-IDF scores

    # * Recompute top keywords with TF-IDF boost
    top_keywords = all_keywords.most_common(top_n)

    return {
        "keywords": top_keywords,
        "by_category": by_category,
        "by_vacancy": by_vacancy,
    }


def serialize_keyword_report(report: dict) -> dict:
    """
    Convert keyword analysis output into JSON-serializable data.

    Args:
        report: Result from analyze_vacancies().

    Returns:
        Dictionary with keywords and scores as plain dicts/lists.
    """
    keywords = [
        {"keyword": keyword, "score": float(score)}
        for keyword, score in report.get("keywords", [])
    ]

    by_category = {
        category: keywords_list
        for category, keywords_list in report.get("by_category", {}).items()
    }

    by_vacancy = {}
    for name, keywords_list in report.get("by_vacancy", {}).items():
        by_vacancy[name] = [
            {"keyword": keyword, "score": float(score)}
            for keyword, score in keywords_list
        ]

    return {
        "keywords": keywords,
        "by_category": by_category,
        "by_vacancy": by_vacancy,
    }
