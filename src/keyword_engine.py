"""
Keyword Extraction and Clustering Engine.

Extracts keywords from job descriptions and clusters them into
categories for resume variant generation.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
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
    "mlops": [
        "mlops", "ml ops", "mlflow", "kubeflow", "model registry", "model monitoring",
        "feature store", "experiment tracking", "model deployment", "model serving",
        "ci/cd", "cicd", "continuous integration", "continuous deployment",
        "docker", "kubernetes", "k8s", "containerization", "helm",
    ],
    "nlp_llm": [
        "nlp", "natural language processing", "llm", "large language model",
        "langchain", "langgraph", "transformer", "bert", "gpt", "chatgpt",
        "openai", "huggingface", "hugging face", "rag", "retrieval augmented",
        "prompt engineering", "fine-tuning", "embeddings", "vector database",
        "pinecone", "faiss", "chromadb", "text generation", "sentiment analysis",
        "named entity recognition", "ner", "tokenization",
    ],
    "cloud_aws": [
        "aws", "amazon web services", "sagemaker", "ec2", "s3", "lambda",
        "bedrock", "step functions", "cloudwatch", "iam", "vpc",
        "azure", "microsoft azure", "databricks", "gcp", "google cloud",
        "bigquery", "cloud functions", "cloud run", "vertex ai",
        "cloud infrastructure", "terraform", "cloudformation",
    ],
    "data_engineering": [
        "spark", "pyspark", "apache spark", "hadoop", "hive", "emr",
        "airflow", "apache airflow", "dag", "etl", "elt", "data pipeline",
        "kafka", "apache kafka", "streaming", "batch processing",
        "data warehouse", "data lake", "snowflake", "redshift",
        "clickhouse", "postgresql", "mysql", "sql", "nosql", "mongodb",
        "data quality", "data governance", "dbt",
    ],
    "classical_ml": [
        "machine learning", "scikit-learn", "sklearn", "xgboost", "catboost",
        "lightgbm", "lgbm", "gradient boosting", "random forest",
        "logistic regression", "linear regression", "classification",
        "regression", "clustering", "feature engineering", "feature selection",
        "cross-validation", "hyperparameter tuning", "grid search",
        "model evaluation", "a/b testing", "ab testing", "experimentation",
        "churn prediction", "recommendation", "forecasting", "time series",
        "prophet", "arima", "anomaly detection",
    ],
}

# * Flatten taxonomy for quick lookup
ALL_TECH_KEYWORDS = set()
for keywords in TECH_TAXONOMY.values():
    ALL_TECH_KEYWORDS.update(keywords)


def extract_keywords_from_text(text: str, top_n: int = 50) -> list[tuple[str, float]]:
    """
    Extract keywords from text using TF-IDF and NLP.

    Args:
        text: Input text to extract keywords from.
        top_n: Number of top keywords to return.

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

    # * Return top N keywords
    return keywords.most_common(top_n)


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
    clustered["other"] = []

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
            clustered["other"].append(keyword)

    return clustered


def analyze_vacancies(
    vacancies_dir: str | Path,
    top_n: int = 100,
) -> dict:
    """
    Analyze all vacancy files and extract keyword statistics.

    Args:
        vacancies_dir: Path to directory containing vacancy files.
        top_n: Number of top keywords to return.

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
        doc_keywords = extract_keywords_from_text(content, top_n=50)
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


def get_category_keywords(category: str) -> list[str]:
    """
    Get predefined keywords for a category.

    Args:
        category: Category name (mlops, nlp_llm, cloud_aws, data_engineering, classical_ml).

    Returns:
        List of keywords for the category.
    """
    return TECH_TAXONOMY.get(category, [])


def score_text_against_category(text: str, category: str) -> float:
    """
    Score how well a text matches a keyword category.

    Args:
        text: Text to score.
        category: Category to match against.

    Returns:
        Score between 0 and 1 indicating match strength.
    """
    if category not in TECH_TAXONOMY:
        return 0.0

    category_keywords = TECH_TAXONOMY[category]
    text_lower = text.lower()

    matches = 0
    total_weight = 0

    for keyword in category_keywords:
        weight = len(keyword.split())  # * Multi-word keywords worth more
        total_weight += weight

        if keyword in text_lower:
            matches += weight

    if total_weight == 0:
        return 0.0

    return matches / total_weight


def match_job_to_categories(job_text: str) -> dict[str, float]:
    """
    Match a job description to all categories.

    Args:
        job_text: Job description text.

    Returns:
        Dictionary mapping category name to match score.
    """
    scores = {}

    for category in TECH_TAXONOMY:
        scores[category] = score_text_against_category(job_text, category)

    # * Normalize scores
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    return scores


def get_resume_themes() -> dict[str, dict]:
    """
    Get predefined resume themes for the 5 versions.

    Returns:
        Dictionary mapping theme name to theme configuration.
    """
    return {
        "mlops": {
            "name": "MLOps & Platform Engineering",
            "primary_category": "mlops",
            "secondary_categories": ["cloud_aws", "data_engineering"],
            "skills_priority": [
                "Docker", "Kubernetes", "MLflow", "CI/CD", "model registry",
                "model monitoring", "AWS Sagemaker", "Airflow",
            ],
            "experience_keywords": [
                "pipeline", "deployment", "production", "infrastructure",
                "monitoring", "scalable", "reliable", "automated",
            ],
        },
        "nlp_llm": {
            "name": "NLP & LLM Engineering",
            "primary_category": "nlp_llm",
            "secondary_categories": ["classical_ml", "cloud_aws"],
            "skills_priority": [
                "LangChain", "LangGraph", "LLM", "NLP", "Transformers",
                "PyTorch", "Hugging Face", "RAG", "embeddings",
            ],
            "experience_keywords": [
                "LLM", "language model", "NLP", "chatbot", "text",
                "prompt", "generation", "transformer",
            ],
        },
        "cloud_aws": {
            "name": "Cloud & AWS Infrastructure",
            "primary_category": "cloud_aws",
            "secondary_categories": ["mlops", "data_engineering"],
            "skills_priority": [
                "AWS", "Sagemaker", "EC2", "S3", "Bedrock", "Lambda",
                "Docker", "Terraform", "cloud infrastructure",
            ],
            "experience_keywords": [
                "AWS", "cloud", "Sagemaker", "infrastructure", "scalable",
                "serverless", "migration", "deployment",
            ],
        },
        "data_engineering": {
            "name": "Data Engineering & Pipelines",
            "primary_category": "data_engineering",
            "secondary_categories": ["classical_ml", "cloud_aws"],
            "skills_priority": [
                "Apache Airflow", "Apache Kafka", "Spark", "SQL",
                "ClickHouse", "ETL", "data pipeline", "Snowflake",
            ],
            "experience_keywords": [
                "pipeline", "data", "ETL", "processing", "Airflow",
                "Kafka", "SQL", "warehouse", "quality",
            ],
        },
        "classical_ml": {
            "name": "Classical ML & Analytics",
            "primary_category": "classical_ml",
            "secondary_categories": ["data_engineering", "mlops"],
            "skills_priority": [
                "XGBoost", "CatBoost", "LightGBM", "scikit-learn",
                "feature engineering", "A/B testing", "Prophet", "forecasting",
            ],
            "experience_keywords": [
                "model", "prediction", "classification", "regression",
                "feature", "performance", "evaluation", "optimization",
            ],
        },
    }


def find_best_theme_for_job(job_text: str) -> tuple[str, float]:
    """
    Find the best resume theme for a job description.

    Args:
        job_text: Job description text.

    Returns:
        Tuple of (theme_name, confidence_score).
    """
    category_scores = match_job_to_categories(job_text)
    themes = get_resume_themes()

    theme_scores = {}

    for theme_name, theme_config in themes.items():
        primary = theme_config["primary_category"]
        secondary = theme_config["secondary_categories"]

        # * Weighted score: primary category counts more
        score = category_scores.get(primary, 0) * 0.6

        for sec_cat in secondary:
            score += category_scores.get(sec_cat, 0) * 0.2

        theme_scores[theme_name] = score

    # * Find best theme
    best_theme = max(theme_scores, key=theme_scores.get)
    best_score = theme_scores[best_theme]

    return best_theme, best_score

