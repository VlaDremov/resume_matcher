/**
 * API Client for Resume Matcher Backend
 */

const API_BASE = '/api';

export interface CategoryScore {
  category: string;
  score: number;
  display_name: string;
}

export interface RewrittenBullet {
  original: string;
  rewritten: string;
  keywords_added: string[];
  confidence: number;
}

export interface AnalyzeResponse {
  relevancy_score: number;
  best_variant: string;
  best_variant_display: string;
  category_scores: CategoryScore[];
  key_matches: string[];
  missing_keywords: string[];
  rewritten_bullets: RewrittenBullet[];
  reasoning: string;
  tex_path: string | null;
  pdf_path: string | null;
}

export interface VacancyInfo {
  filename: string;
  filepath: string;
  company: string | null;
  position: string | null;
  preview: string;
}

export interface SaveVacancyResponse {
  success: boolean;
  filepath: string;
  message: string;
}

export interface ResumeVariantInfo {
  name: string;
  display_name: string;
  description: string;
  tex_exists: boolean;
  pdf_exists: boolean;
  tex_path: string | null;
  pdf_path: string | null;
}

/**
 * Analyze a job description and find the best resume match.
 */
export async function analyzeJob(
  jobDescription: string,
  useSemantic: boolean = true,
  rewriteBullets: boolean = true
): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      job_description: jobDescription,
      use_semantic: useSemantic,
      rewrite_bullets: rewriteBullets,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Analysis failed');
  }

  return response.json();
}

/**
 * Save a vacancy to the database.
 */
export async function saveVacancy(
  jobDescription: string,
  filename: string,
  company?: string,
  position?: string
): Promise<SaveVacancyResponse> {
  const response = await fetch(`${API_BASE}/save-vacancy`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      job_description: jobDescription,
      filename,
      company,
      position,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to save vacancy');
  }

  return response.json();
}

/**
 * List all saved vacancies.
 */
export async function listVacancies(): Promise<VacancyInfo[]> {
  const response = await fetch(`${API_BASE}/vacancies`);

  if (!response.ok) {
    throw new Error('Failed to fetch vacancies');
  }

  const data = await response.json();
  return data.vacancies;
}

/**
 * List all resume variants.
 */
export async function listVariants(): Promise<ResumeVariantInfo[]> {
  const response = await fetch(`${API_BASE}/variants`);

  if (!response.ok) {
    throw new Error('Failed to fetch variants');
  }

  const data = await response.json();
  return data.variants;
}

/**
 * Get PDF URL for a variant.
 */
export function getPdfUrl(variantName: string): string {
  return `${API_BASE}/resume/${variantName}/pdf`;
}

/**
 * Get LaTeX URL for a variant.
 */
export function getTexUrl(variantName: string): string {
  return `${API_BASE}/resume/${variantName}/tex`;
}

