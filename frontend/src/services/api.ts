import { PredictionRequest, PredictionResponse } from '../types/api';

const DEFAULT_API_URL = "/api/backend/predict";
const STORAGE_KEY = 'phishxpert_api_url';

export function getApiEndpoint(): string {
  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return saved;
  }
  return DEFAULT_API_URL;
}

export function setApiEndpoint(url: string): void {
  if (typeof window !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, url.trim());
  }
}

export function resetApiEndpoint(): void {
  if (typeof window !== 'undefined') {
    localStorage.removeItem(STORAGE_KEY);
  }
}

export interface ApiErrorDetails {
  message: string;
  isNetworkError: boolean;
  status?: number;
  urlAttempted: string;
}

export async function analyzeUrl(url: string, endpointOverride?: string): Promise<PredictionResponse> {
  const endpoint = endpointOverride || getApiEndpoint();

  const payload: PredictionRequest = {
    url: url.trim(),
  };

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let errorDetail = `Server responded with status ${response.status} (${response.statusText})`;
      try {
        const errJson = await response.json();
        if (errJson.detail) {
          errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        }
      } catch {
        // use default errorDetail
      }
      const error: ApiErrorDetails = {
        message: errorDetail,
        isNetworkError: false,
        status: response.status,
        urlAttempted: endpoint,
      };
      throw error;
    }

    const data: PredictionResponse = await response.json();
    return data;
  } catch (err: unknown) {
    if (err && typeof err === 'object' && 'isNetworkError' in err) {
      throw err;
    }

    const isMixedContent = typeof window !== 'undefined' &&
      window.location.protocol === 'https:' &&
      endpoint.startsWith('http://');

    let customMsg = 'Unable to connect to PhishXpert backend. Make sure FastAPI is running on port 8001.';
    if (isMixedContent) {
      customMsg += ' (Note: HTTPS preview cannot call insecure HTTP localhost due to browser Mixed Content policies. Use the Vite proxy /api/backend/predict or run locally with npm run dev).';
    }

    const netError: ApiErrorDetails = {
      message: customMsg,
      isNetworkError: true,
      urlAttempted: endpoint,
    };
    throw netError;
  }
}
