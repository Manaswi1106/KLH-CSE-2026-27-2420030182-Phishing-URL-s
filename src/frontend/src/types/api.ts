export interface DomainVerification {
  status: string;
  hostname: string;
  registered_domain: string;
  trusted: boolean;
}

export interface ShapFeature {
  feature: string;
  impact: number;
}

export interface WebpageEvidence {
  suspicious_keyword_count: number;
  html_length: number;
  text_length: number;
  num_links: number;
  external_links: number;
  external_link_ratio: number;
  form_count: number;
  password_field_count: number;
  has_login_form: number;
  external_form_count: number;
  external_form_ratio: number;
}

export interface PredictionResponse {
  prediction: string;
  confidence: number;
  final_decision: string;
  decision_reason: string;
  access_status: string;
  feature_mode: string;
  model: string;

  // NEW
  shap_explanation: ShapFeature[];

  domain_verification: DomainVerification;
  evidence: WebpageEvidence;
  explanation: string;
}

export interface PredictionRequest {
  url: string;
}

export interface HistoryItem {
  id: string;
  url: string;
  final_decision: string;
  prediction: string;
  confidence: number;
  timestamp: string;
  result: PredictionResponse;
}

export type DecisionType = "Benign" | "Phishing" | "Suspicious" | string;