const API_BASE = "/api";

// ── Types ───────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface ModelInfoResponse {
  n_qubits: number;
  latent_dim: number;
  n_ansatz_layers: number;
  n_classes: number;
  trainable_params: Record<string, number>;
  device: string;
}

export interface PredictionResponse {
  prediction: number;
  label: string;
  confidence: number;
  probabilities: number[] | null;
  quantum_expectation_values: number[] | null;
}

export interface BatchPredictionResponse {
  results: PredictionResponse[];
  total: number;
  successful: number;
  failed: number;
  summary: {
    benign: number;
    malignant: number;
    avg_confidence: number;
  };
  errors: string[] | null;
}

export interface PredictionDetail {
  id: number;
  image_filename: string;
  image_hash: string | null;
  prediction: number;
  label: string;
  confidence: number;
  probabilities: number[] | null;
  quantum_expectation_values: number[] | null;
  model_version: string;
  n_qubits: number;
  created_at: string | null;
}

export interface PaginatedPredictions {
  items: PredictionDetail[];
  total: number;
  offset: number;
  limit: number;
}

export interface TrainingRun {
  id: number;
  dataset: string;
  n_qubits: number;
  n_layers: number;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  final_train_acc: number | null;
  final_val_acc: number | null;
  final_test_acc: number | null;
  auc_roc: number | null;
  f1: number | null;
  duration_seconds: number | null;
  checkpoint_path: string | null;
  history: {
    train_loss: number[];
    val_loss: number[];
    train_acc: number[];
    val_acc: number[];
  } | null;
  created_at: string | null;
  benchmarks: BenchmarkMetric[];
}

export interface TrainingRunListResponse {
  items: TrainingRun[];
  total: number;
  offset: number;
  limit: number;
}

export interface BenchmarkMetric {
  id: number;
  training_run_id: number;
  metric_name: string;
  metric_value: number;
  created_at: string | null;
}

export interface BenchmarkListResponse {
  items: BenchmarkMetric[];
  total: number;
  offset: number;
  limit: number;
}

// ── API Functions ───────────────────────────────────────

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE}/info`);
  if (!res.ok) throw new Error(`Model info failed: ${res.status}`);
  return res.json();
}

export async function predict(
  imageBase64: string
): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_base64: imageBase64,
      return_probabilities: true,
    }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Prediction failed: ${res.status}`);
  }
  return res.json();
}

export async function predictBatch(
  images: string[]
): Promise<BatchPredictionResponse> {
  const res = await fetch(`${API_BASE}/predict/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ images }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Batch prediction failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchPredictions(params?: {
  offset?: number;
  limit?: number;
  label?: string;
  confidence_min?: number;
  filename?: string;
}): Promise<PaginatedPredictions> {
  const searchParams = new URLSearchParams();
  if (params?.offset) searchParams.set("offset", String(params.offset));
  if (params?.limit) searchParams.set("limit", String(params.limit));
  if (params?.label) searchParams.set("label", params.label);
  if (params?.confidence_min)
    searchParams.set("confidence_min", String(params.confidence_min));
  if (params?.filename) searchParams.set("filename", params.filename);

  const qs = searchParams.toString();
  const res = await fetch(`${API_BASE}/predictions${qs ? `?${qs}` : ""}`);
  if (!res.ok) throw new Error(`Failed to fetch predictions: ${res.status}`);
  return res.json();
}

export async function fetchPrediction(id: number): Promise<PredictionDetail> {
  const res = await fetch(`${API_BASE}/predictions/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch prediction: ${res.status}`);
  return res.json();
}

export async function fetchTrainingRuns(params?: {
  offset?: number;
  limit?: number;
}): Promise<TrainingRunListResponse> {
  const searchParams = new URLSearchParams();
  if (params?.offset) searchParams.set("offset", String(params.offset));
  if (params?.limit) searchParams.set("limit", String(params.limit));

  const qs = searchParams.toString();
  const res = await fetch(`${API_BASE}/training-runs${qs ? `?${qs}` : ""}`);
  if (!res.ok) throw new Error(`Failed to fetch training runs: ${res.status}`);
  return res.json();
}

export async function fetchTrainingRun(id: number): Promise<TrainingRun> {
  const res = await fetch(`${API_BASE}/training-runs/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch training run: ${res.status}`);
  return res.json();
}

export async function fetchBenchmarks(params?: {
  training_run_id?: number;
  offset?: number;
  limit?: number;
}): Promise<BenchmarkListResponse> {
  const searchParams = new URLSearchParams();
  if (params?.training_run_id)
    searchParams.set("training_run_id", String(params.training_run_id));
  if (params?.offset) searchParams.set("offset", String(params.offset));
  if (params?.limit) searchParams.set("limit", String(params.limit));

  const qs = searchParams.toString();
  const res = await fetch(`${API_BASE}/benchmarks${qs ? `?${qs}` : ""}`);
  if (!res.ok) throw new Error(`Failed to fetch benchmarks: ${res.status}`);
  return res.json();
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Strip the data:...;base64, prefix
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
