const API_BASE = "/api";

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
