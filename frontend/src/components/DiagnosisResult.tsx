import type { PredictionResponse } from "@/lib/api";
import QuantumBar from "./QuantumBar";

interface DiagnosisResultProps {
  result: PredictionResponse;
}

export default function DiagnosisResult({ result }: DiagnosisResultProps) {
  const isBenign = result.label === "Benign";
  const confidencePct = (result.confidence * 100).toFixed(1);

  return (
    <div className="space-y-5 animate-in fade-in duration-500">
      {/* Prediction card */}
      <div
        className={`rounded-xl p-5 border ${
          isBenign
            ? "bg-success/5 border-success/20"
            : "bg-danger/5 border-danger/20"
        }`}
      >
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-wider text-muted mb-1">
              Classification
            </p>
            <p
              className={`text-2xl font-bold ${
                isBenign ? "text-success" : "text-danger"
              }`}
            >
              {result.label}
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs uppercase tracking-wider text-muted mb-1">
              Confidence
            </p>
            <p className="text-2xl font-bold font-mono">{confidencePct}%</p>
          </div>
        </div>

        {/* Probability bar */}
        {result.probabilities && (
          <div className="mt-4">
            <div className="flex justify-between text-xs text-muted mb-1">
              <span>Benign ({(result.probabilities[0] * 100).toFixed(1)}%)</span>
              <span>
                Malignant ({(result.probabilities[1] * 100).toFixed(1)}%)
              </span>
            </div>
            <div className="h-3 bg-white/10 rounded-full overflow-hidden flex">
              <div
                className="bg-success/80 transition-all duration-700 rounded-l-full"
                style={{ width: `${result.probabilities[0] * 100}%` }}
              />
              <div
                className="bg-danger/80 transition-all duration-700 rounded-r-full"
                style={{ width: `${result.probabilities[1] * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Quantum expectation values */}
      {result.quantum_expectation_values && (
        <div className="rounded-xl p-5 border border-card-border bg-card-bg">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <span className="text-accent-light">&#x269B;</span>
            Quantum Expectation Values
            <span className="text-xs text-muted font-normal">
              per-qubit Pauli-Z
            </span>
          </h3>
          <div className="space-y-2">
            {result.quantum_expectation_values.map((val, i) => (
              <QuantumBar key={i} qubitIndex={i} value={val} />
            ))}
          </div>
          <p className="text-xs text-muted/70 mt-3">
            Values near -1 correlate with malignancy; values near +1 indicate
            benign features.
          </p>
        </div>
      )}

      {/* Disclaimer */}
      <div className="rounded-lg p-3 bg-warning/5 border border-warning/20">
        <p className="text-xs text-warning/90">
          <strong>Disclaimer:</strong> This is an AI-assisted analysis using
          quantum-enhanced inference. It is NOT a clinical diagnosis. Consult a
          qualified radiologist or pathologist for clinical decision-making.
        </p>
      </div>
    </div>
  );
}
