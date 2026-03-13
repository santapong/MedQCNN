"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { fetchPrediction, type PredictionDetail } from "@/lib/api";
import QuantumBar from "@/components/QuantumBar";
import { CardSkeleton } from "@/components/Skeleton";

export default function PredictionDetailPage() {
  const params = useParams();
  const id = Number(params.id);
  const [data, setData] = useState<PredictionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const result = await fetchPrediction(id);
        setData(result);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to load prediction"
        );
      } finally {
        setLoading(false);
      }
    }
    if (!isNaN(id)) load();
  }, [id]);

  if (loading) {
    return (
      <main className="max-w-4xl mx-auto px-6 py-10 space-y-6">
        <CardSkeleton />
        <CardSkeleton />
      </main>
    );
  }

  if (error || !data) {
    return (
      <main className="max-w-4xl mx-auto px-6 py-10">
        <div className="rounded-xl p-6 bg-danger/10 border border-danger/30">
          <h2 className="text-lg font-semibold text-danger mb-2">
            Prediction Not Found
          </h2>
          <p className="text-sm text-danger/80">{error}</p>
          <Link
            href="/history"
            className="text-accent-light text-sm mt-3 inline-block"
          >
            Back to History
          </Link>
        </div>
      </main>
    );
  }

  const isBenign = data.label === "Benign";
  const date = data.created_at
    ? new Date(data.created_at).toLocaleString()
    : "Unknown";

  return (
    <main className="max-w-4xl mx-auto px-6 py-10">
      <div className="mb-6">
        <Link
          href="/history"
          className="text-sm text-accent-light hover:text-accent"
        >
          &larr; Back to History
        </Link>
      </div>

      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">
          Prediction #{data.id}
        </h1>
        <p className="text-muted mt-1 text-sm">
          {data.image_filename} &middot; {date}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Classification */}
        <div
          className={`rounded-xl p-5 border ${
            isBenign
              ? "bg-success/5 border-success/20"
              : "bg-danger/5 border-danger/20"
          }`}
        >
          <p className="text-xs uppercase tracking-wider text-muted mb-1">
            Classification
          </p>
          <p
            className={`text-3xl font-bold ${
              isBenign ? "text-success" : "text-danger"
            }`}
          >
            {data.label}
          </p>
          <p className="text-2xl font-bold font-mono mt-2">
            {(data.confidence * 100).toFixed(1)}%
          </p>

          {data.probabilities && (
            <div className="mt-4">
              <div className="flex justify-between text-xs text-muted mb-1">
                <span>
                  Benign ({(data.probabilities[0] * 100).toFixed(1)}%)
                </span>
                <span>
                  Malignant ({(data.probabilities[1] * 100).toFixed(1)}%)
                </span>
              </div>
              <div className="h-3 bg-white/10 rounded-full overflow-hidden flex">
                <div
                  className="bg-success/80 rounded-l-full"
                  style={{ width: `${data.probabilities[0] * 100}%` }}
                />
                <div
                  className="bg-danger/80 rounded-r-full"
                  style={{ width: `${data.probabilities[1] * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Metadata */}
        <div className="rounded-xl border border-card-border bg-card-bg p-5">
          <h3 className="text-sm font-semibold mb-4 text-muted uppercase tracking-wider">
            Metadata
          </h3>
          <Row label="Prediction ID">{data.id}</Row>
          <Row label="Image Hash">{data.image_hash || "—"}</Row>
          <Row label="Model Version">{data.model_version}</Row>
          <Row label="Qubits">{data.n_qubits}</Row>
          <Row label="Created">{date}</Row>
        </div>

        {/* Quantum values */}
        {data.quantum_expectation_values && (
          <div className="rounded-xl p-5 border border-card-border bg-card-bg md:col-span-2">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <span className="text-accent-light">&#x269B;</span>
              Quantum Expectation Values
              <span className="text-xs text-muted font-normal">
                per-qubit Pauli-Z
              </span>
            </h3>
            <div className="space-y-2">
              {data.quantum_expectation_values.map((val, i) => (
                <QuantumBar key={i} qubitIndex={i} value={val} />
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

function Row({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-card-border last:border-0">
      <span className="text-sm text-muted">{label}</span>
      <span className="text-sm font-medium font-mono">{children}</span>
    </div>
  );
}
