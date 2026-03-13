"use client";

import { useCallback, useState } from "react";
import {
  fileToBase64,
  predictBatch,
  type BatchPredictionResponse,
  type PredictionResponse,
} from "@/lib/api";
import QuantumBar from "@/components/QuantumBar";

export default function BatchPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<BatchPredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files).filter((f) =>
      f.type.startsWith("image/")
    );
    setFiles((prev) => [...prev, ...dropped]);
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = Array.from(e.target.files || []);
      setFiles((prev) => [...prev, ...selected]);
    },
    []
  );

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }

  async function handleSubmit() {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setExpandedIdx(null);

    try {
      const base64Images = await Promise.all(files.map(fileToBase64));
      const response = await predictBatch(base64Images);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Batch prediction failed");
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setFiles([]);
    setResult(null);
    setError(null);
    setExpandedIdx(null);
  }

  const totalSize = files.reduce((sum, f) => sum + f.size, 0);
  const maxSize = 250 * 1024 * 1024;

  return (
    <main className="max-w-6xl mx-auto px-6 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Batch Diagnosis</h1>
        <p className="text-muted mt-2">
          Upload multiple medical images for batch quantum-classical inference
        </p>
      </div>

      {!result ? (
        <div className="space-y-6">
          {/* Drop zone */}
          <label
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="flex flex-col items-center justify-center w-full h-48 rounded-xl border-2 border-dashed border-card-border hover:border-accent/50 cursor-pointer transition-all"
          >
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileChange}
              className="hidden"
              disabled={loading}
            />
            <svg
              className="w-10 h-10 text-muted mb-3"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <p className="text-sm text-muted">
              <span className="text-accent-light font-medium">
                Click to upload
              </span>{" "}
              or drag and drop multiple files
            </p>
            <p className="text-xs text-muted/70 mt-1">
              PNG, JPG, BMP, TIFF — up to 250 MB total, max 100 images
            </p>
          </label>

          {/* File list */}
          {files.length > 0 && (
            <div className="rounded-xl border border-card-border bg-card-bg overflow-hidden">
              <div className="px-4 py-3 border-b border-card-border flex items-center justify-between">
                <span className="text-sm font-medium">
                  {files.length} file{files.length !== 1 ? "s" : ""} selected
                </span>
                <span className="text-xs text-muted">
                  {(totalSize / 1024 / 1024).toFixed(1)} /{" "}
                  {(maxSize / 1024 / 1024).toFixed(0)} MB
                </span>
              </div>
              <div className="max-h-64 overflow-y-auto">
                {files.map((file, i) => (
                  <div
                    key={i}
                    className="px-4 py-2 border-b border-card-border last:border-0 flex items-center justify-between text-sm"
                  >
                    <span className="text-muted truncate max-w-[70%]">
                      {file.name}
                    </span>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-muted/70">
                        {(file.size / 1024).toFixed(0)} KB
                      </span>
                      <button
                        onClick={() => removeFile(i)}
                        className="text-danger/70 hover:text-danger text-xs"
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                ))}
              </div>
              <div className="px-4 py-3 border-t border-card-border flex gap-3">
                <button
                  onClick={handleSubmit}
                  disabled={loading || totalSize > maxSize}
                  className="px-4 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 transition-colors"
                >
                  {loading ? "Processing..." : "Run Batch Inference"}
                </button>
                <button
                  onClick={handleReset}
                  className="px-4 py-2 text-muted hover:text-foreground text-sm transition-colors"
                >
                  Clear All
                </button>
              </div>
            </div>
          )}

          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="flex flex-col items-center gap-3">
                <div className="w-10 h-10 border-3 border-accent border-t-transparent rounded-full animate-spin" />
                <p className="text-sm text-accent-light">
                  Running batch quantum inference...
                </p>
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-xl p-4 bg-danger/10 border border-danger/30 text-danger text-sm">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-6">
          {/* Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <SummaryCard
              label="Total"
              value={result.total}
              color="foreground"
            />
            <SummaryCard
              label="Benign"
              value={result.summary.benign}
              color="success"
            />
            <SummaryCard
              label="Malignant"
              value={result.summary.malignant}
              color="danger"
            />
            <SummaryCard
              label="Avg Confidence"
              value={`${(result.summary.avg_confidence * 100).toFixed(1)}%`}
              color="accent-light"
            />
          </div>

          {result.errors && result.errors.length > 0 && (
            <div className="rounded-xl p-4 bg-warning/10 border border-warning/30 text-sm">
              <p className="font-medium text-warning mb-1">
                {result.failed} image(s) failed:
              </p>
              {result.errors.map((err, i) => (
                <p key={i} className="text-warning/80 text-xs">
                  {err}
                </p>
              ))}
            </div>
          )}

          {/* Results table */}
          <div className="rounded-xl border border-card-border bg-card-bg overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-card-border text-left text-xs text-muted uppercase tracking-wider">
                  <th className="px-4 py-3">#</th>
                  <th className="px-4 py-3">File</th>
                  <th className="px-4 py-3">Label</th>
                  <th className="px-4 py-3">Confidence</th>
                  <th className="px-4 py-3">Details</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((r, i) => (
                  <ResultRow
                    key={i}
                    index={i}
                    result={r}
                    filename={files[i]?.name || `Image ${i}`}
                    expanded={expandedIdx === i}
                    onToggle={() =>
                      setExpandedIdx(expandedIdx === i ? null : i)
                    }
                  />
                ))}
              </tbody>
            </table>
          </div>

          <button
            onClick={handleReset}
            className="px-4 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent/90 transition-colors"
          >
            New Batch
          </button>
        </div>
      )}
    </main>
  );
}

function SummaryCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number | string;
  color: string;
}) {
  return (
    <div className="rounded-xl border border-card-border bg-card-bg p-4 text-center">
      <p className={`text-2xl font-bold font-mono text-${color}`}>{value}</p>
      <p className="text-xs text-muted mt-1">{label}</p>
    </div>
  );
}

function ResultRow({
  index,
  result,
  filename,
  expanded,
  onToggle,
}: {
  index: number;
  result: PredictionResponse;
  filename: string;
  expanded: boolean;
  onToggle: () => void;
}) {
  const isError = result.label === "error";
  const isBenign = result.label === "Benign";

  return (
    <>
      <tr
        className="border-b border-card-border hover:bg-white/[0.02] cursor-pointer transition-colors"
        onClick={onToggle}
      >
        <td className="px-4 py-3 text-muted">{index + 1}</td>
        <td className="px-4 py-3 truncate max-w-[200px]">{filename}</td>
        <td className="px-4 py-3">
          <span
            className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
              isError
                ? "bg-warning/10 text-warning"
                : isBenign
                  ? "bg-success/10 text-success"
                  : "bg-danger/10 text-danger"
            }`}
          >
            {result.label}
          </span>
        </td>
        <td className="px-4 py-3 font-mono">
          {isError ? "—" : `${(result.confidence * 100).toFixed(1)}%`}
        </td>
        <td className="px-4 py-3 text-accent-light text-xs">
          {isError ? "" : expanded ? "Hide" : "Show"}
        </td>
      </tr>
      {expanded && !isError && (
        <tr>
          <td colSpan={5} className="px-4 py-4 bg-card-bg/50">
            <div className="space-y-3 max-w-xl">
              {result.probabilities && (
                <div>
                  <p className="text-xs text-muted mb-1">Probabilities</p>
                  <div className="h-3 bg-white/10 rounded-full overflow-hidden flex">
                    <div
                      className="bg-success/80 rounded-l-full"
                      style={{ width: `${result.probabilities[0] * 100}%` }}
                    />
                    <div
                      className="bg-danger/80 rounded-r-full"
                      style={{ width: `${result.probabilities[1] * 100}%` }}
                    />
                  </div>
                </div>
              )}
              {result.quantum_expectation_values && (
                <div>
                  <p className="text-xs text-muted mb-1">
                    Quantum Expectation Values
                  </p>
                  <div className="space-y-1">
                    {result.quantum_expectation_values.map((val, qi) => (
                      <QuantumBar key={qi} qubitIndex={qi} value={val} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
