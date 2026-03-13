"use client";

import { useState } from "react";
import ImageUploader from "@/components/ImageUploader";
import DiagnosisResult from "@/components/DiagnosisResult";
import { fileToBase64, predict, type PredictionResponse } from "@/lib/api";

export default function DiagnosePage() {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleImageSelected(file: File, previewUrl: string) {
    setPreview(previewUrl);
    setFileName(file.name);
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      const base64 = await fileToBase64(file);
      const prediction = await predict(base64);
      setResult(prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setPreview(null);
    setFileName("");
    setResult(null);
    setError(null);
  }

  return (
    <main className="max-w-6xl mx-auto px-6 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">
          Quantum Diagnostic Analysis
        </h1>
        <p className="text-muted mt-2">
          Upload a medical image for hybrid quantum-classical inference
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left: Upload / Preview */}
        <div className="space-y-4">
          {!preview ? (
            <ImageUploader
              onImageSelected={handleImageSelected}
              disabled={loading}
            />
          ) : (
            <div className="rounded-xl border border-card-border bg-card-bg overflow-hidden">
              <div className="relative aspect-square bg-black/30 flex items-center justify-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={preview}
                  alt="Uploaded medical image"
                  className="max-w-full max-h-full object-contain"
                />
                {loading && (
                  <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                    <div className="flex flex-col items-center gap-3">
                      <div className="w-10 h-10 border-3 border-accent border-t-transparent rounded-full animate-spin" />
                      <p className="text-sm text-accent-light">
                        Running quantum inference...
                      </p>
                    </div>
                  </div>
                )}
              </div>
              <div className="px-4 py-3 flex items-center justify-between border-t border-card-border">
                <span className="text-sm text-muted truncate max-w-[60%]">
                  {fileName}
                </span>
                <button
                  onClick={handleReset}
                  className="text-xs text-accent-light hover:text-accent transition-colors px-3 py-1 rounded-md hover:bg-white/5"
                >
                  Upload new
                </button>
              </div>
            </div>
          )}

          {/* Pipeline diagram */}
          <div className="rounded-xl border border-card-border bg-card-bg p-4">
            <h3 className="text-xs font-semibold text-muted uppercase tracking-wider mb-3">
              Pipeline
            </h3>
            <div className="flex items-center gap-2 text-xs font-mono text-muted overflow-x-auto">
              <Step label="Image" active={!!preview} />
              <Arrow />
              <Step label="ResNet-18" active={loading || !!result} />
              <Arrow />
              <Step label="Projector" active={loading || !!result} />
              <Arrow />
              <Step label="Quantum" active={loading || !!result} highlight />
              <Arrow />
              <Step label="Classify" active={!!result} />
            </div>
          </div>
        </div>

        {/* Right: Results */}
        <div>
          {error && (
            <div className="rounded-xl p-4 bg-danger/10 border border-danger/30 text-danger text-sm">
              <strong>Error:</strong> {error}
            </div>
          )}

          {result && <DiagnosisResult result={result} />}

          {!result && !error && !loading && (
            <div className="rounded-xl border border-card-border bg-card-bg h-64 flex items-center justify-center">
              <p className="text-muted text-sm">
                Upload an image to begin analysis
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

function Step({
  label,
  active,
  highlight,
}: {
  label: string;
  active: boolean;
  highlight?: boolean;
}) {
  return (
    <span
      className={`px-2.5 py-1.5 rounded-md border whitespace-nowrap transition-all duration-300 ${
        active
          ? highlight
            ? "bg-accent/20 border-accent/40 text-accent-light"
            : "bg-white/10 border-white/20 text-foreground"
          : "border-card-border text-muted/50"
      }`}
    >
      {label}
    </span>
  );
}

function Arrow() {
  return <span className="text-muted/40 shrink-0">&rarr;</span>;
}
