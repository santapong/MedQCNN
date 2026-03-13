"use client";

import { useEffect, useState } from "react";
import {
  fetchHealth,
  fetchModelInfo,
  type HealthResponse,
  type ModelInfoResponse,
} from "@/lib/api";

export default function ModelInfoPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [h, m] = await Promise.all([fetchHealth(), fetchModelInfo()]);
        setHealth(h);
        setInfo(m);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to connect to API"
        );
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <main className="max-w-4xl mx-auto px-6 py-10">
        <div className="flex items-center justify-center h-64">
          <div className="w-8 h-8 border-3 border-accent border-t-transparent rounded-full animate-spin" />
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="max-w-4xl mx-auto px-6 py-10">
        <div className="rounded-xl p-6 bg-danger/10 border border-danger/30">
          <h2 className="text-lg font-semibold text-danger mb-2">
            Cannot connect to API
          </h2>
          <p className="text-sm text-danger/80">{error}</p>
          <p className="text-xs text-muted mt-3">
            Make sure the backend is running:{" "}
            <code className="bg-white/10 px-2 py-0.5 rounded">
              uv run python scripts/serve.py
            </code>
          </p>
        </div>
      </main>
    );
  }

  return (
    <main className="max-w-4xl mx-auto px-6 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">
          Model Architecture
        </h1>
        <p className="text-muted mt-2">
          HybridQCNN quantum-classical pipeline details
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Service status */}
        {health && (
          <Card title="Service Status">
            <Row label="Status">
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-success animate-pulse" />
                <span className="text-success font-medium">
                  {health.status}
                </span>
              </span>
            </Row>
            <Row label="Service">{health.service}</Row>
            <Row label="Version">{health.version}</Row>
          </Card>
        )}

        {/* Quantum config */}
        {info && (
          <Card title="Quantum Configuration">
            <Row label="Qubits">{info.n_qubits}</Row>
            <Row label="Latent Dimension">{info.latent_dim}</Row>
            <Row label="Ansatz Layers">{info.n_ansatz_layers}</Row>
            <Row label="Classes">{info.n_classes}</Row>
            <Row label="Device">
              <code className="bg-white/10 px-2 py-0.5 rounded text-xs">
                {info.device}
              </code>
            </Row>
          </Card>
        )}

        {/* Parameters */}
        {info && (
          <Card title="Trainable Parameters" className="md:col-span-2">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {Object.entries(info.trainable_params).map(([key, value]) => (
                <div key={key} className="text-center">
                  <p className="text-2xl font-bold font-mono">
                    {value.toLocaleString()}
                  </p>
                  <p className="text-xs text-muted capitalize mt-1">{key}</p>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* Architecture diagram */}
        <Card title="Pipeline Architecture" className="md:col-span-2">
          <div className="flex flex-wrap items-center justify-center gap-3 py-4">
            <PipelineNode
              label="Medical Image"
              detail="224 x 224 x 1"
              color="muted"
            />
            <PipelineArrow />
            <PipelineNode
              label="ResNet-18"
              detail="Frozen backbone"
              color="foreground"
            />
            <PipelineArrow />
            <PipelineNode
              label="Projector"
              detail={`FC -> R^${info?.latent_dim ?? 256}`}
              color="foreground"
            />
            <PipelineArrow />
            <PipelineNode
              label="Quantum Circuit"
              detail={`${info?.n_qubits ?? 8}q, ${info?.n_ansatz_layers ?? 4} layers`}
              color="accent-light"
              highlight
            />
            <PipelineArrow />
            <PipelineNode
              label="Classifier"
              detail={`${info?.n_classes ?? 2} classes`}
              color="foreground"
            />
          </div>
        </Card>
      </div>
    </main>
  );
}

function Card({
  title,
  children,
  className = "",
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl border border-card-border bg-card-bg p-5 ${className}`}
    >
      <h3 className="text-sm font-semibold mb-4 text-muted uppercase tracking-wider">
        {title}
      </h3>
      {children}
    </div>
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
      <span className="text-sm font-medium">{children}</span>
    </div>
  );
}

function PipelineNode({
  label,
  detail,
  color,
  highlight,
}: {
  label: string;
  detail: string;
  color: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={`text-center px-4 py-3 rounded-lg border transition-all ${
        highlight
          ? "bg-accent/15 border-accent/40"
          : "bg-white/5 border-card-border"
      }`}
    >
      <p className={`text-sm font-semibold text-${color}`}>{label}</p>
      <p className="text-xs text-muted mt-0.5">{detail}</p>
    </div>
  );
}

function PipelineArrow() {
  return <span className="text-muted/40 text-lg">&rarr;</span>;
}
