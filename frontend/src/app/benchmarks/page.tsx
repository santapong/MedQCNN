"use client";

import { useEffect, useState } from "react";
import {
  fetchBenchmarks,
  fetchTrainingRuns,
  type BenchmarkMetric,
  type TrainingRun,
} from "@/lib/api";
import { CardSkeleton } from "@/components/Skeleton";
import BenchmarkCharts from "@/components/BenchmarkCharts";

export default function BenchmarksPage() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkMetric[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [runsRes, benchRes] = await Promise.all([
          fetchTrainingRuns({ limit: 50 }),
          fetchBenchmarks({ limit: 500 }),
        ]);
        setRuns(runsRes.items);
        setBenchmarks(benchRes.items);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to load benchmarks"
        );
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Precomputed reference data for display even without DB entries
  const referenceData = {
    configs: [
      { label: "4-Qubit (Demo)", qubits: 4, totalParams: 8706, quantumParams: 32, memoryGb: 2.0, latencyMs: 70 },
      { label: "8-Qubit (Prod)", qubits: 8, totalParams: 132578, quantumParams: 64, memoryGb: 4.0, latencyMs: 220 },
    ],
    classicalBaseline: { label: "ResNet-18 (Full)", totalParams: 11309000, memoryGb: 6.0, latencyMs: 45 },
  };

  return (
    <main className="max-w-7xl mx-auto px-6 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Benchmarks</h1>
        <p className="text-muted mt-2">
          Parameter counts, memory usage, inference latency, and quantum vs
          classical comparison
        </p>
      </div>

      {error && (
        <div className="rounded-xl p-4 bg-danger/10 border border-danger/30 text-danger text-sm mb-6">
          {error}
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <CardSkeleton />
          <CardSkeleton />
          <CardSkeleton />
        </div>
      ) : (
        <div className="space-y-8">
          {/* Reference comparison cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {referenceData.configs.map((cfg) => (
              <div
                key={cfg.label}
                className="rounded-xl border border-card-border bg-card-bg p-5"
              >
                <h3 className="text-sm font-semibold text-accent-light mb-3">
                  {cfg.label}
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted">Total Trainable</span>
                    <span className="font-mono">
                      {cfg.totalParams.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">Quantum Params</span>
                    <span className="font-mono">{cfg.quantumParams}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">Memory (RAM)</span>
                    <span className="font-mono">{cfg.memoryGb} GB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">Inference Latency</span>
                    <span className="font-mono">~{cfg.latencyMs} ms</span>
                  </div>
                </div>
              </div>
            ))}
            <div className="rounded-xl border border-card-border bg-card-bg p-5">
              <h3 className="text-sm font-semibold text-warning mb-3">
                {referenceData.classicalBaseline.label}
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted">Total Trainable</span>
                  <span className="font-mono">
                    {referenceData.classicalBaseline.totalParams.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">Quantum Params</span>
                  <span className="font-mono text-muted/50">N/A</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">Memory (RAM)</span>
                  <span className="font-mono">
                    {referenceData.classicalBaseline.memoryGb} GB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">Inference Latency</span>
                  <span className="font-mono">
                    ~{referenceData.classicalBaseline.latencyMs} ms
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Charts */}
          <BenchmarkCharts
            runs={runs}
            benchmarks={benchmarks}
            referenceData={referenceData}
          />

          {/* Benchmark metrics from DB */}
          {benchmarks.length > 0 && (
            <div className="rounded-xl border border-card-border bg-card-bg overflow-hidden">
              <div className="px-4 py-3 border-b border-card-border">
                <h3 className="text-sm font-semibold text-muted uppercase tracking-wider">
                  Stored Benchmark Metrics
                </h3>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-card-border text-left text-xs text-muted uppercase tracking-wider">
                    <th className="px-4 py-3">Run ID</th>
                    <th className="px-4 py-3">Metric</th>
                    <th className="px-4 py-3">Value</th>
                    <th className="px-4 py-3">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {benchmarks.slice(0, 30).map((b) => (
                    <tr
                      key={b.id}
                      className="border-b border-card-border last:border-0"
                    >
                      <td className="px-4 py-3 font-mono text-muted">
                        {b.training_run_id}
                      </td>
                      <td className="px-4 py-3">{b.metric_name}</td>
                      <td className="px-4 py-3 font-mono">
                        {b.metric_value.toLocaleString(undefined, {
                          maximumFractionDigits: 4,
                        })}
                      </td>
                      <td className="px-4 py-3 text-muted text-xs">
                        {b.created_at
                          ? new Date(b.created_at).toLocaleDateString()
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
