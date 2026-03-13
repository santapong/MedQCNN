"use client";

import { useEffect, useState } from "react";
import {
  fetchTrainingRuns,
  type TrainingRun,
  type TrainingRunListResponse,
} from "@/lib/api";
import { CardSkeleton, TableSkeleton } from "@/components/Skeleton";
import TrainingCharts from "@/components/TrainingCharts";

export default function TrainingPage() {
  const [data, setData] = useState<TrainingRunListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<TrainingRun | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const result = await fetchTrainingRuns({ limit: 50 });
        setData(result);
        if (result.items.length > 0) {
          setSelectedRun(result.items[0]);
        }
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to load training runs"
        );
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <main className="max-w-7xl mx-auto px-6 py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">
          Training Dashboard
        </h1>
        <p className="text-muted mt-2">
          View training runs, metrics, and loss/accuracy curves
        </p>
      </div>

      {error && (
        <div className="rounded-xl p-4 bg-danger/10 border border-danger/30 text-danger text-sm mb-6">
          {error}
        </div>
      )}

      {loading ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <CardSkeleton />
            <CardSkeleton />
            <CardSkeleton />
          </div>
          <TableSkeleton />
        </div>
      ) : data && data.items.length > 0 ? (
        <div className="space-y-8">
          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Total Runs" value={data.total} />
            <MetricCard
              label="Best Val Acc"
              value={`${(Math.max(...data.items.filter((r) => r.final_val_acc !== null).map((r) => r.final_val_acc!), 0) * 100).toFixed(1)}%`}
            />
            <MetricCard
              label="Best Test Acc"
              value={
                data.items.some((r) => r.final_test_acc !== null)
                  ? `${(Math.max(...data.items.filter((r) => r.final_test_acc !== null).map((r) => r.final_test_acc!), 0) * 100).toFixed(1)}%`
                  : "—"
              }
            />
            <MetricCard
              label="Datasets"
              value={new Set(data.items.map((r) => r.dataset)).size}
            />
          </div>

          {/* Training runs table */}
          <div className="rounded-xl border border-card-border bg-card-bg overflow-hidden">
            <div className="px-4 py-3 border-b border-card-border">
              <h3 className="text-sm font-semibold text-muted uppercase tracking-wider">
                Training Runs
              </h3>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-card-border text-left text-xs text-muted uppercase tracking-wider">
                  <th className="px-4 py-3">ID</th>
                  <th className="px-4 py-3">Dataset</th>
                  <th className="px-4 py-3">Qubits</th>
                  <th className="px-4 py-3">Epochs</th>
                  <th className="px-4 py-3">Train Acc</th>
                  <th className="px-4 py-3">Val Acc</th>
                  <th className="px-4 py-3">Test Acc</th>
                  <th className="px-4 py-3">Duration</th>
                  <th className="px-4 py-3">Charts</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((run) => (
                  <tr
                    key={run.id}
                    className={`border-b border-card-border hover:bg-white/[0.02] cursor-pointer transition-colors ${
                      selectedRun?.id === run.id ? "bg-accent/5" : ""
                    }`}
                    onClick={() => setSelectedRun(run)}
                  >
                    <td className="px-4 py-3 font-mono text-muted">
                      {run.id}
                    </td>
                    <td className="px-4 py-3">{run.dataset}</td>
                    <td className="px-4 py-3">{run.n_qubits}</td>
                    <td className="px-4 py-3">{run.epochs}</td>
                    <td className="px-4 py-3 font-mono">
                      {run.final_train_acc !== null
                        ? `${(run.final_train_acc * 100).toFixed(1)}%`
                        : "—"}
                    </td>
                    <td className="px-4 py-3 font-mono">
                      {run.final_val_acc !== null
                        ? `${(run.final_val_acc * 100).toFixed(1)}%`
                        : "—"}
                    </td>
                    <td className="px-4 py-3 font-mono">
                      {run.final_test_acc !== null
                        ? `${(run.final_test_acc * 100).toFixed(1)}%`
                        : "—"}
                    </td>
                    <td className="px-4 py-3 text-muted">
                      {run.duration_seconds !== null
                        ? `${run.duration_seconds.toFixed(0)}s`
                        : "—"}
                    </td>
                    <td className="px-4 py-3">
                      {run.history ? (
                        <span className="text-accent-light text-xs">View</span>
                      ) : (
                        <span className="text-muted/50 text-xs">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Charts for selected run */}
          {selectedRun && selectedRun.history && (
            <TrainingCharts run={selectedRun} />
          )}
        </div>
      ) : (
        <div className="rounded-xl border border-card-border bg-card-bg h-48 flex items-center justify-center">
          <p className="text-muted text-sm">
            No training runs yet. Run{" "}
            <code className="bg-white/10 px-2 py-0.5 rounded">
              uv run python scripts/train.py
            </code>{" "}
            to see results here.
          </p>
        </div>
      )}
    </main>
  );
}

function MetricCard({
  label,
  value,
}: {
  label: string;
  value: number | string;
}) {
  return (
    <div className="rounded-xl border border-card-border bg-card-bg p-4 text-center">
      <p className="text-2xl font-bold font-mono">{value}</p>
      <p className="text-xs text-muted mt-1">{label}</p>
    </div>
  );
}
