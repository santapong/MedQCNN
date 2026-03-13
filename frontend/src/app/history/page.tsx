"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  fetchPredictions,
  type PaginatedPredictions,
  type PredictionDetail,
} from "@/lib/api";
import { TableSkeleton } from "@/components/Skeleton";

export default function HistoryPage() {
  const [data, setData] = useState<PaginatedPredictions | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [labelFilter, setLabelFilter] = useState<string>("");
  const [filenameSearch, setFilenameSearch] = useState("");
  const limit = 20;

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const result = await fetchPredictions({
          offset: page * limit,
          limit,
          label: labelFilter || undefined,
          filename: filenameSearch || undefined,
        });
        setData(result);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to load predictions"
        );
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [page, labelFilter, filenameSearch]);

  function exportCsv() {
    if (!data || data.items.length === 0) return;
    const headers = [
      "id",
      "filename",
      "label",
      "confidence",
      "model_version",
      "n_qubits",
      "created_at",
    ];
    const rows = data.items.map((r) =>
      [
        r.id,
        r.image_filename,
        r.label,
        r.confidence.toFixed(4),
        r.model_version,
        r.n_qubits,
        r.created_at || "",
      ].join(",")
    );
    const csv = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="max-w-6xl mx-auto px-6 py-10">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Prediction History
          </h1>
          <p className="text-muted mt-2">
            Browse and filter all past predictions
          </p>
        </div>
        <button
          onClick={exportCsv}
          disabled={!data || data.items.length === 0}
          className="px-4 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent/90 disabled:opacity-50 transition-colors"
        >
          Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-6">
        <select
          value={labelFilter}
          onChange={(e) => {
            setLabelFilter(e.target.value);
            setPage(0);
          }}
          className="bg-card-bg border border-card-border rounded-lg px-3 py-2 text-sm text-foreground"
        >
          <option value="">All Labels</option>
          <option value="Benign">Benign</option>
          <option value="Malignant">Malignant</option>
        </select>
        <input
          type="text"
          placeholder="Search by filename..."
          value={filenameSearch}
          onChange={(e) => {
            setFilenameSearch(e.target.value);
            setPage(0);
          }}
          className="bg-card-bg border border-card-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted/50 w-64"
        />
        {data && (
          <span className="text-sm text-muted self-center ml-auto">
            {data.total} total prediction{data.total !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {error && (
        <div className="rounded-xl p-4 bg-danger/10 border border-danger/30 text-danger text-sm mb-6">
          {error}
        </div>
      )}

      {loading ? (
        <TableSkeleton rows={8} />
      ) : data && data.items.length > 0 ? (
        <>
          <div className="rounded-xl border border-card-border bg-card-bg overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-card-border text-left text-xs text-muted uppercase tracking-wider">
                  <th className="px-4 py-3">ID</th>
                  <th className="px-4 py-3">Filename</th>
                  <th className="px-4 py-3">Label</th>
                  <th className="px-4 py-3">Confidence</th>
                  <th className="px-4 py-3">Qubits</th>
                  <th className="px-4 py-3">Date</th>
                  <th className="px-4 py-3">Detail</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((row) => (
                  <PredictionRow key={row.id} row={row} />
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between mt-4">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-3 py-1.5 text-sm rounded-lg border border-card-border text-muted hover:text-foreground disabled:opacity-30 transition-colors"
            >
              Previous
            </button>
            <span className="text-sm text-muted">
              Page {page + 1} of {Math.max(1, Math.ceil(data.total / limit))}
            </span>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={(page + 1) * limit >= data.total}
              className="px-3 py-1.5 text-sm rounded-lg border border-card-border text-muted hover:text-foreground disabled:opacity-30 transition-colors"
            >
              Next
            </button>
          </div>
        </>
      ) : (
        <div className="rounded-xl border border-card-border bg-card-bg h-48 flex items-center justify-center">
          <p className="text-muted text-sm">
            No predictions yet. Run a diagnosis to see results here.
          </p>
        </div>
      )}
    </main>
  );
}

function PredictionRow({ row }: { row: PredictionDetail }) {
  const isBenign = row.label === "Benign";
  const date = row.created_at
    ? new Date(row.created_at).toLocaleDateString()
    : "—";

  return (
    <tr className="border-b border-card-border hover:bg-white/[0.02] transition-colors">
      <td className="px-4 py-3 font-mono text-muted">{row.id}</td>
      <td className="px-4 py-3 truncate max-w-[180px]">
        {row.image_filename}
      </td>
      <td className="px-4 py-3">
        <span
          className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
            isBenign
              ? "bg-success/10 text-success"
              : "bg-danger/10 text-danger"
          }`}
        >
          {row.label}
        </span>
      </td>
      <td className="px-4 py-3 font-mono">
        {(row.confidence * 100).toFixed(1)}%
      </td>
      <td className="px-4 py-3 text-muted">{row.n_qubits}</td>
      <td className="px-4 py-3 text-muted text-xs">{date}</td>
      <td className="px-4 py-3">
        <Link
          href={`/history/${row.id}`}
          className="text-accent-light hover:text-accent text-xs"
        >
          View
        </Link>
      </td>
    </tr>
  );
}
