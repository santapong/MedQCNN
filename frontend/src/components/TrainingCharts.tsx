"use client";

import type { TrainingRun } from "@/lib/api";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface TrainingChartsProps {
  run: TrainingRun;
}

export default function TrainingCharts({ run }: TrainingChartsProps) {
  if (!run.history) return null;

  const epochs = run.history.train_loss.length;
  const lossData = Array.from({ length: epochs }, (_, i) => ({
    epoch: i + 1,
    train_loss: run.history!.train_loss[i],
    val_loss: run.history!.val_loss[i],
  }));

  const accData = Array.from({ length: epochs }, (_, i) => ({
    epoch: i + 1,
    train_acc: +(run.history!.train_acc[i] * 100).toFixed(1),
    val_acc: +(run.history!.val_acc[i] * 100).toFixed(1),
  }));

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-card-border bg-card-bg p-5">
        <h3 className="text-sm font-semibold text-muted uppercase tracking-wider mb-1">
          Training Run #{run.id} — {run.dataset} ({run.n_qubits}q,{" "}
          {run.epochs} epochs)
        </h3>
        <p className="text-xs text-muted mb-4">
          LR: {run.learning_rate} | Batch: {run.batch_size} | Layers:{" "}
          {run.n_layers}
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Loss chart */}
          <div>
            <h4 className="text-xs font-medium text-muted mb-3">
              Loss Curves
            </h4>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="epoch"
                  stroke="#64748b"
                  fontSize={11}
                  label={{
                    value: "Epoch",
                    position: "insideBottom",
                    offset: -5,
                    fill: "#64748b",
                    fontSize: 11,
                  }}
                />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    background: "#111827",
                    border: "1px solid #1e293b",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={false}
                  name="Train Loss"
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                  name="Val Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Accuracy chart */}
          <div>
            <h4 className="text-xs font-medium text-muted mb-3">
              Accuracy Curves (%)
            </h4>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={accData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="epoch"
                  stroke="#64748b"
                  fontSize={11}
                  label={{
                    value: "Epoch",
                    position: "insideBottom",
                    offset: -5,
                    fill: "#64748b",
                    fontSize: 11,
                  }}
                />
                <YAxis
                  stroke="#64748b"
                  fontSize={11}
                  domain={[0, 100]}
                  unit="%"
                />
                <Tooltip
                  contentStyle={{
                    background: "#111827",
                    border: "1px solid #1e293b",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                  formatter={(value: number) => `${value}%`}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line
                  type="monotone"
                  dataKey="train_acc"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={false}
                  name="Train Acc"
                />
                <Line
                  type="monotone"
                  dataKey="val_acc"
                  stroke="#818cf8"
                  strokeWidth={2}
                  dot={false}
                  name="Val Acc"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
