"use client";

import type { BenchmarkMetric, TrainingRun } from "@/lib/api";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface BenchmarkChartsProps {
  runs: TrainingRun[];
  benchmarks: BenchmarkMetric[];
  referenceData: {
    configs: {
      label: string;
      qubits: number;
      totalParams: number;
      quantumParams: number;
      memoryGb: number;
      latencyMs: number;
    }[];
    classicalBaseline: {
      label: string;
      totalParams: number;
      memoryGb: number;
      latencyMs: number;
    };
  };
}

const COLORS = ["#6366f1", "#818cf8", "#f59e0b"];

export default function BenchmarkCharts({
  runs,
  benchmarks,
  referenceData,
}: BenchmarkChartsProps) {
  // Parameter comparison data
  const paramData = [
    ...referenceData.configs.map((c) => ({
      name: c.label,
      params: c.totalParams,
    })),
    {
      name: referenceData.classicalBaseline.label,
      params: referenceData.classicalBaseline.totalParams,
    },
  ];

  // Memory comparison data
  const memoryData = [
    ...referenceData.configs.map((c) => ({
      name: c.label,
      memory: c.memoryGb,
    })),
    {
      name: referenceData.classicalBaseline.label,
      memory: referenceData.classicalBaseline.memoryGb,
    },
  ];

  // Latency comparison data
  const latencyData = [
    ...referenceData.configs.map((c) => ({
      name: c.label,
      latency: c.latencyMs,
    })),
    {
      name: referenceData.classicalBaseline.label,
      latency: referenceData.classicalBaseline.latencyMs,
    },
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Parameter count */}
      <div className="rounded-xl border border-card-border bg-card-bg p-5">
        <h4 className="text-xs font-medium text-muted uppercase tracking-wider mb-4">
          Trainable Parameters
        </h4>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={paramData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis type="number" stroke="#64748b" fontSize={10} />
            <YAxis
              type="category"
              dataKey="name"
              stroke="#64748b"
              fontSize={10}
              width={90}
            />
            <Tooltip
              contentStyle={{
                background: "#111827",
                border: "1px solid #1e293b",
                borderRadius: "8px",
                fontSize: "11px",
              }}
              formatter={(value: number) => value.toLocaleString()}
            />
            <Bar dataKey="params" radius={[0, 4, 4, 0]}>
              {paramData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Memory */}
      <div className="rounded-xl border border-card-border bg-card-bg p-5">
        <h4 className="text-xs font-medium text-muted uppercase tracking-wider mb-4">
          Memory Usage (GB)
        </h4>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={memoryData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis type="number" stroke="#64748b" fontSize={10} unit=" GB" />
            <YAxis
              type="category"
              dataKey="name"
              stroke="#64748b"
              fontSize={10}
              width={90}
            />
            <Tooltip
              contentStyle={{
                background: "#111827",
                border: "1px solid #1e293b",
                borderRadius: "8px",
                fontSize: "11px",
              }}
              formatter={(value: number) => `${value} GB`}
            />
            <Bar dataKey="memory" radius={[0, 4, 4, 0]}>
              {memoryData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Latency */}
      <div className="rounded-xl border border-card-border bg-card-bg p-5">
        <h4 className="text-xs font-medium text-muted uppercase tracking-wider mb-4">
          Inference Latency (ms)
        </h4>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={latencyData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis type="number" stroke="#64748b" fontSize={10} unit=" ms" />
            <YAxis
              type="category"
              dataKey="name"
              stroke="#64748b"
              fontSize={10}
              width={90}
            />
            <Tooltip
              contentStyle={{
                background: "#111827",
                border: "1px solid #1e293b",
                borderRadius: "8px",
                fontSize: "11px",
              }}
              formatter={(value: number) => `${value} ms`}
            />
            <Bar dataKey="latency" radius={[0, 4, 4, 0]}>
              {latencyData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
