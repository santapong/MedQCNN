"use client";

import type { NoiseSensitivityPoint } from "@/lib/api";
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

interface NoiseSensitivityChartProps {
  points: NoiseSensitivityPoint[];
  metricName: string;
}

const LINE_COLORS = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"];

/**
 * Plots accuracy (or any benchmark) versus depolarising noise rate.
 * One line per training_run_id so multiple runs can be overlaid.
 */
export default function NoiseSensitivityChart({
  points,
  metricName,
}: NoiseSensitivityChartProps) {
  if (points.length === 0) {
    return (
      <div className="rounded-xl p-6 bg-surface border border-default text-muted text-sm">
        <p className="font-medium text-fg">No noise sensitivity data yet.</p>
        <p className="mt-1">
          Run{" "}
          <code className="text-xs bg-bg px-1.5 py-0.5 rounded">
            scripts/noise_sensitivity_eval.py --training-run-id &lt;id&gt;
          </code>{" "}
          to populate this chart.
        </p>
      </div>
    );
  }

  // Group points by training_run_id and pivot so each x-value (noise
  // rate) is one row carrying one column per run.
  const runIds = Array.from(new Set(points.map((p) => p.training_run_id))).sort(
    (a, b) => a - b
  );
  const allRates = Array.from(
    new Set(points.map((p) => p.depolarising_p))
  ).sort((a, b) => a - b);

  const data = allRates.map((rate) => {
    const row: Record<string, number> = { depolarising_p: rate };
    for (const runId of runIds) {
      const point = points.find(
        (p) => p.depolarising_p === rate && p.training_run_id === runId
      );
      if (point) row[`run_${runId}`] = point.metric_value;
    }
    return row;
  });

  return (
    <div className="rounded-xl p-6 bg-surface border border-default">
      <h3 className="text-lg font-semibold mb-1">Noise sensitivity</h3>
      <p className="text-sm text-muted mb-4">
        {metricName} vs. depolarising rate per gate. One line per training
        run.
      </p>
      <div style={{ width: "100%", height: 320 }}>
        <ResponsiveContainer>
          <LineChart
            data={data}
            margin={{ top: 5, right: 20, bottom: 20, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.3} />
            <XAxis
              dataKey="depolarising_p"
              type="number"
              domain={[0, "dataMax"]}
              tickFormatter={(v: number) => v.toFixed(3)}
              label={{
                value: "depolarising p",
                position: "insideBottom",
                offset: -10,
                fontSize: 12,
              }}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v: number) => v.toFixed(2)}
              label={{
                value: metricName,
                angle: -90,
                position: "insideLeft",
                offset: 10,
                fontSize: 12,
              }}
            />
            <Tooltip
              formatter={(v: number) => v.toFixed(4)}
              labelFormatter={(rate: number) =>
                `depolarising p = ${rate.toFixed(4)}`
              }
            />
            <Legend />
            {runIds.map((runId, idx) => (
              <Line
                key={runId}
                type="monotone"
                dataKey={`run_${runId}`}
                name={`run #${runId}`}
                stroke={LINE_COLORS[idx % LINE_COLORS.length]}
                strokeWidth={2}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
