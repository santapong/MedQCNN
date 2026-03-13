interface QuantumBarProps {
  qubitIndex: number;
  value: number; // in [-1, 1]
}

export default function QuantumBar({ qubitIndex, value }: QuantumBarProps) {
  // Map [-1, 1] to [0, 100] for the bar width
  const percentage = ((value + 1) / 2) * 100;
  const isNegative = value < 0;

  return (
    <div className="flex items-center gap-3">
      <span className="text-xs font-mono text-muted w-6 text-right">
        Q{qubitIndex}
      </span>
      <div className="flex-1 h-6 bg-white/5 rounded-md overflow-hidden relative">
        {/* Center line */}
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-white/20 z-10" />
        {/* Bar from center */}
        <div
          className={`absolute top-0.5 bottom-0.5 rounded-sm transition-all duration-500 ${
            isNegative ? "bg-danger/70" : "bg-success/70"
          }`}
          style={{
            left: isNegative ? `${percentage}%` : "50%",
            width: `${Math.abs(percentage - 50)}%`,
          }}
        />
      </div>
      <span
        className={`text-xs font-mono w-14 text-right ${
          isNegative ? "text-danger" : "text-success"
        }`}
      >
        {value >= 0 ? "+" : ""}
        {value.toFixed(3)}
      </span>
    </div>
  );
}
