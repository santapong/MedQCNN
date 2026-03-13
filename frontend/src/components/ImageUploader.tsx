"use client";

import { useCallback, useState } from "react";

interface ImageUploaderProps {
  onImageSelected: (file: File, previewUrl: string) => void;
  disabled?: boolean;
}

export default function ImageUploader({
  onImageSelected,
  disabled,
}: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      const url = URL.createObjectURL(file);
      onImageSelected(file, url);
    },
    [onImageSelected]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`
        relative flex flex-col items-center justify-center
        w-full h-64 rounded-xl border-2 border-dashed
        cursor-pointer transition-all duration-200
        ${
          isDragging
            ? "border-accent bg-accent/10 scale-[1.02]"
            : "border-card-border hover:border-accent/50 hover:bg-white/[0.02]"
        }
        ${disabled ? "opacity-50 pointer-events-none" : ""}
      `}
    >
      <input
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="hidden"
        disabled={disabled}
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
          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
        />
      </svg>
      <p className="text-sm text-muted">
        <span className="text-accent-light font-medium">Click to upload</span>{" "}
        or drag and drop
      </p>
      <p className="text-xs text-muted/70 mt-1">PNG, JPG, BMP up to 10 MB</p>
    </label>
  );
}
