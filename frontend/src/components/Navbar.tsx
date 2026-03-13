"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Diagnose" },
  { href: "/model", label: "Model Info" },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-card-border bg-card-bg/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-lg bg-accent flex items-center justify-center text-white font-bold text-sm">
            Q
          </div>
          <span className="font-semibold text-lg tracking-tight">
            Med<span className="text-accent-light">QCNN</span>
          </span>
        </Link>

        <div className="flex items-center gap-1">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                pathname === link.href
                  ? "bg-accent/15 text-accent-light"
                  : "text-muted hover:text-foreground hover:bg-white/5"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
