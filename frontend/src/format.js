// Small display helpers shared by the pages.

// The backend stores times in UTC; SQLite drops the "Z", so add it back before parsing.
export function parseDate(s) {
  if (!s) return null;
  return new Date(/[zZ]$|[+-]\d\d:\d\d$/.test(s) ? s : `${s}Z`);
}

export function formatDate(s) {
  const d = parseDate(s);
  if (!d || Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function formatAge(months) {
  if (months == null) return "—";
  if (months < 24) return `${months} mo`;
  return `${Math.floor(months / 12)} y ${months % 12} mo`;
}

export const pct = (p) => (p == null ? "—" : `${Math.round(p * 100)}%`);

export function priority(p) {
  if (p == null) return { label: "—", tone: "locked" };
  if (p >= 0.7) return { label: "High", tone: "crit" };
  if (p >= 0.4) return { label: "Check", tone: "warn" };
  return { label: "Low", tone: "ok" };
}

export const toneColor = (p) => (p >= 0.7 ? "var(--crit)" : p >= 0.4 ? "var(--warn)" : "var(--ok)");

export const sexLabel = (s) => (s === "M" ? "Male" : s === "F" ? "Female" : "—");
