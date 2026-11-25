import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { listStudies } from "../api.js";
import { formatAge, formatDate, pct, priority, toneColor } from "../format.js";

const TABS = [
  { key: "review", label: "To review" },
  { key: "approved", label: "Approved" },
  { key: "all", label: "All" },
];

export default function Worklist() {
  const [tab, setTab] = useState("review");
  const [q, setQ] = useState("");
  const [rows, setRows] = useState(null);
  const [all, setAll] = useState([]);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  // counts for the summary tiles
  useEffect(() => {
    listStudies().then(setAll).catch(() => {});
  }, []);

  // the table, reloaded when the tab or the search text changes
  useEffect(() => {
    let alive = true;
    setError("");
    const t = setTimeout(() => {
      listStudies({ status: tab === "all" ? undefined : tab, q: q.trim() || undefined })
        .then((r) => alive && setRows(r))
        .catch((e) => alive && setError(e.message));
    }, q ? 250 : 0);
    return () => { alive = false; clearTimeout(t); };
  }, [tab, q]);

  const stats = useMemo(() => {
    const review = all.filter((s) => s.status === "review");
    return {
      review: review.length,
      high: review.filter((s) => (s.pneumonia_probability ?? 0) >= 0.7).length,
      approved: all.filter((s) => s.status === "approved").length,
      total: all.length,
    };
  }, [all]);

  const open = (id) => navigate(`/studies/${id}`);

  return (
    <section className="view">
      <div className="view-head">
        <div>
          <h1>Worklist</h1>
          <p>X-rays waiting for a doctor, highest pneumonia probability first.</p>
        </div>
        <Link className="btn primary" to="/new">+ New study</Link>
      </div>

      <div className="stats">
        <div className="panel stat"><div className="k">To review</div><div className="v">{stats.review}</div></div>
        <div className="panel stat crit"><div className="k">High priority waiting</div><div className="v">{stats.high}</div></div>
        <div className="panel stat"><div className="k">Approved</div><div className="v">{stats.approved}</div></div>
        <div className="panel stat"><div className="k">Total studies</div><div className="v">{stats.total}</div></div>
      </div>

      <div className="panel">
        <div className="panel-h">
          <div className="tools">
            <div className="tabs" role="tablist" aria-label="Filter studies">
              {TABS.map((t) => (
                <button key={t.key} role="tab" aria-selected={tab === t.key} onClick={() => setTab(t.key)}>{t.label}</button>
              ))}
            </div>
            <input id="search" className="search" type="search" placeholder="Search study ID or patient"
                   aria-label="Search studies" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
        </div>

        {error && <div className="panel-b"><div className="error">{error}</div></div>}
        {!error && rows === null && <div className="state"><span className="spinner" />Loading studies…</div>}
        {!error && rows?.length === 0 && (
          <div className="state">
            {q ? "No studies match this search." : tab === "review" ? "No X-rays waiting for review." : "No studies here yet."}
            {!q && <Link className="btn" to="/new">Upload an X-ray</Link>}
          </div>
        )}
        {!error && rows?.length > 0 && (
          <div className="tablewrap">
            <table>
              <thead>
                <tr><th>Study</th><th>Patient</th><th>Age · Sex</th><th>Uploaded</th><th>Pneumonia</th><th>Priority</th><th>Status</th></tr>
              </thead>
              <tbody>
                {rows.map((s) => {
                  const pr = priority(s.pneumonia_probability);
                  const p = s.pneumonia_probability ?? 0;
                  return (
                    <tr key={s.id} className="clickable" onClick={() => open(s.id)}>
                      <td><button className="idlink" type="button" onClick={(e) => { e.stopPropagation(); open(s.id); }}>{s.code}</button></td>
                      <td>{s.patient_ref || <span className="help">—</span>}</td>
                      <td className="num">{formatAge(s.age_months)} · {s.sex}</td>
                      <td className="num">{formatDate(s.created_at)}</td>
                      <td>
                        <div className="bar">
                          <i><b style={{ width: `${Math.round(p * 100)}%`, background: toneColor(p) }} /></i>
                          <span>{pct(s.pneumonia_probability)}</span>
                        </div>
                      </td>
                      <td><span className={`chip ${pr.tone}`}>{pr.label}</span></td>
                      <td>{s.status === "approved" ? <span className="chip ok">Approved</span> : <span className="chip warn">To review</span>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
