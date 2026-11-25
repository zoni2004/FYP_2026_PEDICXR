import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fileUrl, getStudy, pdfUrl, saveReport, sendFeedback } from "../api.js";
import { formatAge, formatDate, pct, sexLabel, toneColor } from "../format.js";
import { useToast } from "../components/Toast.jsx";
import XrayViewer from "../components/XrayViewer.jsx";

export default function StudyViewer() {
  const { id } = useParams();
  const [study, setStudy] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    setStudy(null);
    setError("");
    getStudy(id).then((s) => alive && setStudy(s)).catch((e) => alive && setError(e.message));
    return () => { alive = false; };
  }, [id]);

  if (error) {
    return (
      <section className="view">
        <div className="error">{error}</div>
        <Link className="btn" to="/worklist" style={{ alignSelf: "flex-start" }}>Back to worklist</Link>
      </section>
    );
  }
  if (!study) return <section className="view"><div className="state"><span className="spinner" />Loading study…</div></section>;

  return (
    <section className="view">
      <div className="vhead">
        <h1>{study.code}</h1>
        {study.status === "approved" ? <span className="chip ok">Approved</span> : <span className="chip warn">To review</span>}
        {study.model_name && <span className="chip info">{study.model_name}</span>}
      </div>
      <div className="meta">
        <span>Patient <b>{study.patient_ref || "—"}</b></span>
        <span>Age <b>{formatAge(study.age_months)}</b></span>
        <span>Sex <b>{sexLabel(study.sex)}</b></span>
        <span>Uploaded <b>{formatDate(study.created_at)}</b></span>
        <span>File <b>{study.original_name}</b></span>
      </div>

      <div className="vgrid">
        <div className="imgcol">
          <XrayViewer imageUrl={fileUrl(study.image_url)} heatmapUrl={fileUrl(study.heatmap_url)} />
        </div>
        <ResultPanel study={study} onChange={setStudy} />
        <ReportPanel study={study} onChange={setStudy} />
      </div>
    </section>
  );
}

function ResultPanel({ study, onChange }) {
  const [threshold, setThreshold] = useState(study.threshold);
  const [sending, setSending] = useState(false);
  const toast = useToast();
  const p = study.pneumonia_probability ?? 0;
  const positive = p >= threshold;

  async function feedback(finding, agree) {
    setSending(true);
    try {
      await sendFeedback(study.id, finding, agree);
      onChange({ ...study, feedback: { ...study.feedback, [finding]: agree } });
      toast(`Saved: you ${agree ? "agree" : "disagree"} with ${finding}`);
    } catch (e) {
      toast(e.message);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="stack">
      <div className="panel">
        <div className="panel-h"><h2>AI result</h2><span className="help">P(pneumonia)</span></div>
        <div className="panel-b stack">
          <div className={`verdict ${positive ? "crit" : "ok"}`}>
            <span className="t">{positive ? "Pneumonia likely" : "No pneumonia detected"}</span>
            <span className="p">{pct(p)}</span>
            <span className="help">{positive ? "Above" : "Below"} the {threshold.toFixed(2)} threshold.</span>
          </div>
          <div className="thrline">
            <label htmlFor="thr">Threshold</label>
            <input id="thr" type="range" min="0.05" max="0.95" step="0.05" value={threshold} onChange={(e) => setThreshold(+e.target.value)} />
            <span className="mono num">{threshold.toFixed(2)}</span>
          </div>
          <span className="help">Changing the threshold only changes this view. The model's default is {study.threshold.toFixed(2)}.</span>
        </div>
      </div>

      <div className="panel">
        <div className="panel-h"><h2>Findings</h2><span className="help">Do you agree?</span></div>
        <div className="panel-b">
          <ul className="findings">
            {study.findings.map((f) => {
              const mine = study.feedback[f.name];
              return (
                <li className="finding" key={f.name}>
                  <div className="fh"><span className="fn">{f.name}</span><span className="mono num">{pct(f.probability)}</span></div>
                  <div className="track">
                    <b style={{ width: `${Math.round(f.probability * 100)}%`, background: toneColor(f.probability) }} />
                    <span className="thr" style={{ left: `calc(${threshold * 100}% - 1px)` }} />
                  </div>
                  <div className="fb">
                    <button type="button" className="yes" aria-pressed={mine === true} disabled={sending} onClick={() => feedback(f.name, true)}>Agree</button>
                    <button type="button" className="no" aria-pressed={mine === false} disabled={sending} onClick={() => feedback(f.name, false)}>Disagree</button>
                  </div>
                </li>
              );
            })}
          </ul>
          {study.findings.length === 1 && (
            <p className="help" style={{ margin: "14px 0 0" }}>The current model checks pneumonia only. More findings (effusion, cardiomegaly, TB) come with the new vision model.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function ReportPanel({ study, onChange }) {
  const saved = study.report?.text ?? "";
  const approved = study.report?.status === "approved";
  const [text, setText] = useState(saved);
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState("");
  const toast = useToast();
  const dirty = text !== saved;

  async function save(status) {
    if (!text.trim()) { setError("The report can't be empty."); return; }
    setBusy(true);
    setError("");
    try {
      const s = await saveReport(study.id, text, status);
      onChange(s);
      setConfirming(false);
      toast(status === "approved" ? "Report approved" : "Draft saved");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <div className="panel">
        <div className="panel-h">
          <h2>Report</h2>
          {approved ? <span className="chip ok">Approved</span> : <span className="chip warn">{dirty ? "Draft · unsaved changes" : "Draft"}</span>}
        </div>
        <div className="panel-b stack">
          <span className="help">
            {approved ? "This report is approved and can't be changed." : "Draft written from the AI result. Check it and edit before approving."}
          </span>
          <label htmlFor="report" className="help" hidden>Report text</label>
          <textarea id="report" className="report" spellCheck={false} value={text} readOnly={approved}
                    onChange={(e) => setText(e.target.value)} />
          {error && <div className="error" role="alert">{error}</div>}

          {confirming ? (
            <div className="confirm">
              <strong>Approve this report?</strong>
              <span>After approving you can't edit it any more.</span>
              <div className="rep-actions">
                <button type="button" className="btn danger" disabled={busy} onClick={() => save("approved")}>{busy ? "Approving…" : "Yes, approve"}</button>
                <button type="button" className="btn" disabled={busy} onClick={() => setConfirming(false)}>Cancel</button>
              </div>
            </div>
          ) : (
            <div className="rep-actions">
              {!approved && <button type="button" className="btn" disabled={busy || !dirty} onClick={() => save("draft")}>Save draft</button>}
              {!approved && <button type="button" className="btn primary" disabled={busy} onClick={() => setConfirming(true)}>Approve report</button>}
              <a className="btn" href={pdfUrl(study.id)} aria-disabled={dirty}
                 onClick={(e) => { if (dirty) { e.preventDefault(); toast("Save the report first, the PDF uses the saved text."); } }}>
                Export PDF
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
