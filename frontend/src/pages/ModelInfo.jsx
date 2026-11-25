import { useEffect, useState } from "react";
import { getModelInfo } from "../api.js";

const METRICS = [
  ["auc", "AUC"], ["recall", "RECALL"], ["specificity", "SPECIFICITY"],
  ["precision", "PRECISION"], ["f1", "F1"], ["acc", "ACCURACY"],
];

export default function ModelInfo() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getModelInfo().then(setInfo).catch((e) => setError(e.message));
  }, []);

  if (error) return <section className="view"><div className="error">{error}</div></section>;
  if (!info) return <section className="view"><div className="state"><span className="spinner" />Loading model info…</div></section>;

  return (
    <section className="view">
      <div className="view-head">
        <div>
          <h1>Model info</h1>
          <p>{info.notes}</p>
        </div>
        <div className="tools">
          <span className="chip info">{info.name}</span>
          <span className={`chip ${info.loaded ? "ok" : "crit"}`}>{info.loaded ? "Loaded" : "Not loaded"}</span>
          <span className="chip locked">{info.uses_age_sex ? "Uses age + sex" : "Image only"}</span>
        </div>
      </div>

      <div className="mtiles">
        {METRICS.filter(([k]) => info.test_metrics[k]).map(([k, label]) => (
          <div className="panel mtile" key={k}>
            <div className="k">{label}</div>
            <div className="v">{info.test_metrics[k].mean.toFixed(3)}</div>
            <div className="s">± {info.test_metrics[k].std.toFixed(3)} over 5 folds</div>
          </div>
        ))}
      </div>

      <div className="panel">
        <div className="panel-h"><h2>Per-fold test results</h2><span className="help">Checkpoint in use: {info.checkpoint}</span></div>
        <div className="tablewrap">
          <table className="num">
            <thead>
              <tr><th>Fold</th><th>Best epoch</th>{METRICS.map(([k, l]) => <th key={k}>{l}</th>)}</tr>
            </thead>
            <tbody>
              {info.folds.map((f) => (
                <tr key={f.fold}>
                  <td>{f.fold}{f.fold === 0 && <> <span className="chip info">in use</span></>}</td>
                  <td>{f.best_epoch}</td>
                  {METRICS.map(([k]) => <td key={k}>{f[k]?.toFixed(3) ?? "—"}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="note">
        Decision threshold: {info.threshold.toFixed(2)}. The test set is about 84% normal, so precision is low; AUC and recall are the
        numbers to watch. Findings checked: {info.findings.join(", ")}.
      </p>
    </section>
  );
}
