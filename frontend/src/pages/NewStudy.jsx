import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createStudy } from "../api.js";
import { useToast } from "../components/Toast.jsx";

const MAX_MB = 25;
const OK_EXT = [".png", ".jpg", ".jpeg", ".dcm", ".dicom"];
const extOf = (name) => (name.match(/\.[^.]+$/)?.[0] || "").toLowerCase();

export default function NewStudy() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [drag, setDrag] = useState(false);
  const [patientRef, setPatientRef] = useState("");
  const [years, setYears] = useState("");
  const [months, setMonths] = useState("0");
  const [sex, setSex] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);
  const navigate = useNavigate();
  const toast = useToast();

  // free the preview image when it changes or the page closes
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  function pick(f) {
    if (!f) return;
    const ext = extOf(f.name);
    if (!OK_EXT.includes(ext)) { setError("That file type isn't supported. Choose a PNG, JPG or DICOM (.dcm) file."); return; }
    if (f.size > MAX_MB * 1024 * 1024) { setError(`The file is larger than ${MAX_MB} MB.`); return; }
    setError("");
    setFile(f);
    setPreview(ext === ".dcm" || ext === ".dicom" ? null : URL.createObjectURL(f));
  }

  function clear() {
    setFile(null);
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function submit(e) {
    e.preventDefault();
    if (!file) { setError("Choose an X-ray file first."); return; }
    const y = Number.parseInt(years, 10);
    const m = Number.parseInt(months || "0", 10);
    if (Number.isNaN(y) || y < 0 || y > 18 || Number.isNaN(m) || m < 0 || m > 11) {
      setError("Enter an age between 0 and 18 years, and 0 to 11 months."); return;
    }
    if (y === 18 && m > 0) { setError("This model is for children up to 18 years."); return; }
    if (!sex) { setError("Choose the patient's sex."); return; }
    setError("");
    setBusy(true);
    try {
      const study = await createStudy({ file, ageMonths: y * 12 + m, sex, patientRef: patientRef.trim() });
      toast(`${study.code} analyzed`);
      navigate(`/studies/${study.id}`);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  const isDicom = file && !preview;

  return (
    <section className="view">
      <div className="view-head">
        <div>
          <h1>New study</h1>
          <p>Upload a chest X-ray with the child's age and sex. The AI model analyzes it and opens the result.</p>
        </div>
      </div>

      <form className="newgrid" onSubmit={submit} noValidate>
        <div className="panel">
          <div className="panel-h"><h2>Chest X-ray</h2><span className="help">PNG, JPG or DICOM · up to {MAX_MB} MB</span></div>
          <div className="panel-b stack">
            <label
              className={`drop${drag ? " drag" : ""}`}
              htmlFor="file"
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); }}
            >
              {preview && <img src={preview} alt="Selected X-ray preview" />}
              <strong>{file ? file.name : "Drop an X-ray here or click to choose"}</strong>
              <span className="help">
                {isDicom ? "DICOM file chosen. The backend converts it, so you'll see it after analysis."
                  : file ? "Click to choose a different file" : "Frontal (PA/AP) chest X-ray of a child, 0–18 years"}
              </span>
              <input ref={inputRef} id="file" type="file" hidden accept=".png,.jpg,.jpeg,.dcm,.dicom"
                     onChange={(e) => pick(e.target.files?.[0])} />
            </label>
            {file && <button type="button" className="btn sm" style={{ alignSelf: "flex-start" }} onClick={clear} disabled={busy}>Remove file</button>}
          </div>
        </div>

        <div className="panel">
          <div className="panel-h"><h2>Patient details</h2><span className="chip info">Used by the age-aware model</span></div>
          <div className="panel-b form">
            <div className="field">
              <label htmlFor="pref">Patient reference <span className="help">(optional)</span></label>
              <input id="pref" type="text" maxLength={64} autoComplete="off" placeholder="e.g. hospital number"
                     value={patientRef} onChange={(e) => setPatientRef(e.target.value)} />
              <span className="help">Use a hospital number, not a name.</span>
            </div>
            <div className="row2">
              <div className="field">
                <label htmlFor="ageY">Age (years)</label>
                <input id="ageY" type="number" min="0" max="18" placeholder="0–18" value={years} onChange={(e) => setYears(e.target.value)} />
              </div>
              <div className="field">
                <label htmlFor="ageM">Months</label>
                <input id="ageM" type="number" min="0" max="11" value={months} onChange={(e) => setMonths(e.target.value)} />
              </div>
            </div>
            <fieldset className="field">
              <legend>Sex</legend>
              <div className="seg">
                <label><input type="radio" name="sex" value="M" checked={sex === "M"} onChange={() => setSex("M")} />Male</label>
                <label><input type="radio" name="sex" value="F" checked={sex === "F"} onChange={() => setSex("F")} />Female</label>
              </div>
            </fieldset>
            {error && <div className="error" role="alert">{error}</div>}
            <button className="btn primary" type="submit" disabled={busy}>{busy ? "Analyzing…" : "Analyze X-ray"}</button>
            {busy && <div className="busy"><span className="spinner" />Uploading, running the model and making the heatmap…</div>}
          </div>
        </div>
      </form>
    </section>
  );
}
