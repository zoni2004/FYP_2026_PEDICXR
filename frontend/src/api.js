// Every call to the backend lives in this file.
// Change the address with VITE_API_URL in frontend/.env.local if the backend runs somewhere else.
export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8010").replace(/\/$/, "");

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(API_URL + path, options);
  } catch {
    throw new Error(`Can't reach the backend at ${API_URL}. Start the server and try again.`);
  }
  if (!res.ok) {
    let message = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((d) => `${d.loc?.at(-1) ?? "field"}: ${d.msg}`).join("; ");
    } catch {
      /* response had no JSON body */
    }
    const err = new Error(message);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

const json = (method, body) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export function getHealth() {
  return request("/api/health");
}

export function listStudies({ status, q } = {}) {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  if (q) params.set("q", q);
  const qs = params.toString();
  return request(`/api/studies${qs ? `?${qs}` : ""}`);
}

export function getStudy(id) {
  return request(`/api/studies/${id}`);
}

export function createStudy({ file, ageMonths, sex, patientRef }) {
  const form = new FormData();
  form.append("image", file);
  form.append("age_months", String(ageMonths));
  form.append("sex", sex);
  if (patientRef) form.append("patient_ref", patientRef);
  return request("/api/studies", { method: "POST", body: form });
}

export function saveReport(id, text, status) {
  return request(`/api/studies/${id}/report`, json("PUT", { text, status }));
}

export function sendFeedback(id, finding, agree) {
  return request(`/api/studies/${id}/feedback`, json("POST", { finding, agree }));
}

export function getModelInfo() {
  return request("/api/model-info");
}

export const fileUrl = (path) => (path ? API_URL + path : null);
export const pdfUrl = (id) => `${API_URL}/api/studies/${id}/pdf`;
