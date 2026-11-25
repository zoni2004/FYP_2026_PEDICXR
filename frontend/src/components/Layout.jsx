import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { getHealth } from "../api.js";

const icons = {
  worklist: <path d="M4 6h16M4 12h16M4 18h10" />,
  new: <path d="M12 5v14M5 12h14" />,
  model: <path d="M4 19V9M10 19V5M16 19v-7M22 19H2" />,
};
const Icon = ({ name }) => <svg viewBox="0 0 24 24" aria-hidden="true">{icons[name]}</svg>;

function BackendStatus() {
  const [state, setState] = useState({ kind: "checking" });

  useEffect(() => {
    let alive = true;
    const check = () =>
      getHealth()
        .then((h) => alive && setState({ kind: h.model_loaded ? "ok" : "nomodel", model: h.model }))
        .catch(() => alive && setState({ kind: "down" }));
    check();
    const t = setInterval(check, 15000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  if (state.kind === "checking") return <span className="chip locked">Checking backend…</span>;
  if (state.kind === "ok") return <span className="chip ok" title="Backend is running and the AI model is loaded">Backend connected · {state.model}</span>;
  if (state.kind === "nomodel") return <span className="chip warn">Backend up, model not loaded</span>;
  return <span className="chip crit" title="Start the backend server on port 8010">Backend offline</span>;
}

export default function Layout() {
  return (
    <div className="app">
      <aside className="side">
        <div className="brand">PediCXR<span>Pediatric X-ray report assistant</span></div>
        <nav aria-label="Main">
          <NavLink to="/worklist"><Icon name="worklist" />Worklist</NavLink>
          <NavLink to="/new"><Icon name="new" />New study</NavLink>
          <NavLink to="/model"><Icon name="model" />Model info</NavLink>
        </nav>
        <div className="side-foot">Research prototype. Not for clinical use.</div>
      </aside>
      <main>
        <header className="top">
          <BackendStatus />
          <div className="user">
            <span className="avatar" aria-hidden="true">DR</span>
          </div>
        </header>
        <Outlet />
      </main>
    </div>
  );
}
