import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <section className="view">
      <div className="state">
        <h1>Page not found</h1>
        <Link className="btn" to="/worklist">Go to the worklist</Link>
      </div>
    </section>
  );
}
