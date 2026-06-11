import { Gauge, Layers3, ListChecks } from "lucide-react";

export function HomePage({ backendStatus }) {
  return (
    <section className="page-grid">
      <article className="overview-panel">
        <div>
          <p className="eyebrow">Current backend</p>
          <h2>{backendStatus.title}</h2>
          <p>{backendStatus.detail}</p>
        </div>
        <div className="metric-row">
          <div>
            <span>{backendStatus.paths.length}</span>
            <p>endpoint paths</p>
          </div>
          <div>
            <span>{backendStatus.paths.includes("/search") ? "Ready" : "Missing"}</span>
            <p>search API</p>
          </div>
        </div>
      </article>

      <article className="quick-panel">
        <h2>Workspace</h2>
        <a href="#/upload">
          <Layers3 aria-hidden="true" size={20} />
          Upload text documents
        </a>
        <a href="#/search">
          <Gauge aria-hidden="true" size={20} />
          Query semantic vectors
        </a>
        <a href="#/home">
          <ListChecks aria-hidden="true" size={20} />
          Review backend readiness
        </a>
      </article>
    </section>
  );
}
