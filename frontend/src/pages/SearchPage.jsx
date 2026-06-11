import { Clock3, Database, FileText, Search, SlidersHorizontal } from "lucide-react";
import { Fragment, useState } from "react";

import { searchDocuments } from "../lib/api.js";

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState([]);
  const [searchMeta, setSearchMeta] = useState(null);
  const [state, setState] = useState("idle");
  const [message, setMessage] = useState("");

  async function onSubmit(event) {
    event.preventDefault();
    setState("loading");
    setMessage("");
    try {
      const response = await searchDocuments(query.trim(), Number(topK));
      const nextResults = Array.isArray(response) ? response : response.results;
      setResults(nextResults);
      setSearchMeta(
        Array.isArray(response)
          ? null
          : {
              documentCount: response.document_count,
              latencyMs: response.latency_ms
            }
      );
      setState("done");
      setMessage(nextResults.length ? "" : "No results returned.");
    } catch (error) {
      setResults([]);
      setSearchMeta(null);
      setState("error");
      setMessage(error.message);
    }
  }

  return (
    <section className="work-surface">
      <form className="search-form" onSubmit={onSubmit}>
        <label htmlFor="query">Search query</label>
        <div className="query-row">
          <Search aria-hidden="true" size={20} />
          <input
            id="query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Find documents about vector indexing"
            required
          />
        </div>
        <div className="control-row">
          <label className="topk-control" htmlFor="top-k">
            <SlidersHorizontal aria-hidden="true" size={18} />
            <span>Top K</span>
            <input
              id="top-k"
              type="number"
              min="1"
              max="100"
              value={topK}
              onChange={(event) => setTopK(event.target.value)}
            />
          </label>
          <button type="submit" disabled={state === "loading"}>
            <Search aria-hidden="true" size={18} />
            {state === "loading" ? "Searching" : "Search"}
          </button>
        </div>
      </form>

      {message && <p className={`notice ${state}`}>{message}</p>}

      {searchMeta && (
        <div className="search-summary" aria-live="polite">
          <div>
            <Database aria-hidden="true" size={18} />
            <span>{searchMeta.documentCount}</span>
            <p>documents indexed</p>
          </div>
          <div>
            <Clock3 aria-hidden="true" size={18} />
            <span>{formatLatency(searchMeta.latencyMs)}</span>
            <p>search latency</p>
          </div>
        </div>
      )}

      <div className="results-list" aria-live="polite">
        {results.map((result) => (
          <article className="result-card" key={result.document_id}>
            <header>
              <div className="result-title">
                <FileText aria-hidden="true" size={18} />
                <span>{result.filename || `Document #${result.document_id}`}</span>
              </div>
              <strong>{formatSimilarity(result.similarity)}</strong>
            </header>
            <p>{highlightQueryTerms(result.text, query)}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function formatSimilarity(value) {
  const percentage = Number(value) * 100;
  return `${percentage.toFixed(1)}%`;
}

function formatLatency(value) {
  return `${Number(value).toFixed(1)} ms`;
}

function highlightQueryTerms(text, query) {
  const terms = Array.from(
    new Set(
      query
        .trim()
        .split(/\s+/)
        .filter((term) => term.length > 1)
        .map(escapeRegExp)
    )
  );

  if (!terms.length) {
    return text;
  }

  const pattern = new RegExp(`(${terms.join("|")})`, "gi");
  const exactTerm = new RegExp(`^(${terms.join("|")})$`, "i");
  return text.split(pattern).map((part, index) =>
    exactTerm.test(part) ? (
      <mark key={`${part}-${index}`}>{part}</mark>
    ) : (
      <Fragment key={`${part}-${index}`}>{part}</Fragment>
    )
  );
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
