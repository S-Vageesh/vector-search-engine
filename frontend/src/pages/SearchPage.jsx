import { Search, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import { searchDocuments } from "../lib/api.js";

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState([]);
  const [state, setState] = useState("idle");
  const [message, setMessage] = useState("");

  async function onSubmit(event) {
    event.preventDefault();
    setState("loading");
    setMessage("");
    try {
      const nextResults = await searchDocuments(query.trim(), Number(topK));
      setResults(nextResults);
      setState("done");
      setMessage(nextResults.length ? "" : "No results returned.");
    } catch (error) {
      setResults([]);
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

      <div className="results-list" aria-live="polite">
        {results.map((result) => (
          <article className="result-card" key={result.document_id}>
            <header>
              <span>Document #{result.document_id}</span>
              <strong>{Number(result.similarity).toFixed(4)}</strong>
            </header>
            <p>{result.text}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
