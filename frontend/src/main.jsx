import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, Database, FileUp, Home, Search } from "lucide-react";

import { BackendStatus } from "./components/BackendStatus.jsx";
import { HomePage } from "./pages/HomePage.jsx";
import { SearchPage } from "./pages/SearchPage.jsx";
import { UploadPage } from "./pages/UploadPage.jsx";
import { getBackendStatus } from "./lib/api.js";
import "./styles.css";

const pages = [
  { id: "home", label: "Home", icon: Home },
  { id: "upload", label: "Upload", icon: FileUp },
  { id: "search", label: "Search", icon: Search }
];

function readRoute() {
  return window.location.hash.replace("#/", "") || "home";
}

function App() {
  const [activePage, setActivePage] = useState(readRoute);
  const [status, setStatus] = useState({
    state: "checking",
    title: "Checking backend",
    detail: "Waiting for FastAPI",
    paths: []
  });

  useEffect(() => {
    const onHashChange = () => setActivePage(readRoute());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    let ignore = false;
    getBackendStatus().then((nextStatus) => {
      if (!ignore) {
        setStatus(nextStatus);
      }
    });
    return () => {
      ignore = true;
    };
  }, []);

  const page = useMemo(() => {
    if (activePage === "upload") {
      return <UploadPage backendStatus={status} />;
    }
    if (activePage === "search") {
      return <SearchPage />;
    }
    return <HomePage backendStatus={status} />;
  }, [activePage, status]);

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Primary navigation">
        <a className="brand" href="#/home">
          <Database aria-hidden="true" size={24} />
          <span>Vector Search</span>
        </a>
        <nav className="nav-list">
          {pages.map((item) => {
            const Icon = item.icon;
            return (
              <a
                className={activePage === item.id ? "nav-item active" : "nav-item"}
                href={`#/${item.id}`}
                key={item.id}
              >
                <Icon aria-hidden="true" size={18} />
                <span>{item.label}</span>
              </a>
            );
          })}
        </nav>
        <div className="sidebar-footer">
          <Activity aria-hidden="true" size={16} />
          <span>{status.title}</span>
        </div>
      </aside>
      <main className="main-panel">
        <header className="topbar">
          <div>
            <p className="eyebrow">Semantic retrieval workspace</p>
            <h1>Vector Search Engine</h1>
          </div>
          <BackendStatus status={status} />
        </header>
        {page}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
