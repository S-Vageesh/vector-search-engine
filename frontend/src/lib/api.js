const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers
    },
    ...options
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }

  return response.json();
}

export async function getBackendStatus() {
  try {
    const spec = await request("/openapi.json");
    const paths = Object.keys(spec.paths || {});
    return {
      state: "online",
      title: "Backend online",
      detail: `${spec.info?.title || "FastAPI"} exposes ${paths.length} endpoint paths`,
      paths
    };
  } catch (error) {
    return {
      state: "offline",
      title: "Backend offline",
      detail: error.message,
      paths: []
    };
  }
}

export function searchDocuments(query, topK) {
  return request("/search", {
    method: "POST",
    body: JSON.stringify({ query, top_k: topK })
  });
}

export async function uploadTextDocument(file, backendPaths = []) {
  const uploadPath = ["/upload", "/documents/upload", "/ingest", "/documents/ingest"].find(
    (path) => backendPaths.includes(path)
  );

  if (!uploadPath) {
    throw new Error("No upload endpoint is currently exposed by the FastAPI backend.");
  }

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}${uploadPath}`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Upload failed with ${response.status}`);
  }

  return response.json();
}
