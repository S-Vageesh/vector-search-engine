import { FileText, FileUp, Upload } from "lucide-react";
import { useMemo, useState } from "react";

import { uploadTextDocument } from "../lib/api.js";

export function UploadPage({ backendStatus }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [state, setState] = useState("idle");
  const [message, setMessage] = useState("");

  const hasUploadEndpoint = useMemo(
    () =>
      ["/upload", "/documents/upload", "/ingest", "/documents/ingest"].some((path) =>
        backendStatus.paths.includes(path)
      ),
    [backendStatus.paths]
  );

  async function onFileChange(event) {
    const nextFile = event.target.files?.[0] || null;
    setFile(nextFile);
    setState("idle");
    setMessage("");
    if (!nextFile) {
      setPreview("");
      return;
    }
    setPreview(await nextFile.text());
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (!file) {
      return;
    }

    setState("loading");
    setMessage("");
    try {
      await uploadTextDocument(file, backendStatus.paths);
      setState("done");
      setMessage(`${file.name} uploaded.`);
    } catch (error) {
      setState("error");
      setMessage(error.message);
    }
  }

  return (
    <section className="work-surface upload-layout">
      <form className="upload-form" onSubmit={onSubmit}>
        <label className="drop-zone" htmlFor="document-file">
          <FileUp aria-hidden="true" size={28} />
          <span>{file ? file.name : "Choose a .txt file"}</span>
          <small>{file ? `${file.size} bytes` : "Plain text documents only"}</small>
        </label>
        <input
          accept=".txt,text/plain"
          id="document-file"
          onChange={onFileChange}
          type="file"
        />
        <button type="submit" disabled={!file || state === "loading" || !hasUploadEndpoint}>
          <Upload aria-hidden="true" size={18} />
          {state === "loading" ? "Uploading" : "Upload"}
        </button>
        {!hasUploadEndpoint && (
          <p className="notice idle">
            No upload endpoint was found in the current FastAPI schema.
          </p>
        )}
        {message && <p className={`notice ${state}`}>{message}</p>}
      </form>

      <article className="preview-panel">
        <header>
          <FileText aria-hidden="true" size={20} />
          <h2>Preview</h2>
        </header>
        <pre>{preview || "Select a text file to preview its contents."}</pre>
      </article>
    </section>
  );
}
