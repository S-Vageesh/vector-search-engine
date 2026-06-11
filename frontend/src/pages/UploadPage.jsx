import { FileText, FileUp, Upload } from "lucide-react";
import { useMemo, useState } from "react";

import { uploadTextDocument } from "../lib/api.js";

export function UploadPage({ backendStatus }) {
  const [files, setFiles] = useState([]);
  const [preview, setPreview] = useState("");
  const [state, setState] = useState("idle");
  const [message, setMessage] = useState("");
  const [uploadResult, setUploadResult] = useState(null);

  const hasUploadEndpoint = useMemo(
    () =>
      ["/upload", "/documents/upload", "/ingest", "/documents/ingest"].some((path) =>
        backendStatus.paths.includes(path)
      ),
    [backendStatus.paths]
  );

  async function onFileChange(event) {
    const nextFiles = Array.from(event.target.files || []);
    setFiles(nextFiles);
    setState("idle");
    setMessage("");
    setUploadResult(null);
    if (!nextFiles.length) {
      setPreview("");
      return;
    }
    const previews = await Promise.all(
      nextFiles.slice(0, 3).map(async (file) => {
        const text = await file.text();
        return `--- ${file.name} ---\n${text}`;
      })
    );
    setPreview(previews.join("\n\n"));
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (!files.length) {
      return;
    }

    setState("loading");
    setMessage("");
    setUploadResult(null);
    try {
      const result = await uploadTextDocument(files, backendStatus.paths);
      setState("done");
      setUploadResult(result);
      setMessage(uploadSummaryMessage(result));
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
          <span>{uploadLabel(files)}</span>
          <small>{uploadDetail(files)}</small>
        </label>
        <input
          accept=".txt,text/plain"
          id="document-file"
          multiple
          onChange={onFileChange}
          type="file"
        />
        <button
          type="submit"
          disabled={!files.length || state === "loading" || !hasUploadEndpoint}
        >
          <Upload aria-hidden="true" size={18} />
          {state === "loading" ? "Uploading" : "Upload"}
        </button>
        {!hasUploadEndpoint && (
          <p className="notice idle">
            No upload endpoint was found in the current FastAPI schema.
          </p>
        )}
        {message && <p className={`notice ${state}`}>{message}</p>}
        {uploadResult && (
          <div className="upload-summary" aria-live="polite">
            <div>
              <strong>{uploadResult.files_processed.length}</strong>
              <span>processed</span>
            </div>
            <div>
              <strong>{uploadResult.files_failed.length}</strong>
              <span>failed</span>
            </div>
            <div>
              <strong>{uploadResult.total_documents_ingested}</strong>
              <span>ingested</span>
            </div>
          </div>
        )}
      </form>

      <article className="preview-panel">
        <header>
          <FileText aria-hidden="true" size={20} />
          <h2>Preview</h2>
        </header>
        <pre>
          {preview ||
            "Select one or more text files to preview their contents."}
        </pre>
      </article>
    </section>
  );
}

function uploadLabel(files) {
  if (!files.length) {
    return "Choose .txt files";
  }
  if (files.length === 1) {
    return files[0].name;
  }
  return `${files.length} files selected`;
}

function uploadDetail(files) {
  if (!files.length) {
    return "Plain text documents only";
  }
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
  return `${totalBytes} bytes total`;
}

function uploadSummaryMessage(result) {
  const processed = result.files_processed.length;
  const failed = result.files_failed.length;
  if (failed === 0) {
    return `${processed} ${processed === 1 ? "file" : "files"} uploaded.`;
  }
  return `${processed} uploaded, ${failed} failed.`;
}
