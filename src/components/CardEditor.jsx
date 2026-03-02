import { useState, useEffect } from "react";
import LatexRenderer from "./LatexRenderer.jsx";

export default function CardEditor({ card, onSave, onClose }) {
  const [front, setFront] = useState(card.front);
  const [back, setBack] = useState(card.back);

  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const unchanged = front === card.front && back === card.back;
  const saveDisabled = !front.trim() || unchanged;

  return (
    <div
      className="card-editor-modal"
      onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="card-editor-box">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ fontSize: 11, color: "var(--th-text-subtle)", letterSpacing: "0.1em", textTransform: "uppercase" }}>edit card</div>
          <button
            onClick={onClose}
            style={{ background: "none", border: "none", color: "var(--th-text-faint)", cursor: "pointer", fontSize: 16, padding: "2px 6px", lineHeight: 1, fontFamily: "inherit" }}
          >✕</button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <div style={{ fontSize: 10, color: "var(--th-text-subtle)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>front</div>
            <div className="editor-row">
              <textarea
                className="input-field"
                value={front}
                onChange={e => setFront(e.target.value)}
                rows={3}
                style={{ resize: "vertical", fontFamily: "'DM Mono', monospace", fontSize: 12, lineHeight: 1.6 }}
                placeholder="Question or concept..."
              />
              <div className="editor-preview">
                {front.trim() ? <LatexRenderer text={front} /> : <span style={{ color: "var(--th-text-ghost)" }}>preview</span>}
              </div>
            </div>
          </div>

          <div>
            <div style={{ fontSize: 10, color: "var(--th-text-subtle)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>back</div>
            <div className="editor-row">
              <textarea
                className="input-field"
                value={back}
                onChange={e => setBack(e.target.value)}
                rows={4}
                style={{ resize: "vertical", fontFamily: "'DM Mono', monospace", fontSize: 12, lineHeight: 1.6 }}
                placeholder="Answer or definition..."
              />
              <div className="editor-preview">
                {back.trim() ? <LatexRenderer text={back} /> : <span style={{ color: "var(--th-text-ghost)" }}>preview</span>}
              </div>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 10, paddingTop: 4 }}>
          <button
            className="btn-primary"
            disabled={saveDisabled}
            onClick={() => onSave(front.trim(), back)}
          >save</button>
          <button className="btn-ghost" onClick={onClose}>cancel</button>
        </div>
      </div>
    </div>
  );
}
