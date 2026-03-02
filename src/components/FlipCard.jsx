import LatexRenderer from "./LatexRenderer.jsx";
import { QUALITY_BTNS } from "../constants/index.js";

export default function FlipCard({ current, flipped, setFlipped, activeDeckData, cardResources, learnMoreOpen, setLearnMoreOpen, handleRate }) {
  return (
    <>
      {/* 3D flip card */}
      <div className="card-scene" style={{ height: 300, marginBottom: 20 }} onClick={() => setFlipped(f => !f)}>
        <div className={`card-inner ${flipped ? "flipped" : ""}`}>
          <div className="card-face card-front-face">
            <div className="card-question-label">question</div>
            <div style={{ fontSize: 22, lineHeight: 1.6, color: "var(--th-text)", flex: 1, display: "flex", alignItems: "center", padding: "16px 0" }}>
              <LatexRenderer text={current.front} serif={true} />
            </div>
            <div className="card-tap-hint">tap to reveal ↗</div>
          </div>
          <div className="card-face card-back-face" style={{ border: `1px solid ${activeDeckData.color}30`, justifyContent: "space-between" }}>
            <div style={{ fontSize: 10, color: activeDeckData.color, letterSpacing: "0.12em", textTransform: "uppercase", opacity: 0.7 }}>answer</div>
            <div style={{ fontSize: 13, lineHeight: 1.9, color: "var(--th-text-answer)", flex: 1, overflowY: "auto", padding: "14px 0" }}>
              <LatexRenderer text={current.back} />
            </div>
            <div className="card-sm2-stats">interval {current.interval}d · ef {current.ef.toFixed(2)} · rep {current.repetitions}</div>
          </div>
        </div>
      </div>

      {/* Learn More */}
      {flipped && cardResources[current.id]?.length > 0 && (
        <div style={{ marginBottom: 14 }}>
          <button className="learn-more-btn" onClick={() => setLearnMoreOpen(o => !o)}>
            {learnMoreOpen ? "▾ learn more" : "▸ learn more"}
          </button>
          {learnMoreOpen && (
            <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
              {cardResources[current.id].map((r, i) => {
                const typeColor = { paper: "#60a5fa", blog: "#4ade80", wiki: "#a78bfa", notes: "#fb923c" }[r.type] ?? "#64748b";
                return (
                  <a key={i} href={r.url} target="_blank" rel="noopener noreferrer" className="resource-link">
                    <span style={{ fontSize: 9, background: `${typeColor}20`, color: typeColor, border: `1px solid ${typeColor}40`, borderRadius: 4, padding: "2px 6px", letterSpacing: "0.08em", flexShrink: 0 }}>{r.type}</span>
                    <span style={{ fontSize: 12, color: "var(--th-text-muted)", lineHeight: 1.4 }}>{r.title}</span>
                  </a>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Rating buttons or recall hint */}
      {flipped ? (
        <div>
          <div className="recall-label">how well did you recall?</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8 }}>
            {QUALITY_BTNS.map(btn => (
              <button key={btn.q} className="rate-btn" onClick={() => handleRate(btn.q)} style={{ background: `${btn.color}15`, border: `1px solid ${btn.color}35`, color: btn.color }}>
                <div style={{ fontWeight: 500 }}>{btn.label}</div>
                <div style={{ fontSize: 10, opacity: 0.6, marginTop: 3 }}>{btn.sub}</div>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="recall-hint">recall your answer, then tap the card</div>
      )}
    </>
  );
}
