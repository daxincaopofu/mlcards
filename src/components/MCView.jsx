import LatexRenderer from "./LatexRenderer.jsx";

export default function MCView({ current, mcState, handleMcSelect, handleMcNext, loadMcChoices, activeDeck, getDeck, sessionStats, qIdx, queueLength }) {
  return (
    <div>
      {/* Question */}
      <div className="mc-question-box">
        <div style={{ fontSize: 20, lineHeight: 1.6, color: "var(--th-text)", width: "100%" }}>
          <LatexRenderer text={current.front} serif={true} />
        </div>
      </div>

      {/* Choices */}
      {mcState?.loading ? (
        <div style={{ textAlign: "center", padding: "32px 0", color: "var(--th-text-muted)" }}>
          <span className="spin" style={{ fontSize: 20, marginRight: 10 }}>◉</span>
          <span style={{ fontSize: 12, letterSpacing: "0.08em" }}>generating choices...</span>
        </div>
      ) : mcState?.error ? (
        <div style={{ textAlign: "center", padding: "24px", color: "#ef4444", fontSize: 12 }}>
          Failed to generate choices.
          <button className="btn-ghost" style={{ marginLeft: 12, fontSize: 11 }} onClick={() => loadMcChoices(current, getDeck(activeDeck).cards, true)}>retry</button>
        </div>
      ) : mcState?.choices ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {mcState.choices.map((choice, idx) => {
            const isSelected = mcState.selected === idx;
            const isRevealed = mcState.selected !== null;
            const isCorrect = choice.correct;
            let bg = "var(--th-bg-card)", border = "var(--th-border-soft)", color = "var(--th-text-muted)";
            if (isRevealed && isCorrect) { bg = "#4ade8015"; border = "#4ade8050"; color = "#4ade80"; }
            else if (isRevealed && isSelected && !isCorrect) { bg = "#ef444415"; border = "#ef444450"; color = "#ef4444"; }
            return (
              <button key={idx} className="mc-choice" onClick={() => handleMcSelect(idx)}
                style={{ background: bg, border: `1px solid ${border}`, borderRadius: 10, padding: "16px 20px", textAlign: "left", cursor: isRevealed ? "default" : "pointer", transition: "all 0.2s", fontFamily: "'DM Mono', monospace", color, width: "100%" }}>
                <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                  <span style={{ width: 22, height: 22, borderRadius: "50%", border: `1px solid ${border}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, flexShrink: 0, marginTop: 2, color }}>
                    {isRevealed && isCorrect ? "✓" : isRevealed && isSelected ? "✗" : String.fromCharCode(65 + idx)}
                  </span>
                  <div style={{ fontSize: 13, lineHeight: 1.7 }}>
                    <LatexRenderer text={choice.text} />
                  </div>
                </div>
              </button>
            );
          })}
          {mcState.selected !== null && (
            <div style={{ marginTop: 8, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 12, color: mcState.choices[mcState.selected].correct ? "#4ade80" : "#ef4444" }}>
                {mcState.choices[mcState.selected].correct ? "✓ Correct" : "✗ Incorrect"}
                <span style={{ color: "var(--th-text-muted)", marginLeft: 8 }}>
                  {sessionStats.reviewed + 1 > 0 ? `${sessionStats.correct + (mcState.choices[mcState.selected].correct ? 1 : 0)}/${sessionStats.reviewed + 1} this session` : ""}
                </span>
              </span>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn-ghost" style={{ fontSize: 11 }}
                  onClick={() => loadMcChoices(current, getDeck(activeDeck).cards, true)}
                  title="Regenerate distractors for this card">
                  ↺ regen
                </button>
                <button className="btn-primary" style={{ fontSize: 12, padding: "8px 22px" }}
                  onClick={() => handleMcNext(mcState.choices[mcState.selected].correct)}>
                  {qIdx + 1 >= queueLength ? "finish" : "next →"}
                </button>
              </div>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
