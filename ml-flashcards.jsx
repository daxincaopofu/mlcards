import { useState, useEffect, useCallback } from "react";

// ── SM-2 Algorithm ──────────────────────────────────────────────────────────
function sm2(card, quality) {
  // quality: 0=Blackout, 1=Hard, 2=Good, 3=Easy (mapped from user buttons)
  let { ef, interval, repetitions } = card;
  if (quality < 1) {
    repetitions = 0;
    interval = 1;
  } else {
    if (repetitions === 0) interval = 1;
    else if (repetitions === 1) interval = 6;
    else interval = Math.round(interval * ef);
    repetitions += 1;
    ef = Math.max(1.3, ef + 0.1 - (3 - quality) * (0.08 + (3 - quality) * 0.02));
  }
  const nextReview = Date.now() + interval * 86400000;
  return { ef, interval, repetitions, nextReview };
}

// ── Seed Cards ───────────────────────────────────────────────────────────────
const SEED_CARDS = [
  // Concepts & Theory
  { id: 1, category: "concepts", front: "What is the bias-variance tradeoff?", back: "Bias error comes from wrong assumptions in the learning algorithm (underfitting). Variance error comes from sensitivity to small fluctuations in training data (overfitting). Increasing model complexity reduces bias but increases variance. The goal is to find the sweet spot that minimizes total error." },
  { id: 2, category: "concepts", front: "Explain the difference between L1 and L2 regularization.", back: "L1 (Lasso) adds |w| penalty — produces sparse solutions by driving some weights to exactly zero, useful for feature selection. L2 (Ridge) adds w² penalty — shrinks weights toward zero but rarely to exactly zero, handling multicollinearity well. Elastic Net combines both." },
  { id: 3, category: "concepts", front: "What is the kernel trick in SVMs?", back: "The kernel trick implicitly maps data into a high-dimensional feature space without explicitly computing the transformation, by replacing dot products with a kernel function K(x, x') = φ(x)·φ(x'). Common kernels: RBF, polynomial, sigmoid. This allows SVMs to find non-linear decision boundaries efficiently." },
  { id: 4, category: "concepts", front: "What is batch normalization and why does it help?", back: "Batch normalization normalizes layer inputs to have zero mean and unit variance per mini-batch, then applies learnable scale (γ) and shift (β). Benefits: reduces internal covariate shift, allows higher learning rates, acts as slight regularization, reduces sensitivity to initialization." },
  { id: 5, category: "concepts", front: "What is the vanishing gradient problem?", back: "In deep networks, gradients computed during backpropagation can become exponentially small as they flow back through many layers, making early layers learn very slowly or not at all. Caused by saturating activations (sigmoid, tanh). Solutions: ReLU activations, residual connections, gradient clipping, batch norm." },
  { id: 6, category: "concepts", front: "Explain attention mechanism in transformers.", back: "Attention computes a weighted sum of values based on the compatibility between queries and keys: Attention(Q,K,V) = softmax(QKᵀ/√d_k)V. Each token can attend to all other tokens. Multi-head attention runs this in parallel across multiple representation subspaces, then concatenates and projects the results." },
  // Math & Derivations
  { id: 7, category: "math", front: "Derive the gradient of cross-entropy loss w.r.t. softmax input (logits).", back: "For true label y (one-hot) and softmax output p:\nL = -Σ yᵢ log(pᵢ)\n∂L/∂zⱼ = pⱼ - yⱼ\n\nThis elegant result comes from the chain rule: ∂L/∂zⱼ = Σᵢ (∂L/∂pᵢ)(∂pᵢ/∂zⱼ), where ∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ - pⱼ). The softmax Jacobian and log loss cancel nicely." },
  { id: 8, category: "math", front: "What is the VC dimension and why does it matter?", back: "VC dimension is the largest set of points that a hypothesis class H can shatter (correctly classify for all labelings). It bounds generalization error: with probability ≥ 1-δ, error ≤ empirical error + O(√(d/n·log(n/d) + log(1/δ)/n)). Higher VC dimension → more expressive but needs more data to generalize." },
  { id: 9, category: "math", front: "Derive the update rule for gradient descent with momentum.", back: "Standard: w ← w - α∇L\nMomentum adds a velocity term:\nvₜ = βvₜ₋₁ + ∇Lₜ\nwₜ = wₜ₋₁ - αvₜ\n\nExpanding: vₜ = Σᵢ βⁱ ∇Lₜ₋ᵢ — exponentially weighted moving average of past gradients. β≈0.9 is common. Accelerates along consistent gradient directions, dampens oscillations." },
  { id: 10, category: "math", front: "What is KL divergence and how is it used in VAEs?", back: "KL(P||Q) = Σ P(x) log(P(x)/Q(x)) — measures how much P diverges from Q. Always ≥ 0.\n\nIn VAEs, ELBO = E[log p(x|z)] - KL(q(z|x)||p(z)). The KL term regularizes the encoder to keep the learned posterior q(z|x) close to the prior p(z)=N(0,I), enabling smooth latent space interpolation." },
  { id: 11, category: "math", front: "Explain the reparameterization trick in VAEs.", back: "Problem: can't backprop through a sampling operation z ~ N(μ,σ²).\nSolution: reparameterize as z = μ + σ·ε where ε ~ N(0,1).\n\nNow z is a deterministic function of μ, σ (the learnable parameters) and ε (random but not a parameter). Gradients flow through μ and σ normally during backprop." },
  // Interview Q&A
  { id: 12, category: "interview", front: "How would you handle class imbalance in a binary classification problem?", back: "Sampling: oversample minority (SMOTE), undersample majority, or both.\nAlgorithmic: adjust class weights (class_weight='balanced'), use cost-sensitive learning.\nThreshold: tune decision threshold via precision-recall curve instead of ROC.\nMetrics: use F1, AUC-PR, Matthews Correlation Coefficient — not accuracy.\nEnsemble: BalancedBaggingClassifier, EasyEnsemble." },
  { id: 13, category: "interview", front: "What questions would you ask before building an ML model for a new problem?", back: "Business: What decision will this model inform? What's the cost of false positives vs. negatives?\nData: How much labeled data? How was it collected? Any distribution shift expected?\nBaseline: What's the current process? What threshold must the model beat?\nConstraints: Latency requirements? Interpretability needed? Retraining frequency?\nEvaluation: What metric aligns with business value?" },
  { id: 14, category: "interview", front: "Explain how you would detect and handle data leakage.", back: "Detection: suspiciously high validation performance, features correlated with target at unnaturally high levels, features that wouldn't exist at prediction time.\nPrevention: strict temporal train/val/test splits for time-series, fit all preprocessing (scalers, imputers, encoders) only on training data, apply transforms to val/test, be wary of target encoding and aggregations that use test data." },
  { id: 15, category: "interview", front: "You deploy a model and performance degrades over time. What do you do?", back: "1. Monitor: track data drift (feature distributions) and concept drift (p(y|x) changes) via PSI, KS-test, or model performance metrics.\n2. Diagnose: compare recent input distributions vs. training data.\n3. Triage: rule out bugs, upstream data pipeline changes first.\n4. Remediate: retrain on recent data, update features, or add drift detection triggers for automatic retraining.\n5. Prevent: build retraining pipelines upfront." },
];

function initCard(card) {
  return { ...card, ef: 2.5, interval: 1, repetitions: 0, nextReview: Date.now() };
}

const CATEGORY_META = {
  concepts: { label: "Concepts & Theory", color: "#4ade80", icon: "⬡" },
  math: { label: "Math & Derivations", color: "#f472b6", icon: "∑" },
  interview: { label: "Interview Q&A", color: "#60a5fa", icon: "◈" },
};

const QUALITY_BUTTONS = [
  { label: "Blackout", sub: "Complete blank", q: 0, color: "#ef4444" },
  { label: "Hard", sub: "Barely recalled", q: 1, color: "#f97316" },
  { label: "Good", sub: "With effort", q: 2, color: "#eab308" },
  { label: "Easy", sub: "Instantly", q: 3, color: "#4ade80" },
];

export default function App() {
  const [cards, setCards] = useState(() => SEED_CARDS.map(initCard));
  const [queue, setQueue] = useState([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [view, setView] = useState("home"); // home | review | stats
  const [sessionStats, setSessionStats] = useState({ reviewed: 0, easy: 0, hard: 0 });
  const [filter, setFilter] = useState("all");
  const [doneAnim, setDoneAnim] = useState(false);

  const dueCards = cards.filter(c => c.nextReview <= Date.now());
  const filteredDue = filter === "all" ? dueCards : dueCards.filter(c => c.category === filter);

  function startSession() {
    const q = [...filteredDue].sort((a, b) => a.nextReview - b.nextReview);
    setQueue(q);
    setCurrentIdx(0);
    setFlipped(false);
    setSessionStats({ reviewed: 0, easy: 0, hard: 0 });
    setView("review");
  }

  function handleRate(quality) {
    const card = queue[currentIdx];
    const updated = sm2(card, quality);
    setCards(prev => prev.map(c => c.id === card.id ? { ...c, ...updated } : c));
    setSessionStats(s => ({
      reviewed: s.reviewed + 1,
      easy: s.easy + (quality >= 2 ? 1 : 0),
      hard: s.hard + (quality < 2 ? 1 : 0),
    }));
    if (currentIdx + 1 >= queue.length) {
      setDoneAnim(true);
      setTimeout(() => { setView("home"); setDoneAnim(false); }, 1800);
    } else {
      setFlipped(false);
      setTimeout(() => setCurrentIdx(i => i + 1), 50);
    }
  }

  const current = queue[currentIdx];
  const progress = queue.length > 0 ? (currentIdx / queue.length) * 100 : 0;

  const totalCards = cards.length;
  const masteredCards = cards.filter(c => c.repetitions >= 3 && c.ef > 2.2).length;

  return (
    <div style={{ minHeight: "100vh", background: "#080c14", fontFamily: "'DM Mono', monospace", color: "#e2e8f0" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Playfair+Display:ital,wght@0,700;1,600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #0d1321; } ::-webkit-scrollbar-thumb { background: #2a3450; border-radius: 2px; }

        .card-scene { perspective: 1200px; cursor: pointer; }
        .card-inner { position: relative; width: 100%; height: 100%; transition: transform 0.55s cubic-bezier(0.4,0,0.2,1); transform-style: preserve-3d; }
        .card-inner.flipped { transform: rotateY(180deg); }
        .card-face { position: absolute; inset: 0; backface-visibility: hidden; -webkit-backface-visibility: hidden; border-radius: 16px; padding: 40px; display: flex; flex-direction: column; }
        .card-back { transform: rotateY(180deg); }

        .rate-btn { border: none; border-radius: 8px; padding: 10px 18px; cursor: pointer; font-family: 'DM Mono', monospace; font-size: 13px; font-weight: 500; transition: transform 0.15s, box-shadow 0.15s, filter 0.15s; position: relative; overflow: hidden; }
        .rate-btn:hover { transform: translateY(-2px); filter: brightness(1.15); }
        .rate-btn:active { transform: translateY(0px) scale(0.97); }

        .nav-btn { background: transparent; border: 1px solid #1e2d4a; color: #94a3b8; padding: 8px 20px; border-radius: 6px; font-family: 'DM Mono', monospace; font-size: 12px; cursor: pointer; transition: all 0.2s; letter-spacing: 0.05em; }
        .nav-btn:hover { background: #1e2d4a; color: #e2e8f0; }
        .nav-btn.active { background: #1e2d4a; color: #60a5fa; border-color: #2d4a6e; }

        .filter-pill { background: transparent; border: 1px solid #1e2d4a; color: #64748b; padding: 6px 14px; border-radius: 20px; font-family: 'DM Mono', monospace; font-size: 11px; cursor: pointer; transition: all 0.2s; }
        .filter-pill.active { background: #0f1f38; color: #60a5fa; border-color: #2d4a6e; }
        .filter-pill:hover:not(.active) { border-color: #2d4a6e; color: #94a3b8; }

        .start-btn { background: linear-gradient(135deg, #1d4ed8, #0ea5e9); border: none; color: white; padding: 14px 40px; border-radius: 10px; font-family: 'DM Mono', monospace; font-size: 14px; font-weight: 500; cursor: pointer; transition: all 0.25s; letter-spacing: 0.05em; }
        .start-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(14,165,233,0.3); }
        .start-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }

        .stat-card { background: #0d1321; border: 1px solid #1e2d4a; border-radius: 12px; padding: 20px 24px; }

        .done-overlay { position: fixed; inset: 0; background: rgba(8,12,20,0.95); display: flex; align-items: center; justify-content: center; z-index: 100; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }
        .done-check { font-size: 64px; animation: popIn 0.5s cubic-bezier(0.34,1.56,0.64,1); }
        @keyframes popIn { from { transform: scale(0) rotate(-15deg); opacity: 0 } to { transform: scale(1) rotate(0); opacity: 1 } }

        .progress-bar { height: 3px; background: #1e2d4a; border-radius: 2px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #1d4ed8, #0ea5e9); border-radius: 2px; transition: width 0.4s ease; }

        .grid-bg { position: fixed; inset: 0; pointer-events: none; background-image: linear-gradient(rgba(30,45,74,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(30,45,74,0.3) 1px, transparent 1px); background-size: 40px 40px; }
        .glow { position: fixed; width: 600px; height: 600px; border-radius: 50%; background: radial-gradient(circle, rgba(14,165,233,0.04) 0%, transparent 70%); top: -200px; right: -100px; pointer-events: none; }

        .flip-hint { font-size: 11px; color: #334155; letter-spacing: 0.08em; text-transform: uppercase; transition: color 0.2s; }
        .card-scene:hover .flip-hint { color: #475569; }

        @keyframes slideUp { from { opacity: 0; transform: translateY(20px) } to { opacity: 1; transform: translateY(0) } }
        .slide-up { animation: slideUp 0.4s ease forwards; }
      `}</style>

      <div className="grid-bg" />
      <div className="glow" />

      {doneAnim && (
        <div className="done-overlay">
          <div style={{ textAlign: "center" }}>
            <div className="done-check">✓</div>
            <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, marginTop: 16, color: "#e2e8f0" }}>Session Complete</div>
            <div style={{ color: "#64748b", fontSize: 13, marginTop: 8 }}>
              {sessionStats.reviewed} cards · {sessionStats.easy} easy · {sessionStats.hard} hard
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div style={{ borderBottom: "1px solid #1e2d4a", padding: "16px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", backdropFilter: "blur(12px)", position: "sticky", top: 0, zIndex: 10, background: "rgba(8,12,20,0.9)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 28, height: 28, background: "linear-gradient(135deg, #1d4ed8, #0ea5e9)", borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14 }}>∇</div>
          <span style={{ fontFamily: "'Playfair Display', serif", fontSize: 18, letterSpacing: "-0.02em" }}>ml<span style={{ color: "#0ea5e9" }}>cards</span></span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className={`nav-btn ${view === "home" ? "active" : ""}`} onClick={() => setView("home")}>home</button>
          <button className={`nav-btn ${view === "stats" ? "active" : ""}`} onClick={() => setView("stats")}>stats</button>
        </div>
      </div>

      <div style={{ maxWidth: 780, margin: "0 auto", padding: "40px 24px", position: "relative" }}>

        {/* HOME VIEW */}
        {view === "home" && (
          <div className="slide-up">
            <div style={{ marginBottom: 48 }}>
              <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 40, lineHeight: 1.1, marginBottom: 12 }}>
                Ready to<br /><span style={{ fontStyle: "italic", color: "#0ea5e9" }}>review?</span>
              </div>
              <div style={{ color: "#475569", fontSize: 13, lineHeight: 1.7 }}>
                SM-2 spaced repetition · {dueCards.length} cards due today · {totalCards} total
              </div>
            </div>

            {/* Stats Row */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 40 }}>
              {[
                { label: "due today", value: dueCards.length, color: dueCards.length > 0 ? "#f97316" : "#4ade80" },
                { label: "mastered", value: masteredCards, color: "#4ade80" },
                { label: "total cards", value: totalCards, color: "#60a5fa" },
              ].map(s => (
                <div key={s.label} className="stat-card" style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 32, fontWeight: 300, color: s.color, fontFamily: "'Playfair Display', serif" }}>{s.value}</div>
                  <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.08em", textTransform: "uppercase", marginTop: 4 }}>{s.label}</div>
                </div>
              ))}
            </div>

            {/* Category breakdown */}
            <div style={{ marginBottom: 32 }}>
              <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 16 }}>categories</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {Object.entries(CATEGORY_META).map(([key, meta]) => {
                  const total = cards.filter(c => c.category === key).length;
                  const due = dueCards.filter(c => c.category === key).length;
                  const mastered = cards.filter(c => c.category === key && c.repetitions >= 3 && c.ef > 2.2).length;
                  return (
                    <div key={key} style={{ background: "#0d1321", border: "1px solid #1e2d4a", borderRadius: 10, padding: "14px 20px", display: "flex", alignItems: "center", gap: 16 }}>
                      <div style={{ width: 32, height: 32, borderRadius: 8, background: `${meta.color}18`, border: `1px solid ${meta.color}30`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, color: meta.color, flexShrink: 0 }}>{meta.icon}</div>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 13, color: "#cbd5e1", marginBottom: 6 }}>{meta.label}</div>
                        <div style={{ background: "#1e2d4a", borderRadius: 3, height: 3, overflow: "hidden" }}>
                          <div style={{ width: `${(mastered / total) * 100}%`, height: "100%", background: meta.color, borderRadius: 3, transition: "width 0.6s ease" }} />
                        </div>
                      </div>
                      <div style={{ textAlign: "right", flexShrink: 0 }}>
                        <div style={{ fontSize: 13, color: due > 0 ? "#f97316" : "#475569" }}>{due} due</div>
                        <div style={{ fontSize: 11, color: "#334155" }}>{mastered}/{total}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Filter */}
            <div style={{ marginBottom: 28 }}>
              <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>review filter</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                <button className={`filter-pill ${filter === "all" ? "active" : ""}`} onClick={() => setFilter("all")}>all ({dueCards.length})</button>
                {Object.entries(CATEGORY_META).map(([key, meta]) => (
                  <button key={key} className={`filter-pill ${filter === key ? "active" : ""}`} onClick={() => setFilter(key)}>
                    {meta.icon} {key} ({dueCards.filter(c => c.category === key).length})
                  </button>
                ))}
              </div>
            </div>

            <button className="start-btn" onClick={startSession} disabled={filteredDue.length === 0}>
              {filteredDue.length === 0 ? "no cards due" : `start review  →  ${filteredDue.length} cards`}
            </button>
          </div>
        )}

        {/* REVIEW VIEW */}
        {view === "review" && current && (
          <div className="slide-up">
            {/* Progress */}
            <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 32 }}>
              <div style={{ flex: 1 }}>
                <div className="progress-bar"><div className="progress-fill" style={{ width: `${progress}%` }} /></div>
              </div>
              <div style={{ fontSize: 12, color: "#475569", flexShrink: 0 }}>{currentIdx + 1} / {queue.length}</div>
              <button style={{ background: "transparent", border: "none", color: "#475569", cursor: "pointer", fontSize: 12, padding: "4px 8px" }} onClick={() => setView("home")}>✕</button>
            </div>

            {/* Category badge */}
            <div style={{ marginBottom: 20 }}>
              <span style={{ background: `${CATEGORY_META[current.category]?.color}18`, border: `1px solid ${CATEGORY_META[current.category]?.color}30`, color: CATEGORY_META[current.category]?.color, borderRadius: 6, padding: "4px 12px", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase" }}>
                {CATEGORY_META[current.category]?.icon} {CATEGORY_META[current.category]?.label}
              </span>
            </div>

            {/* Card */}
            <div className="card-scene" style={{ height: 320, marginBottom: 24 }} onClick={() => setFlipped(f => !f)}>
              <div className={`card-inner ${flipped ? "flipped" : ""}`}>
                {/* Front */}
                <div className="card-face" style={{ background: "#0d1321", border: "1px solid #1e2d4a", justifyContent: "space-between" }}>
                  <div style={{ fontSize: 11, color: "#334155", letterSpacing: "0.1em", textTransform: "uppercase" }}>question</div>
                  <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 22, lineHeight: 1.5, color: "#e2e8f0", flex: 1, display: "flex", alignItems: "center", padding: "20px 0" }}>
                    {current.front}
                  </div>
                  <div className="flip-hint">click to reveal answer ↗</div>
                </div>
                {/* Back */}
                <div className="card-face card-back" style={{ background: "#0a1628", border: "1px solid #2d4a6e", justifyContent: "space-between" }}>
                  <div style={{ fontSize: 11, color: "#0ea5e9", letterSpacing: "0.1em", textTransform: "uppercase" }}>answer</div>
                  <div style={{ fontSize: 13.5, lineHeight: 1.75, color: "#cbd5e1", flex: 1, overflowY: "auto", padding: "16px 0", whiteSpace: "pre-wrap" }}>
                    {current.back}
                  </div>
                  <div style={{ fontSize: 11, color: "#334155", letterSpacing: "0.05em" }}>interval: {current.interval}d · ef: {current.ef.toFixed(2)}</div>
                </div>
              </div>
            </div>

            {/* Rating buttons */}
            {flipped ? (
              <div>
                <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 14, textAlign: "center" }}>how well did you recall?</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
                  {QUALITY_BUTTONS.map(btn => (
                    <button key={btn.q} className="rate-btn" onClick={() => handleRate(btn.q)} style={{ background: `${btn.color}18`, border: `1px solid ${btn.color}40`, color: btn.color }}>
                      <div>{btn.label}</div>
                      <div style={{ fontSize: 10, opacity: 0.7, marginTop: 3 }}>{btn.sub}</div>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ textAlign: "center", color: "#334155", fontSize: 12, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                think about your answer, then flip
              </div>
            )}
          </div>
        )}

        {/* STATS VIEW */}
        {view === "stats" && (
          <div className="slide-up">
            <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 36, marginBottom: 8 }}>Your <span style={{ fontStyle: "italic", color: "#0ea5e9" }}>progress</span></div>
            <div style={{ color: "#475569", fontSize: 13, marginBottom: 40 }}>SM-2 card health overview</div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 32 }}>
              {[
                { label: "Total Cards", value: totalCards, color: "#60a5fa" },
                { label: "Due Today", value: dueCards.length, color: "#f97316" },
                { label: "Mastered (≥3 reps)", value: masteredCards, color: "#4ade80" },
                { label: "Avg. EF", value: (cards.reduce((s,c) => s + c.ef, 0) / cards.length).toFixed(2), color: "#f472b6" },
              ].map(s => (
                <div key={s.label} className="stat-card">
                  <div style={{ fontSize: 36, fontWeight: 300, color: s.color, fontFamily: "'Playfair Display', serif" }}>{s.value}</div>
                  <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.08em", textTransform: "uppercase", marginTop: 4 }}>{s.label}</div>
                </div>
              ))}
            </div>

            <div style={{ marginBottom: 24 }}>
              <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 16 }}>all cards</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {cards.map(c => {
                  const meta = CATEGORY_META[c.category];
                  const due = c.nextReview <= Date.now();
                  const mastered = c.repetitions >= 3 && c.ef > 2.2;
                  return (
                    <div key={c.id} style={{ background: "#0d1321", border: "1px solid #1e2d4a", borderRadius: 8, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12 }}>
                      <div style={{ width: 6, height: 6, borderRadius: "50%", background: mastered ? "#4ade80" : due ? "#f97316" : "#1e2d4a", flexShrink: 0 }} />
                      <div style={{ flex: 1, fontSize: 12, color: "#94a3b8", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.front}</div>
                      <div style={{ fontSize: 10, color: meta.color, flexShrink: 0 }}>{meta.icon}</div>
                      <div style={{ fontSize: 10, color: "#475569", flexShrink: 0, minWidth: 60, textAlign: "right" }}>
                        {due ? <span style={{ color: "#f97316" }}>due</span> : `in ${Math.ceil((c.nextReview - Date.now()) / 86400000)}d`}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
