import { useState, useEffect, useCallback, useRef } from "react";

// ── KaTeX LaTeX Renderer ──────────────────────────────────────────────────────
let katexLoaded = false;
let katexLoadPromise = null;

function loadKatex() {
  if (katexLoaded) return Promise.resolve();
  if (katexLoadPromise) return katexLoadPromise;
  katexLoadPromise = new Promise((resolve, reject) => {
    // Load CSS
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css";
    document.head.appendChild(link);
    // Load JS
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js";
    script.onload = () => { katexLoaded = true; resolve(); };
    script.onerror = reject;
    document.head.appendChild(script);
  });
  return katexLoadPromise;
}

// Splits text into plain and LaTeX segments, renders each appropriately
function LatexRenderer({ text, style = {}, serif = false }) {
  const [ready, setReady] = useState(katexLoaded);
  useEffect(() => { if (!ready) loadKatex().then(() => setReady(true)).catch(() => setReady(true)); }, []);

  if (!ready) return <span style={style}>{text}</span>;

  // Parse $$...$$ (display) and $...$ (inline) segments
  const segments = [];
  const re = /(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) segments.push({ type: "text", content: text.slice(last, m.index) });
    const isDisplay = m[0].startsWith("$$");
    const inner = isDisplay ? m[0].slice(2, -2) : m[0].slice(1, -1);
    segments.push({ type: "latex", display: isDisplay, content: inner });
    last = m.index + m[0].length;
  }
  if (last < text.length) segments.push({ type: "text", content: text.slice(last) });

  return (
    <span style={style}>
      {segments.map((seg, i) => {
        if (seg.type === "text") {
          return (
            <span key={i} style={{ whiteSpace: "pre-wrap", fontFamily: serif ? "'Cormorant Garamond', serif" : "inherit" }}>
              {seg.content}
            </span>
          );
        }
        try {
          const html = window.katex.renderToString(seg.content, {
            displayMode: seg.display,
            throwOnError: false,
            output: "html",
          });
          return (
            <span key={i}
              style={{ display: seg.display ? "block" : "inline", textAlign: seg.display ? "center" : "inherit", margin: seg.display ? "10px 0" : "0 2px" }}
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
        } catch {
          return <span key={i}>{seg.content}</span>;
        }
      })}
    </span>
  );
}

// ── SM-2 ─────────────────────────────────────────────────────────────────────
function sm2(card, quality) {
  let { ef, interval, repetitions } = card;
  if (quality < 1) { repetitions = 0; interval = 1; }
  else {
    if (repetitions === 0) interval = 1;
    else if (repetitions === 1) interval = 6;
    else interval = Math.round(interval * ef);
    repetitions += 1;
    ef = Math.max(1.3, ef + 0.1 - (3 - quality) * (0.08 + (3 - quality) * 0.02));
  }
  return { ef, interval, repetitions, nextReview: Date.now() + interval * 86400000 };
}

function initCard(card) {
  return { ...card, ef: 2.5, interval: 1, repetitions: 0, nextReview: Date.now() };
}

// ── Storage helpers ───────────────────────────────────────────────────────────
async function loadDecks() {
  try {
    const r = await window.storage.get("mlcards:decks");
    return r ? JSON.parse(r.value) : null;
  } catch { return null; }
}
async function saveDecks(decks) {
  try { await window.storage.set("mlcards:decks", JSON.stringify(decks)); } catch {}
}

// ── Seed data ─────────────────────────────────────────────────────────────────
const SEED_DECKS = [
  {
    id: "deck-1", name: "ML Fundamentals", color: "#0ea5e9", icon: "⬡", created: Date.now(),
    cards: [
      { id: "c1", front: "What is the bias-variance tradeoff?", back: "Bias error comes from wrong assumptions in the learning algorithm (underfitting). Variance error comes from sensitivity to small fluctuations in training data (overfitting). Increasing model complexity reduces bias but increases variance. The goal is to find the sweet spot that minimizes total error." },
      { id: "c2", front: "Explain L1 vs L2 regularization.", back: "L1 (Lasso) adds $|w|$ penalty — produces sparse solutions by driving some weights to exactly zero; useful for feature selection. L2 (Ridge) adds $w^2$ penalty — shrinks weights toward zero but rarely exactly; handles multicollinearity well. The combined Elastic Net objective is:\n$$L = L_0 + \\lambda_1 \\|w\\|_1 + \\lambda_2 \\|w\\|_2^2$$" },
      { id: "c3", front: "What is the kernel trick in SVMs?", back: "The kernel trick implicitly maps data into a high-dimensional feature space by replacing dot products with $K(x, x') = \\phi(x) \\cdot \\phi(x')$. This allows SVMs to find non-linear decision boundaries without explicitly computing $\\phi$. Common kernels: RBF $K(x,x')=\\exp(-\\gamma\\|x-x'\\|^2)$, polynomial, and sigmoid." },
    ].map(initCard)
  },
  {
    id: "deck-2", name: "Deep Learning", color: "#f472b6", icon: "∇", created: Date.now(),
    cards: [
      { id: "c4", front: "What is batch normalization and why does it help?", back: "Batch normalization normalizes layer inputs to have zero mean and unit variance per mini-batch, then applies learnable γ (scale) and β (shift). Benefits: reduces internal covariate shift, allows higher learning rates, acts as slight regularization, reduces sensitivity to initialization." },
      { id: "c5", front: "Explain the attention mechanism in transformers.", back: "Attention computes a weighted sum of values based on query-key compatibility:\n$$\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\nThe $\\sqrt{d_k}$ scaling prevents dot products from growing large in high dimensions. Multi-head attention runs this in parallel across $h$ subspaces, then concatenates and projects results." },
      { id: "c6", front: "Derive $\\partial L / \\partial z_j$ for cross-entropy loss with softmax.", back: "For true label $y$ (one-hot) and softmax output $p_i = e^{z_i}/\\sum_k e^{z_k}$:\n$$\\frac{\\partial L}{\\partial z_j} = p_j - y_j$$\nThis elegant result comes from the chain rule where $\\partial p_i/\\partial z_j = p_i(\\delta_{ij} - p_j)$. The softmax Jacobian and log loss cancel beautifully." },
    ].map(initCard)
  },
];

// ── AI Generation ─────────────────────────────────────────────────────────────
async function generateCards(topic, count, deckName) {
  const prompt = `You are an expert ML educator. Generate exactly ${count} high-quality flashcards about "${topic}" for the deck "${deckName}".

Return ONLY valid JSON in this exact format, no other text:
{
  "cards": [
    {
      "front": "Question text here",
      "back": "Detailed answer here. Use \\n for line breaks. Include equations if relevant."
    }
  ]
}

Requirements:
- Mix of conceptual, mathematical, and practical questions
- Answers should be 2-6 sentences, precise and educational
- Use proper ML terminology
- Use LaTeX for ALL mathematical expressions: inline math with $...$ and display/block equations with $$...$$
- Example inline: "The update rule is $w \\leftarrow w - \\alpha \\nabla L$"
- Example display: "$$\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$"
- Always use LaTeX for Greek letters, fractions, subscripts, superscripts, summations`;

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      messages: [{ role: "user", content: prompt }]
    })
  });
  const data = await response.json();
  const text = data.content.map(b => b.text || "").join("");
  const clean = text.replace(/```json|```/g, "").trim();
  const parsed = JSON.parse(clean);
  return parsed.cards.map((c, i) => initCard({ id: `ai-${Date.now()}-${i}`, ...c }));
}

// ── Palette ───────────────────────────────────────────────────────────────────
const DECK_COLORS = ["#0ea5e9","#4ade80","#f472b6","#fb923c","#a78bfa","#34d399","#f59e0b","#60a5fa"];
const DECK_ICONS = ["⬡","∇","∑","◈","⊕","⟁","⊗","◉"];
const QUALITY_BTNS = [
  { label: "Blackout", sub: "No recall", q: 0, color: "#ef4444" },
  { label: "Hard", sub: "Barely", q: 1, color: "#f97316" },
  { label: "Good", sub: "With effort", q: 2, color: "#eab308" },
  { label: "Easy", sub: "Instantly", q: 3, color: "#4ade80" },
];

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [decks, setDecks] = useState(null);
  const [view, setView] = useState("home"); // home | deck | review | generate
  const [activeDeck, setActiveDeck] = useState(null);
  const [queue, setQueue] = useState([]);
  const [qIdx, setQIdx] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [sessionStats, setSessionStats] = useState({ reviewed: 0, easy: 0, hard: 0 });
  const [doneAnim, setDoneAnim] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState("");
  const [genTopic, setGenTopic] = useState("");
  const [genCount, setGenCount] = useState(5);
  const [genDeckId, setGenDeckId] = useState("new");
  const [genDeckName, setGenDeckName] = useState("");
  const [showNewDeckForm, setShowNewDeckForm] = useState(false);
  const [newDeckName, setNewDeckName] = useState("");
  const [newDeckColor, setNewDeckColor] = useState(DECK_COLORS[0]);
  const [newDeckIcon, setNewDeckIcon] = useState(DECK_ICONS[0]);
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  // Load persisted decks
  useEffect(() => {
    loadDecks().then(d => setDecks(d || SEED_DECKS));
  }, []);

  // Persist on change
  useEffect(() => {
    if (decks) saveDecks(decks);
  }, [decks]);

  const updateDecks = useCallback((updater) => setDecks(prev => {
    const next = updater(prev);
    return next;
  }), []);

  function getDeck(id) { return decks?.find(d => d.id === id); }

  function startReview(deckId) {
    const deck = getDeck(deckId);
    if (!deck) return;
    const due = deck.cards.filter(c => c.nextReview <= Date.now());
    if (!due.length) return;
    setActiveDeck(deckId);
    setQueue([...due].sort((a, b) => a.nextReview - b.nextReview));
    setQIdx(0); setFlipped(false);
    setSessionStats({ reviewed: 0, easy: 0, hard: 0 });
    setView("review");
  }

  function handleRate(quality) {
    const card = queue[qIdx];
    const updated = sm2(card, quality);
    updateDecks(prev => prev.map(d =>
      d.id === activeDeck
        ? { ...d, cards: d.cards.map(c => c.id === card.id ? { ...c, ...updated } : c) }
        : d
    ));
    setSessionStats(s => ({ reviewed: s.reviewed + 1, easy: s.easy + (quality >= 2 ? 1 : 0), hard: s.hard + (quality < 2 ? 1 : 0) }));
    if (qIdx + 1 >= queue.length) {
      setDoneAnim(true);
      setTimeout(() => { setDoneAnim(false); setView("deck"); }, 1800);
    } else {
      setFlipped(false);
      setTimeout(() => setQIdx(i => i + 1), 50);
    }
  }

  async function handleGenerate() {
    if (!genTopic.trim()) return;
    setGenerating(true); setGenError("");
    try {
      const targetDeckName = genDeckId === "new" ? (genDeckName.trim() || genTopic) : getDeck(genDeckId)?.name;
      const newCards = await generateCards(genTopic, genCount, targetDeckName);
      if (genDeckId === "new") {
        const newDeck = {
          id: `deck-${Date.now()}`,
          name: genDeckName.trim() || genTopic,
          color: DECK_COLORS[Math.floor(Math.random() * DECK_COLORS.length)],
          icon: DECK_ICONS[Math.floor(Math.random() * DECK_ICONS.length)],
          created: Date.now(),
          cards: newCards,
        };
        updateDecks(prev => [...prev, newDeck]);
        setActiveDeck(newDeck.id);
        setView("deck");
      } else {
        updateDecks(prev => prev.map(d =>
          d.id === genDeckId ? { ...d, cards: [...d.cards, ...newCards] } : d
        ));
        setView("deck");
      }
      setGenTopic(""); setGenDeckName(""); setGenDeckId("new");
    } catch (e) {
      setGenError("Generation failed. Check your API connection and try again.");
    } finally { setGenerating(false); }
  }

  function createDeck() {
    if (!newDeckName.trim()) return;
    const deck = { id: `deck-${Date.now()}`, name: newDeckName.trim(), color: newDeckColor, icon: newDeckIcon, created: Date.now(), cards: [] };
    updateDecks(prev => [...prev, deck]);
    setNewDeckName(""); setShowNewDeckForm(false);
    setActiveDeck(deck.id); setView("deck");
  }

  function deleteDeck(id) {
    updateDecks(prev => prev.filter(d => d.id !== id));
    setDeleteConfirm(null);
    if (activeDeck === id) { setActiveDeck(null); setView("home"); }
  }

  const current = queue[qIdx];
  const progress = queue.length > 0 ? (qIdx / queue.length) * 100 : 0;
  const activeDeckData = activeDeck ? getDeck(activeDeck) : null;

  if (!decks) return (
    <div style={{ minHeight: "100vh", background: "#060a10", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'DM Mono', monospace", color: "#475569" }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 32, marginBottom: 12, animation: "spin 1.5s linear infinite", display: "inline-block" }}>◉</div>
        <div style={{ fontSize: 12, letterSpacing: "0.1em" }}>loading your decks...</div>
      </div>
    </div>
  );

  return (
    <div style={{ minHeight: "100vh", background: "#060a10", fontFamily: "'DM Mono', monospace", color: "#e2e8f0" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 3px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 2px; }
        input, textarea, select { outline: none; }
        .card-scene { perspective: 1400px; cursor: pointer; }
        .card-inner { position: relative; width: 100%; height: 100%; transition: transform 0.6s cubic-bezier(0.4,0,0.2,1); transform-style: preserve-3d; }
        .card-inner.flipped { transform: rotateY(180deg); }
        .card-face { position: absolute; inset: 0; backface-visibility: hidden; -webkit-backface-visibility: hidden; border-radius: 16px; padding: 36px; display: flex; flex-direction: column; }
        .card-back-face { transform: rotateY(180deg); }
        .btn-primary { background: linear-gradient(135deg, #1d4ed8, #0ea5e9); border: none; color: white; padding: 11px 28px; border-radius: 8px; font-family: 'DM Mono', monospace; font-size: 13px; cursor: pointer; transition: all 0.2s; letter-spacing: 0.04em; }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(14,165,233,0.25); }
        .btn-primary:disabled { opacity: 0.35; cursor: not-allowed; transform: none; box-shadow: none; }
        .btn-ghost { background: transparent; border: 1px solid #1e2d4a; color: #64748b; padding: 8px 18px; border-radius: 7px; font-family: 'DM Mono', monospace; font-size: 12px; cursor: pointer; transition: all 0.2s; }
        .btn-ghost:hover { border-color: #2d4a6e; color: #94a3b8; background: #0d1321; }
        .btn-danger { background: transparent; border: 1px solid #ef444430; color: #ef4444; padding: 7px 16px; border-radius: 7px; font-family: 'DM Mono', monospace; font-size: 11px; cursor: pointer; transition: all 0.2s; }
        .btn-danger:hover { background: #ef444415; }
        .input-field { background: #0a1220; border: 1px solid #1e2d4a; border-radius: 8px; padding: 10px 14px; color: #e2e8f0; font-family: 'DM Mono', monospace; font-size: 13px; transition: border-color 0.2s; width: 100%; }
        .input-field:focus { border-color: #2d4a6e; }
        .input-field::placeholder { color: #334155; }
        .rate-btn { border: none; border-radius: 8px; padding: 10px 0; cursor: pointer; font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500; transition: transform 0.15s, filter 0.15s; }
        .rate-btn:hover { transform: translateY(-2px); filter: brightness(1.2); }
        .rate-btn:active { transform: scale(0.97); }
        .deck-card { background: #0a1220; border: 1px solid #1a2540; border-radius: 14px; padding: 22px; cursor: pointer; transition: all 0.22s; position: relative; overflow: hidden; }
        .deck-card:hover { border-color: #2d4a6e; transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); }
        .grid-bg { position: fixed; inset: 0; pointer-events: none; background-image: linear-gradient(rgba(30,45,74,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(30,45,74,0.15) 1px, transparent 1px); background-size: 48px 48px; }
        .glow-orb { position: fixed; border-radius: 50%; pointer-events: none; }
        .nav-item { background: transparent; border: none; color: #475569; padding: 7px 16px; border-radius: 6px; font-family: 'DM Mono', monospace; font-size: 11px; cursor: pointer; transition: all 0.2s; letter-spacing: 0.06em; text-transform: uppercase; }
        .nav-item:hover { color: #94a3b8; background: #0d1321; }
        .nav-item.active { color: #0ea5e9; background: #0d1a2e; }
        .done-overlay { position: fixed; inset: 0; background: rgba(6,10,16,0.96); display: flex; align-items: center; justify-content: center; z-index: 200; animation: fadeIn 0.3s; }
        .color-swatch { width: 28px; height: 28px; border-radius: 50%; cursor: pointer; transition: transform 0.15s; border: 2px solid transparent; flex-shrink: 0; }
        .color-swatch:hover { transform: scale(1.15); }
        .color-swatch.selected { border-color: white; transform: scale(1.1); }
        .icon-swatch { width: 34px; height: 34px; border-radius: 8px; background: #0a1220; border: 1px solid #1e2d4a; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 16px; transition: all 0.15s; }
        .icon-swatch:hover { border-color: #2d4a6e; background: #0d1a2e; }
        .icon-swatch.selected { border-color: #0ea5e9; background: #0a1e38; }
        .progress-bar { height: 2px; background: #1e2d4a; border-radius: 1px; overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 1px; transition: width 0.5s ease; }
        .tag { border-radius: 5px; padding: 3px 10px; font-size: 10px; letter-spacing: 0.07em; text-transform: uppercase; }
        .katex { color: inherit; font-size: 1.05em; }
        .katex-display { overflow-x: auto; overflow-y: hidden; padding: 4px 0; }
        .katex-display > .katex { text-align: center; }
        @keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }
        @keyframes popIn { from { transform: scale(0) rotate(-10deg); opacity: 0 } to { transform: scale(1) rotate(0); opacity: 1 } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(16px) } to { opacity: 1; transform: translateY(0) } }
        @keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }
        @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.4 } }
        .slide-up { animation: slideUp 0.35s ease forwards; }
        .spin { animation: spin 1.2s linear infinite; display: inline-block; }
        .select-field { background: #0a1220; border: 1px solid #1e2d4a; border-radius: 8px; padding: 10px 14px; color: #e2e8f0; font-family: 'DM Mono', monospace; font-size: 13px; width: 100%; cursor: pointer; appearance: none; }
        .select-field:focus { border-color: #2d4a6e; outline: none; }
        .breadcrumb { font-size: 11px; color: #334155; letter-spacing: 0.05em; display: flex; align-items: center; gap: 8px; margin-bottom: 28px; }
        .breadcrumb span { cursor: pointer; transition: color 0.2s; } .breadcrumb span:hover { color: #60a5fa; }
        .card-list-item { background: #0a1220; border: 1px solid #1a2540; border-radius: 8px; padding: 12px 16px; display: flex; gap: 12px; align-items: flex-start; }
        .tooltip { position: relative; } .tooltip:hover::after { content: attr(data-tip); position: absolute; bottom: calc(100% + 6px); left: 50%; transform: translateX(-50%); background: #1e2d4a; color: #94a3b8; font-size: 10px; padding: 4px 8px; border-radius: 4px; white-space: nowrap; pointer-events: none; }
      `}</style>

      <div className="grid-bg" />
      <div className="glow-orb" style={{ width: 500, height: 500, background: "radial-gradient(circle, rgba(14,165,233,0.05) 0%, transparent 70%)", top: -200, right: -100 }} />
      <div className="glow-orb" style={{ width: 400, height: 400, background: "radial-gradient(circle, rgba(244,114,182,0.04) 0%, transparent 70%)", bottom: -100, left: -100 }} />

      {/* Done overlay */}
      {doneAnim && (
        <div className="done-overlay">
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 56, animation: "popIn 0.5s cubic-bezier(0.34,1.56,0.64,1)", color: "#4ade80" }}>✓</div>
            <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 32, marginTop: 16 }}>Session Complete</div>
            <div style={{ color: "#475569", fontSize: 12, marginTop: 8 }}>
              {sessionStats.reviewed} reviewed · {sessionStats.easy} easy · {sessionStats.hard} hard
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div style={{ borderBottom: "1px solid #0f1a2e", padding: "14px 28px", display: "flex", alignItems: "center", justifyContent: "space-between", backdropFilter: "blur(16px)", position: "sticky", top: 0, zIndex: 50, background: "rgba(6,10,16,0.9)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }} onClick={() => setView("home")}>
          <div style={{ width: 26, height: 26, background: "linear-gradient(135deg, #1d4ed8, #0ea5e9)", borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13 }}>∇</div>
          <span style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 20, letterSpacing: "-0.01em" }}>ml<span style={{ color: "#0ea5e9", fontStyle: "italic" }}>cards</span></span>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          <button className={`nav-item ${view === "home" ? "active" : ""}`} onClick={() => setView("home")}>decks</button>
          <button className={`nav-item ${view === "generate" ? "active" : ""}`} onClick={() => { setActiveDeck(null); setView("generate"); }}>+ generate</button>
        </div>
      </div>

      <div style={{ maxWidth: 800, margin: "0 auto", padding: "36px 24px", position: "relative" }}>

        {/* ── HOME ── */}
        {view === "home" && (
          <div className="slide-up">
            <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: 36 }}>
              <div>
                <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 42, lineHeight: 1.1, marginBottom: 8 }}>
                  Your <span style={{ fontStyle: "italic", color: "#0ea5e9" }}>Decks</span>
                </div>
                <div style={{ color: "#334155", fontSize: 12 }}>
                  {decks.length} deck{decks.length !== 1 ? "s" : ""} · {decks.reduce((s, d) => s + d.cards.filter(c => c.nextReview <= Date.now()).length, 0)} cards due
                </div>
              </div>
              <button className="btn-ghost" onClick={() => setShowNewDeckForm(f => !f)}>+ new deck</button>
            </div>

            {/* New deck form */}
            {showNewDeckForm && (
              <div style={{ background: "#0a1220", border: "1px solid #1e2d4a", borderRadius: 12, padding: 24, marginBottom: 28 }}>
                <div style={{ fontSize: 11, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 20 }}>create deck</div>
                <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
                  <input className="input-field" placeholder="Deck name..." value={newDeckName} onChange={e => setNewDeckName(e.target.value)} onKeyDown={e => e.key === "Enter" && createDeck()} style={{ flex: 1 }} />
                </div>
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 10, color: "#334155", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 10 }}>color</div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {DECK_COLORS.map(c => <div key={c} className={`color-swatch ${newDeckColor === c ? "selected" : ""}`} style={{ background: c }} onClick={() => setNewDeckColor(c)} />)}
                  </div>
                </div>
                <div style={{ marginBottom: 20 }}>
                  <div style={{ fontSize: 10, color: "#334155", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 10 }}>icon</div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {DECK_ICONS.map(ic => <div key={ic} className={`icon-swatch ${newDeckIcon === ic ? "selected" : ""}`} onClick={() => setNewDeckIcon(ic)}>{ic}</div>)}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 10 }}>
                  <button className="btn-primary" onClick={createDeck} disabled={!newDeckName.trim()}>create</button>
                  <button className="btn-ghost" onClick={() => setShowNewDeckForm(false)}>cancel</button>
                </div>
              </div>
            )}

            {decks.length === 0 ? (
              <div style={{ textAlign: "center", padding: "60px 0", color: "#334155" }}>
                <div style={{ fontSize: 36, marginBottom: 12 }}>◉</div>
                <div style={{ fontSize: 13 }}>No decks yet. Create one or generate cards with AI.</div>
              </div>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
                {decks.map(deck => {
                  const due = deck.cards.filter(c => c.nextReview <= Date.now()).length;
                  const mastered = deck.cards.filter(c => c.repetitions >= 3 && c.ef > 2.2).length;
                  return (
                    <div key={deck.id} className="deck-card" onClick={() => { setActiveDeck(deck.id); setView("deck"); }}>
                      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, ${deck.color}, transparent)`, borderRadius: "14px 14px 0 0" }} />
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                        <div style={{ width: 40, height: 40, borderRadius: 10, background: `${deck.color}18`, border: `1px solid ${deck.color}30`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20, color: deck.color }}>
                          {deck.icon}
                        </div>
                        {due > 0 && <span className="tag" style={{ background: "#f9731618", color: "#fb923c", border: "1px solid #f9731630" }}>{due} due</span>}
                      </div>
                      <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 20, marginBottom: 6, color: "#e2e8f0" }}>{deck.name}</div>
                      <div style={{ fontSize: 11, color: "#334155", marginBottom: 16 }}>{deck.cards.length} cards · {mastered} mastered</div>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: deck.cards.length ? `${(mastered / deck.cards.length) * 100}%` : "0%", background: deck.color }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ── DECK VIEW ── */}
        {view === "deck" && activeDeckData && (
          <div className="slide-up">
            <div className="breadcrumb">
              <span onClick={() => setView("home")}>decks</span>
              <span style={{ color: "#1e2d4a" }}>›</span>
              <span style={{ color: "#94a3b8", cursor: "default" }}>{activeDeckData.name}</span>
            </div>

            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 32, gap: 16 }}>
              <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                <div style={{ width: 52, height: 52, borderRadius: 14, background: `${activeDeckData.color}18`, border: `1px solid ${activeDeckData.color}30`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 26, color: activeDeckData.color, flexShrink: 0 }}>
                  {activeDeckData.icon}
                </div>
                <div>
                  <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 32, lineHeight: 1 }}>{activeDeckData.name}</div>
                  <div style={{ color: "#334155", fontSize: 11, marginTop: 5 }}>
                    {activeDeckData.cards.length} cards · {activeDeckData.cards.filter(c => c.nextReview <= Date.now()).length} due · {activeDeckData.cards.filter(c => c.repetitions >= 3 && c.ef > 2.2).length} mastered
                  </div>
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, flexShrink: 0, flexWrap: "wrap", justifyContent: "flex-end" }}>
                <button className="btn-ghost" style={{ fontSize: 11 }} onClick={() => { setGenDeckId(activeDeckData.id); setView("generate"); }}>+ generate cards</button>
                <button className="btn-primary" style={{ fontSize: 12 }} onClick={() => startReview(activeDeckData.id)} disabled={!activeDeckData.cards.filter(c => c.nextReview <= Date.now()).length}>
                  review {activeDeckData.cards.filter(c => c.nextReview <= Date.now()).length > 0 ? `(${activeDeckData.cards.filter(c => c.nextReview <= Date.now()).length})` : ""}
                </button>
              </div>
            </div>

            {/* Card list */}
            {activeDeckData.cards.length === 0 ? (
              <div style={{ textAlign: "center", padding: "48px 0", color: "#334155" }}>
                <div style={{ fontSize: 28, marginBottom: 12 }}>◈</div>
                <div style={{ fontSize: 13, marginBottom: 20 }}>This deck is empty.</div>
                <button className="btn-primary" onClick={() => { setGenDeckId(activeDeckData.id); setView("generate"); }}>generate cards with AI</button>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {activeDeckData.cards.map((card, i) => {
                  const due = card.nextReview <= Date.now();
                  const mastered = card.repetitions >= 3 && card.ef > 2.2;
                  const daysUntil = Math.ceil((card.nextReview - Date.now()) / 86400000);
                  return (
                    <div key={card.id} className="card-list-item">
                      <div style={{ width: 8, height: 8, borderRadius: "50%", background: mastered ? "#4ade80" : due ? "#f97316" : "#1e2d4a", marginTop: 5, flexShrink: 0 }} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{card.front}</div>
                        <div style={{ fontSize: 10, color: "#334155", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{card.back}</div>
                      </div>
                      <div style={{ flexShrink: 0, textAlign: "right" }}>
                        <div style={{ fontSize: 10, color: due ? "#f97316" : "#334155" }}>{due ? "due" : `in ${daysUntil}d`}</div>
                        <div style={{ fontSize: 10, color: "#1e2d4a", marginTop: 2 }}>ef {card.ef.toFixed(1)}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            <div style={{ marginTop: 32, paddingTop: 24, borderTop: "1px solid #0f1a2e" }}>
              {deleteConfirm === activeDeckData.id ? (
                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                  <span style={{ fontSize: 12, color: "#ef4444" }}>Delete this deck?</span>
                  <button className="btn-danger" onClick={() => deleteDeck(activeDeckData.id)}>yes, delete</button>
                  <button className="btn-ghost" style={{ fontSize: 11 }} onClick={() => setDeleteConfirm(null)}>cancel</button>
                </div>
              ) : (
                <button className="btn-danger" onClick={() => setDeleteConfirm(activeDeckData.id)}>delete deck</button>
              )}
            </div>
          </div>
        )}

        {/* ── GENERATE ── */}
        {view === "generate" && (
          <div className="slide-up">
            <div className="breadcrumb">
              <span onClick={() => setView("home")}>decks</span>
              <span style={{ color: "#1e2d4a" }}>›</span>
              <span style={{ color: "#94a3b8", cursor: "default" }}>generate cards</span>
            </div>

            <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 38, marginBottom: 8 }}>
              AI <span style={{ fontStyle: "italic", color: "#0ea5e9" }}>Card Generator</span>
            </div>
            <div style={{ color: "#334155", fontSize: 12, marginBottom: 36 }}>Describe a topic and Claude will generate high-quality flashcards.</div>

            <div style={{ background: "#0a1220", border: "1px solid #1e2d4a", borderRadius: 14, padding: 28 }}>
              <div style={{ marginBottom: 20 }}>
                <label style={{ fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>topic</label>
                <input className="input-field" placeholder="e.g. Transformer architecture, Bayesian inference, RLHF..." value={genTopic} onChange={e => setGenTopic(e.target.value)} onKeyDown={e => e.key === "Enter" && !generating && handleGenerate()} />
                <div style={{ fontSize: 10, color: "#1e2d4a", marginTop: 6 }}>Be specific for better cards — "attention mechanism in transformers" beats "deep learning"</div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
                <div>
                  <label style={{ fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>number of cards</label>
                  <div style={{ display: "flex", gap: 6 }}>
                    {[3, 5, 8, 10].map(n => (
                      <button key={n} onClick={() => setGenCount(n)} style={{ flex: 1, padding: "8px 0", borderRadius: 6, background: genCount === n ? "#0d1a2e" : "#060a10", border: `1px solid ${genCount === n ? "#2d4a6e" : "#1a2540"}`, color: genCount === n ? "#60a5fa" : "#475569", cursor: "pointer", fontFamily: "'DM Mono', monospace", fontSize: 12, transition: "all 0.15s" }}>{n}</button>
                    ))}
                  </div>
                </div>

                <div>
                  <label style={{ fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>add to deck</label>
                  <select className="select-field" value={genDeckId} onChange={e => setGenDeckId(e.target.value)}>
                    <option value="new">+ create new deck</option>
                    {decks.map(d => <option key={d.id} value={d.id}>{d.icon} {d.name}</option>)}
                  </select>
                </div>
              </div>

              {genDeckId === "new" && (
                <div style={{ marginBottom: 20 }}>
                  <label style={{ fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: 8 }}>new deck name <span style={{ color: "#334155" }}>(optional, defaults to topic)</span></label>
                  <input className="input-field" placeholder={genTopic || "Deck name..."} value={genDeckName} onChange={e => setGenDeckName(e.target.value)} />
                </div>
              )}

              {genError && <div style={{ fontSize: 12, color: "#ef4444", background: "#ef444410", border: "1px solid #ef444430", borderRadius: 6, padding: "10px 14px", marginBottom: 16 }}>{genError}</div>}

              <button className="btn-primary" onClick={handleGenerate} disabled={!genTopic.trim() || generating} style={{ width: "100%", padding: "13px", fontSize: 13 }}>
                {generating ? <><span className="spin" style={{ marginRight: 8 }}>◉</span>generating {genCount} cards...</> : `generate ${genCount} cards →`}
              </button>
            </div>

            {/* Examples */}
            <div style={{ marginTop: 28 }}>
              <div style={{ fontSize: 10, color: "#334155", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 14 }}>example topics</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {["Gradient descent variants", "Convolutional neural networks", "Bayesian inference", "RLHF & alignment", "Graph neural networks", "Diffusion models", "Mixture of Experts", "Contrastive learning"].map(t => (
                  <button key={t} onClick={() => setGenTopic(t)} style={{ background: "#0a1220", border: "1px solid #1a2540", borderRadius: 20, padding: "6px 14px", color: "#475569", fontSize: 11, cursor: "pointer", fontFamily: "'DM Mono', monospace", transition: "all 0.15s" }}
                    onMouseOver={e => { e.target.style.borderColor = "#2d4a6e"; e.target.style.color = "#94a3b8"; }}
                    onMouseOut={e => { e.target.style.borderColor = "#1a2540"; e.target.style.color = "#475569"; }}>
                    {t}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── REVIEW ── */}
        {view === "review" && current && activeDeckData && (
          <div className="slide-up">
            <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 28 }}>
              <div style={{ flex: 1 }}>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${progress}%`, background: activeDeckData.color }} />
                </div>
              </div>
              <div style={{ fontSize: 11, color: "#334155", flexShrink: 0 }}>{qIdx + 1} / {queue.length}</div>
              <button style={{ background: "transparent", border: "none", color: "#334155", cursor: "pointer", fontSize: 14, padding: "2px 6px", transition: "color 0.2s" }} onClick={() => setView("deck")} onMouseOver={e => e.target.style.color="#94a3b8"} onMouseOut={e => e.target.style.color="#334155"}>✕</button>
            </div>

            <div style={{ marginBottom: 18 }}>
              <span className="tag" style={{ background: `${activeDeckData.color}15`, color: activeDeckData.color, border: `1px solid ${activeDeckData.color}25` }}>
                {activeDeckData.icon} {activeDeckData.name}
              </span>
            </div>

            {/* Flip card */}
            <div className="card-scene" style={{ height: 300, marginBottom: 20 }} onClick={() => setFlipped(f => !f)}>
              <div className={`card-inner ${flipped ? "flipped" : ""}`}>
                <div className="card-face" style={{ background: "#0a1220", border: "1px solid #1a2540", justifyContent: "space-between" }}>
                  <div style={{ fontSize: 10, color: "#1e2d4a", letterSpacing: "0.12em", textTransform: "uppercase" }}>question</div>
                  <div style={{ fontSize: 22, lineHeight: 1.6, color: "#e2e8f0", flex: 1, display: "flex", alignItems: "center", padding: "16px 0" }}>
                    <LatexRenderer text={current.front} serif={true} />
                  </div>
                  <div style={{ fontSize: 10, color: "#1e2d4a", letterSpacing: "0.08em" }}>tap to reveal ↗</div>
                </div>
                <div className="card-face card-back-face" style={{ background: "#080f1e", border: `1px solid ${activeDeckData.color}30`, justifyContent: "space-between" }}>
                  <div style={{ fontSize: 10, color: activeDeckData.color, letterSpacing: "0.12em", textTransform: "uppercase", opacity: 0.7 }}>answer</div>
                  <div style={{ fontSize: 13, lineHeight: 1.9, color: "#b0bec5", flex: 1, overflowY: "auto", padding: "14px 0" }}>
                    <LatexRenderer text={current.back} />
                  </div>
                  <div style={{ fontSize: 10, color: "#1e2d4a" }}>interval {current.interval}d · ef {current.ef.toFixed(2)} · rep {current.repetitions}</div>
                </div>
              </div>
            </div>

            {flipped ? (
              <div>
                <div style={{ fontSize: 10, color: "#334155", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12, textAlign: "center" }}>how well did you recall?</div>
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
              <div style={{ textAlign: "center", color: "#1e2d4a", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase", padding: "16px 0" }}>
                recall your answer, then tap the card
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
