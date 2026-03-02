import { useState, useEffect, useCallback, useRef, useReducer } from "react";
import { sm2, initCard } from "./utils/sm2.js";
import { loadDecks, saveDecks, loadDistractorCache, saveDistractorCache, exportDistractorCache, importDistractorFile } from "./utils/storage.js";
import { generateCards, generateDistractors, shuffle } from "./utils/api.js";
import { SEED_DECKS, DECK_COLORS, DECK_ICONS, QUALITY_BTNS } from "./constants/index.js";
import LatexRenderer from "./components/LatexRenderer.jsx";
import FlipCard from "./components/FlipCard.jsx";
import MCView from "./components/MCView.jsx";
import CardEditor from "./components/CardEditor.jsx";

// ── Generate state (Refactor 3: useReducer) ───────────────────────────────────
const genInitial = { topic: "", count: 5, deckId: "new", deckName: "", generating: false, error: "" };
function genReducer(state, action) {
  switch (action.type) {
    case "SET": return { ...state, [action.key]: action.value };
    case "START": return { ...state, generating: true, error: "" };
    case "ERROR": return { ...state, generating: false, error: action.error };
    case "RESET": return genInitial;
    default: return state;
  }
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [decks, setDecks] = useState(null);
  const [view, setView] = useState("home"); // home | deck | review | generate
  const [reviewMode, setReviewMode] = useState("flip"); // flip | mc
  const [activeDeck, setActiveDeck] = useState(null);
  const [queue, setQueue] = useState([]);
  const [qIdx, setQIdx] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [sessionStats, setSessionStats] = useState({ reviewed: 0, easy: 0, hard: 0, correct: 0 });
  const [doneAnim, setDoneAnim] = useState(false);
  const [mcState, setMcState] = useState(null);
  const [distractorCache, setDistractorCache] = useState({});
  const importFileRef = useRef(null);
  const [gen, dispatchGen] = useReducer(genReducer, genInitial);
  const [showNewDeckForm, setShowNewDeckForm] = useState(false);
  const [newDeckName, setNewDeckName] = useState("");
  const [newDeckColor, setNewDeckColor] = useState(DECK_COLORS[0]);
  const [newDeckIcon, setNewDeckIcon] = useState(DECK_ICONS[0]);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [cardResources, setCardResources] = useState({});
  const [learnMoreOpen, setLearnMoreOpen] = useState(false);
  const [editingCard, setEditingCard] = useState(null); // { deckId, card } | null
  const [lightMode, setLightMode] = useState(() => localStorage.getItem("mlcards:theme") === "light");

  const toggleTheme = () => setLightMode(prev => {
    const next = !prev;
    localStorage.setItem("mlcards:theme", next ? "light" : "dark");
    return next;
  });

  const themeVars = lightMode ? {
    "--th-bg": "#f0f4f8", "--th-bg-card": "#ffffff", "--th-bg-card-alt": "#f3f7fb",
    "--th-bg-nav": "rgba(240,244,248,0.92)", "--th-bg-hover": "#e4eaf2", "--th-bg-active": "#dbeafe",
    "--th-bg-overlay": "rgba(240,244,248,0.97)",
    "--th-border": "#c8d5e6", "--th-border-soft": "#dde6f0", "--th-border-nav": "#dde6f0", "--th-border-focus": "#93c5fd",
    "--th-text": "#0f172a", "--th-text-answer": "#374151", "--th-text-muted": "#4b5563",
    "--th-text-dim": "#6b7280", "--th-text-subtle": "#6b7280", "--th-text-faint": "#94a3b8", "--th-text-ghost": "#c8d5e6",
    "--th-grid": "rgba(100,116,139,0.08)", "--th-scrollbar": "#c8d5e6",
    "--th-tooltip-bg": "#e2e8f0", "--th-tooltip-color": "#334155",
    "--th-glow1": "rgba(14,165,233,0.07)", "--th-glow2": "rgba(244,114,182,0.05)",
  } : {
    "--th-bg": "#060a10", "--th-bg-card": "#0a1220", "--th-bg-card-alt": "#080f1e",
    "--th-bg-nav": "rgba(6,10,16,0.9)", "--th-bg-hover": "#0d1321", "--th-bg-active": "#0d1a2e",
    "--th-bg-overlay": "rgba(6,10,16,0.96)",
    "--th-border": "#1e2d4a", "--th-border-soft": "#1a2540", "--th-border-nav": "#0f1a2e", "--th-border-focus": "#2d4a6e",
    "--th-text": "#e2e8f0", "--th-text-answer": "#b0bec5", "--th-text-muted": "#94a3b8",
    "--th-text-dim": "#64748b", "--th-text-subtle": "#475569", "--th-text-faint": "#334155", "--th-text-ghost": "#1e2d4a",
    "--th-grid": "rgba(30,45,74,0.15)", "--th-scrollbar": "#1e2d4a",
    "--th-tooltip-bg": "#1e2d4a", "--th-tooltip-color": "#94a3b8",
    "--th-glow1": "rgba(14,165,233,0.05)", "--th-glow2": "rgba(244,114,182,0.04)",
  };

  useEffect(() => {
    loadDecks(SEED_DECKS, initCard).then(d => setDecks(d));
    loadDistractorCache().then(c => setDistractorCache(c));
    fetch("/resources.json").then(r => r.ok ? r.json() : {}).then(setCardResources).catch(() => {});
  }, []);

  useEffect(() => { setLearnMoreOpen(false); }, [qIdx]);

  useEffect(() => {
    if (decks) saveDecks(decks);
  }, [decks]);

  const updateDecks = useCallback((updater) => setDecks(prev => updater(prev)), []);

  function getDeck(id) { return decks?.find(d => d.id === id); }

  function startReview(deckId, mode = "flip") {
    const deck = getDeck(deckId);
    if (!deck) return;
    const due = deck.cards.filter(c => c.nextReview <= Date.now());
    if (!due.length) return;
    const sorted = [...due].sort((a, b) => a.nextReview - b.nextReview);
    setActiveDeck(deckId);
    setQueue(sorted);
    setQIdx(0); setFlipped(false);
    setReviewMode(mode);
    setSessionStats({ reviewed: 0, easy: 0, hard: 0, correct: 0 });
    setMcState(null);
    setView("review");
    if (mode === "mc") loadMcChoices(sorted[0], deck.cards);
  }

  async function loadMcChoices(card, allCards, forceRegenerate = false) {
    const cached = !forceRegenerate && distractorCache[card.id];
    if (cached) {
      const choices = shuffle([
        { text: card.back, correct: true },
        ...cached.map(d => ({ text: d, correct: false }))
      ]);
      setMcState({ choices, selected: null, loading: false, cardId: card.id });
      return;
    }
    setMcState({ choices: null, selected: null, loading: true, cardId: card.id });
    try {
      const distractors = await generateDistractors(card, allCards);
      const newCache = { ...distractorCache, [card.id]: distractors };
      setDistractorCache(newCache);
      saveDistractorCache(newCache);
      const choices = shuffle([
        { text: card.back, correct: true },
        ...distractors.map(d => ({ text: d, correct: false }))
      ]);
      setMcState({ choices, selected: null, loading: false, cardId: card.id });
    } catch {
      setMcState({ choices: null, selected: null, loading: false, error: true, cardId: card.id });
    }
  }

  function handleMcSelect(idx) {
    setMcState(prev => prev.selected !== null ? prev : { ...prev, selected: idx });
  }

  function handleMcNext(wasCorrect) {
    const card = queue[qIdx];
    const quality = wasCorrect ? 3 : 1;
    const updated = sm2(card, quality);
    updateDecks(prev => prev.map(d =>
      d.id === activeDeck
        ? { ...d, cards: d.cards.map(c => c.id === card.id ? { ...c, ...updated } : c) }
        : d
    ));
    setSessionStats(s => ({
      reviewed: s.reviewed + 1,
      easy: s.easy + (wasCorrect ? 1 : 0),
      hard: s.hard + (wasCorrect ? 0 : 1),
      correct: s.correct + (wasCorrect ? 1 : 0),
    }));
    if (qIdx + 1 >= queue.length) {
      setDoneAnim(true);
      setTimeout(() => { setDoneAnim(false); setView("deck"); }, 1800);
    } else {
      const nextCard = queue[qIdx + 1];
      const deck = getDeck(activeDeck);
      setMcState(null);
      setQIdx(i => i + 1);
      loadMcChoices(nextCard, deck.cards, false);
    }
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
    if (!gen.topic.trim()) return;
    dispatchGen({ type: "START" });
    try {
      const targetDeckName = gen.deckId === "new" ? (gen.deckName.trim() || gen.topic) : getDeck(gen.deckId)?.name;
      const newCards = await generateCards(gen.topic, gen.count, targetDeckName);
      if (gen.deckId === "new") {
        const newDeck = {
          id: `deck-${Date.now()}`,
          name: gen.deckName.trim() || gen.topic,
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
          d.id === gen.deckId ? { ...d, cards: [...d.cards, ...newCards] } : d
        ));
        setView("deck");
      }
      dispatchGen({ type: "RESET" });
    } catch {
      dispatchGen({ type: "ERROR", error: "Generation failed. Check your API connection and try again." });
    }
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

  function saveCardEdit(deckId, cardId, newFront, newBack) {
    updateDecks(prev => prev.map(d =>
      d.id === deckId
        ? { ...d, cards: d.cards.map(c => c.id === cardId ? { ...c, front: newFront, back: newBack } : c) }
        : d
    ));
    if (distractorCache[cardId]) {
      const newCache = { ...distractorCache };
      delete newCache[cardId];
      setDistractorCache(newCache);
      saveDistractorCache(newCache);
    }
    setEditingCard(null);
  }

  const current = queue[qIdx];
  const progress = queue.length > 0 ? (qIdx / queue.length) * 100 : 0;
  const activeDeckData = activeDeck ? getDeck(activeDeck) : null;

  if (!decks) return (
    <div style={{ minHeight: "100vh", background: "var(--th-bg)", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'DM Mono', monospace", color: "var(--th-text-subtle)", ...themeVars }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 32, marginBottom: 12, animation: "spin 1.5s linear infinite", display: "inline-block" }}>◉</div>
        <div style={{ fontSize: 12, letterSpacing: "0.1em" }}>loading your decks...</div>
      </div>
    </div>
  );

  return (
    <div style={{ minHeight: "100vh", background: "var(--th-bg)", fontFamily: "'DM Mono', monospace", color: "var(--th-text)", ...themeVars }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 3px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: var(--th-scrollbar); border-radius: 2px; }
        input, textarea, select { outline: none; }
        .card-scene { perspective: 1400px; cursor: pointer; }
        .card-inner { position: relative; width: 100%; height: 100%; transition: transform 0.6s cubic-bezier(0.4,0,0.2,1); transform-style: preserve-3d; }
        .card-inner.flipped { transform: rotateY(180deg); }
        .card-face { position: absolute; inset: 0; backface-visibility: hidden; -webkit-backface-visibility: hidden; border-radius: 16px; padding: 36px; display: flex; flex-direction: column; }
        .card-back-face { transform: rotateY(180deg); background: var(--th-bg-card-alt); }
        .card-front-face { background: var(--th-bg-card); border: 1px solid var(--th-border-soft); justify-content: space-between; }
        .card-question-label { font-size: 10px; color: var(--th-text-ghost); letter-spacing: 0.12em; text-transform: uppercase; }
        .card-tap-hint { font-size: 10px; color: var(--th-text-ghost); letter-spacing: 0.08em; }
        .card-sm2-stats { font-size: 10px; color: var(--th-text-ghost); }
        .learn-more-btn { background: none; border: 1px solid var(--th-border-soft); border-radius: 6px; color: var(--th-text-dim); font-size: 11px; padding: 5px 12px; cursor: pointer; letter-spacing: 0.06em; width: 100%; font-family: 'DM Mono', monospace; }
        .resource-link { display: flex; align-items: center; gap: 8px; background: var(--th-bg-card); border: 1px solid var(--th-border-soft); border-radius: 6px; padding: 7px 10px; text-decoration: none; }
        .mc-question-box { background: var(--th-bg-card); border: 1px solid var(--th-border-soft); border-radius: 14px; padding: 28px 32px; margin-bottom: 20px; min-height: 120px; display: flex; align-items: center; }
        .recall-label { font-size: 10px; color: var(--th-text-muted); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 12px; text-align: center; }
        .recall-hint { text-align: center; color: var(--th-text-ghost); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; padding: 16px 0; }
        .review-counter { font-size: 11px; color: var(--th-text-muted); flex-shrink: 0; }
        .generate-panel { background: var(--th-bg-card); border: 1px solid var(--th-border); border-radius: 14px; padding: 28px; }
        .gen-label { font-size: 10px; color: var(--th-text-subtle); letter-spacing: 0.1em; text-transform: uppercase; display: block; margin-bottom: 8px; }
        .btn-primary { background: linear-gradient(135deg, #1d4ed8, #0ea5e9); border: none; color: white; padding: 11px 28px; border-radius: 8px; font-family: 'DM Mono', monospace; font-size: 13px; cursor: pointer; transition: all 0.2s; letter-spacing: 0.04em; }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(14,165,233,0.25); }
        .btn-primary:disabled { opacity: 0.35; cursor: not-allowed; transform: none; box-shadow: none; }
        .btn-ghost { background: transparent; border: 1px solid var(--th-border); color: var(--th-text-dim); padding: 8px 18px; border-radius: 7px; font-family: 'DM Mono', monospace; font-size: 12px; cursor: pointer; transition: all 0.2s; }
        .btn-ghost:hover { border-color: var(--th-border-focus); color: var(--th-text-muted); background: var(--th-bg-hover); }
        .btn-danger { background: transparent; border: 1px solid #ef444430; color: #ef4444; padding: 7px 16px; border-radius: 7px; font-family: 'DM Mono', monospace; font-size: 11px; cursor: pointer; transition: all 0.2s; }
        .btn-danger:hover { background: #ef444415; }
        .input-field { background: var(--th-bg-card); border: 1px solid var(--th-border); border-radius: 8px; padding: 10px 14px; color: var(--th-text); font-family: 'DM Mono', monospace; font-size: 13px; transition: border-color 0.2s; width: 100%; }
        .input-field:focus { border-color: var(--th-border-focus); }
        .input-field::placeholder { color: var(--th-text-faint); }
        .rate-btn { border: none; border-radius: 8px; padding: 10px 0; cursor: pointer; font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500; transition: transform 0.15s, filter 0.15s; }
        .rate-btn:hover { transform: translateY(-2px); filter: brightness(1.2); }
        .rate-btn:active { transform: scale(0.97); }
        .deck-card { background: var(--th-bg-card); border: 1px solid var(--th-border-soft); border-radius: 14px; padding: 22px; cursor: pointer; transition: all 0.22s; position: relative; overflow: hidden; }
        .deck-card:hover { border-color: var(--th-border-focus); transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.15); }
        .grid-bg { position: fixed; inset: 0; pointer-events: none; background-image: linear-gradient(var(--th-grid) 1px, transparent 1px), linear-gradient(90deg, var(--th-grid) 1px, transparent 1px); background-size: 48px 48px; }
        .glow-orb { position: fixed; border-radius: 50%; pointer-events: none; }
        .nav-item { background: transparent; border: none; color: var(--th-text-subtle); padding: 7px 16px; border-radius: 6px; font-family: 'DM Mono', monospace; font-size: 11px; cursor: pointer; transition: all 0.2s; letter-spacing: 0.06em; text-transform: uppercase; }
        .nav-item:hover { color: var(--th-text-muted); background: var(--th-bg-hover); }
        .nav-item.active { color: #0ea5e9; background: var(--th-bg-active); }
        .done-overlay { position: fixed; inset: 0; background: var(--th-bg-overlay); display: flex; align-items: center; justify-content: center; z-index: 200; animation: fadeIn 0.3s; }
        .color-swatch { width: 28px; height: 28px; border-radius: 50%; cursor: pointer; transition: transform 0.15s; border: 2px solid transparent; flex-shrink: 0; }
        .color-swatch:hover { transform: scale(1.15); }
        .color-swatch.selected { border-color: white; transform: scale(1.1); }
        .icon-swatch { width: 34px; height: 34px; border-radius: 8px; background: var(--th-bg-card); border: 1px solid var(--th-border); display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 16px; transition: all 0.15s; }
        .icon-swatch:hover { border-color: var(--th-border-focus); background: var(--th-bg-hover); }
        .icon-swatch.selected { border-color: #0ea5e9; background: var(--th-bg-active); }
        .mc-choice:not([disabled]):hover { filter: brightness(1.1); transform: translateX(3px); }
        .progress-bar { height: 2px; background: var(--th-border); border-radius: 1px; overflow: hidden; }
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
        .select-field { background: var(--th-bg-card); border: 1px solid var(--th-border); border-radius: 8px; padding: 10px 14px; color: var(--th-text); font-family: 'DM Mono', monospace; font-size: 13px; width: 100%; cursor: pointer; appearance: none; }
        .select-field:focus { border-color: var(--th-border-focus); outline: none; }
        .breadcrumb { font-size: 11px; color: var(--th-text-faint); letter-spacing: 0.05em; display: flex; align-items: center; gap: 8px; margin-bottom: 28px; }
        .breadcrumb span { cursor: pointer; transition: color 0.2s; } .breadcrumb span:hover { color: #60a5fa; }
        .card-list-item { background: var(--th-bg-card); border: 1px solid var(--th-border-soft); border-radius: 8px; padding: 12px 16px; display: flex; gap: 12px; align-items: flex-start; }
        .tooltip { position: relative; } .tooltip:hover::after { content: attr(data-tip); position: absolute; bottom: calc(100% + 6px); left: 50%; transform: translateX(-50%); background: var(--th-tooltip-bg); color: var(--th-tooltip-color); font-size: 10px; padding: 4px 8px; border-radius: 4px; white-space: nowrap; pointer-events: none; }
        .card-editor-modal { position: fixed; inset: 0; background: var(--th-bg-overlay); display: flex; align-items: center; justify-content: center; z-index: 200; animation: fadeIn 0.2s; padding: 24px; }
        .card-editor-box { background: var(--th-bg-card); border: 1px solid var(--th-border); border-radius: 16px; width: 100%; max-width: 780px; max-height: 88vh; overflow-y: auto; padding: 28px; display: flex; flex-direction: column; gap: 20px; }
        .editor-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
        .editor-preview { background: var(--th-bg-card-alt); border: 1px solid var(--th-border-soft); border-radius: 8px; padding: 12px 14px; font-size: 13px; line-height: 1.8; color: var(--th-text-answer); min-height: 80px; }
        .card-edit-btn { background: none; border: none; color: var(--th-text-faint); cursor: pointer; font-size: 13px; padding: 2px 4px; opacity: 0; transition: opacity 0.15s, color 0.15s; line-height: 1; font-family: inherit; }
        .card-list-item:hover .card-edit-btn { opacity: 1; }
        .card-edit-btn:hover { color: #0ea5e9; }
      `}</style>

      <div className="grid-bg" />
      <div className="glow-orb" style={{ width: 500, height: 500, background: `radial-gradient(circle, var(--th-glow1) 0%, transparent 70%)`, top: -200, right: -100 }} />
      <div className="glow-orb" style={{ width: 400, height: 400, background: `radial-gradient(circle, var(--th-glow2) 0%, transparent 70%)`, bottom: -100, left: -100 }} />

      {doneAnim && (
        <div className="done-overlay">
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 56, animation: "popIn 0.5s cubic-bezier(0.34,1.56,0.64,1)", color: "#4ade80" }}>✓</div>
            <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 32, marginTop: 16 }}>Session Complete</div>
            <div style={{ color: "var(--th-text-subtle)", fontSize: 12, marginTop: 8 }}>
              {sessionStats.reviewed} reviewed · {sessionStats.easy} easy · {sessionStats.hard} hard
            </div>
          </div>
        </div>
      )}

      {editingCard && (
        <CardEditor
          card={editingCard.card}
          onSave={(front, back) => saveCardEdit(editingCard.deckId, editingCard.card.id, front, back)}
          onClose={() => setEditingCard(null)}
        />
      )}

      {/* Header */}
      <div style={{ borderBottom: "1px solid var(--th-border-nav)", padding: "14px 28px", display: "flex", alignItems: "center", justifyContent: "space-between", backdropFilter: "blur(16px)", position: "sticky", top: 0, zIndex: 50, background: "var(--th-bg-nav)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }} onClick={() => setView("home")}>
          <div style={{ width: 26, height: 26, background: "linear-gradient(135deg, #1d4ed8, #0ea5e9)", borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13 }}>∇</div>
          <span style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 20, letterSpacing: "-0.01em" }}>ml<span style={{ color: "#0ea5e9", fontStyle: "italic" }}>cards</span></span>
        </div>
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <button className={`nav-item ${view === "home" ? "active" : ""}`} onClick={() => setView("home")}>decks</button>
          <button className={`nav-item ${view === "generate" ? "active" : ""}`} onClick={() => { setActiveDeck(null); setView("generate"); }}>+ generate</button>
          <button onClick={toggleTheme} style={{ background: "none", border: "1px solid var(--th-border)", borderRadius: 6, color: "var(--th-text-dim)", fontSize: 11, padding: "5px 10px", cursor: "pointer", fontFamily: "'DM Mono', monospace", letterSpacing: "0.06em", transition: "all 0.2s", marginLeft: 4 }}>
            {lightMode ? "◑ dark" : "◐ light"}
          </button>
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
                <div style={{ color: "var(--th-text-faint)", fontSize: 12 }}>
                  {decks.length} deck{decks.length !== 1 ? "s" : ""} · {decks.reduce((s, d) => s + d.cards.filter(c => c.nextReview <= Date.now()).length, 0)} cards due
                </div>
              </div>
              <button className="btn-ghost" onClick={() => setShowNewDeckForm(f => !f)}>+ new deck</button>
            </div>

            {showNewDeckForm && (
              <div style={{ background: "var(--th-bg-card)", border: "1px solid var(--th-border)", borderRadius: 12, padding: 24, marginBottom: 28 }}>
                <div style={{ fontSize: 11, color: "var(--th-text-subtle)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 20 }}>create deck</div>
                <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
                  <input className="input-field" placeholder="Deck name..." value={newDeckName} onChange={e => setNewDeckName(e.target.value)} onKeyDown={e => e.key === "Enter" && createDeck()} style={{ flex: 1 }} />
                </div>
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 10, color: "var(--th-text-faint)", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 10 }}>color</div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {DECK_COLORS.map(c => <div key={c} className={`color-swatch ${newDeckColor === c ? "selected" : ""}`} style={{ background: c }} onClick={() => setNewDeckColor(c)} />)}
                  </div>
                </div>
                <div style={{ marginBottom: 20 }}>
                  <div style={{ fontSize: 10, color: "var(--th-text-faint)", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 10 }}>icon</div>
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
              <div style={{ textAlign: "center", padding: "60px 0", color: "var(--th-text-faint)" }}>
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
                      <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 20, marginBottom: 6, color: "var(--th-text)" }}>{deck.name}</div>
                      <div style={{ fontSize: 11, color: "var(--th-text-faint)", marginBottom: 16 }}>{deck.cards.length} cards · {mastered} mastered</div>
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
              <span style={{ color: "var(--th-text-ghost)" }}>›</span>
              <span style={{ color: "var(--th-text-muted)", cursor: "default" }}>{activeDeckData.name}</span>
            </div>

            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 32, gap: 16 }}>
              <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                <div style={{ width: 52, height: 52, borderRadius: 14, background: `${activeDeckData.color}18`, border: `1px solid ${activeDeckData.color}30`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 26, color: activeDeckData.color, flexShrink: 0 }}>
                  {activeDeckData.icon}
                </div>
                <div>
                  <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 32, lineHeight: 1 }}>{activeDeckData.name}</div>
                  <div style={{ color: "var(--th-text-faint)", fontSize: 11, marginTop: 5 }}>
                    {activeDeckData.cards.length} cards · {activeDeckData.cards.filter(c => c.nextReview <= Date.now()).length} due · {activeDeckData.cards.filter(c => c.repetitions >= 3 && c.ef > 2.2).length} mastered
                  </div>
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, flexShrink: 0, flexWrap: "wrap", justifyContent: "flex-end" }}>
                <button className="btn-ghost" style={{ fontSize: 11 }} onClick={() => { dispatchGen({ type: "SET", key: "deckId", value: activeDeckData.id }); setView("generate"); }}>+ generate cards</button>
                {(() => {
                  const dueCount = activeDeckData.cards.filter(c => c.nextReview <= Date.now()).length;
                  const disabled = !dueCount;
                  return (<>
                    <button className="btn-ghost" style={{ fontSize: 12 }} onClick={() => startReview(activeDeckData.id, "flip")} disabled={disabled}>
                      flip {dueCount > 0 ? `(${dueCount})` : ""}
                    </button>
                    <button className="btn-primary" style={{ fontSize: 12 }} onClick={() => startReview(activeDeckData.id, "mc")} disabled={disabled}>
                      quiz {dueCount > 0 ? `(${dueCount})` : ""}
                    </button>
                  </>);
                })()}
              </div>
            </div>

            {activeDeckData.cards.length === 0 ? (
              <div style={{ textAlign: "center", padding: "48px 0", color: "var(--th-text-faint)" }}>
                <div style={{ fontSize: 28, marginBottom: 12 }}>◈</div>
                <div style={{ fontSize: 13, marginBottom: 20 }}>This deck is empty.</div>
                <button className="btn-primary" onClick={() => { dispatchGen({ type: "SET", key: "deckId", value: activeDeckData.id }); setView("generate"); }}>generate cards with AI</button>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {activeDeckData.cards.map((card) => {
                  const due = card.nextReview <= Date.now();
                  const mastered = card.repetitions >= 3 && card.ef > 2.2;
                  const daysUntil = Math.ceil((card.nextReview - Date.now()) / 86400000);
                  return (
                    <div key={card.id} className="card-list-item">
                      <div style={{ width: 8, height: 8, borderRadius: "50%", background: mastered ? "#4ade80" : due ? "#f97316" : "var(--th-border)", marginTop: 5, flexShrink: 0 }} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 12, color: "var(--th-text-muted)", marginBottom: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{card.front}</div>
                        <div style={{ fontSize: 10, color: "var(--th-text-faint)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{card.back}</div>
                      </div>
                      <div style={{ flexShrink: 0, textAlign: "right", display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4 }}>
                        <div style={{ fontSize: 10, color: due ? "#f97316" : "var(--th-text-faint)" }}>{due ? "due" : `in ${daysUntil}d`}</div>
                        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                          <button className="card-edit-btn" onClick={e => { e.stopPropagation(); setEditingCard({ deckId: activeDeck, card }); }}>✎</button>
                          {distractorCache[card.id] && (
                            <span title="distractors cached" style={{ fontSize: 9, color: "#4ade8060", letterSpacing: "0.05em" }}>◆ mc</span>
                          )}
                          <span style={{ fontSize: 10, color: "var(--th-text-ghost)" }}>ef {card.ef.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            <div style={{ marginTop: 32, paddingTop: 24, borderTop: "1px solid var(--th-border-nav)" }}>
              {(() => {
                const deckCards = activeDeckData.cards;
                const cachedCount = deckCards.filter(c => distractorCache[c.id]).length;
                const totalCount = deckCards.length;
                return totalCount > 0 ? (
                  <div style={{ marginBottom: 20 }}>
                    <div style={{ fontSize: 10, color: "var(--th-text-subtle)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>quiz distractors</div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <span style={{ fontSize: 11, color: cachedCount === totalCount ? "#4ade80" : "#f97316" }}>
                        {cachedCount}/{totalCount} cached
                      </span>
                      <button className="btn-ghost" style={{ fontSize: 11 }} onClick={() => exportDistractorCache(distractorCache)}>
                        ↓ export distractors.json
                      </button>
                      <button className="btn-ghost" style={{ fontSize: 11 }} onClick={() => importFileRef.current?.click()}>
                        ↑ import distractors.json
                      </button>
                      <input ref={importFileRef} type="file" accept=".json" style={{ display: "none" }}
                        onChange={async e => {
                          const file = e.target.files?.[0];
                          if (!file) return;
                          try {
                            const imported = await importDistractorFile(file);
                            const merged = { ...distractorCache, ...imported };
                            setDistractorCache(merged);
                            saveDistractorCache(merged);
                          } catch { alert("Failed to import: invalid JSON file"); }
                          e.target.value = "";
                        }} />
                    </div>
                  </div>
                ) : null;
              })()}

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
              <span style={{ color: "var(--th-border)" }}>›</span>
              <span style={{ color: "var(--th-text-muted)", cursor: "default" }}>generate cards</span>
            </div>

            <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 38, marginBottom: 8 }}>
              AI <span style={{ fontStyle: "italic", color: "#0ea5e9" }}>Card Generator</span>
            </div>
            <div style={{ color: "var(--th-text-muted)", fontSize: 12, marginBottom: 36 }}>Describe a topic and Claude will generate high-quality flashcards.</div>

            <div className="generate-panel">
              <div style={{ marginBottom: 20 }}>
                <label className="gen-label">topic</label>
                <input className="input-field" placeholder="e.g. Transformer architecture, Bayesian inference, RLHF..." value={gen.topic} onChange={e => dispatchGen({ type: "SET", key: "topic", value: e.target.value })} onKeyDown={e => e.key === "Enter" && !gen.generating && handleGenerate()} />
                <div style={{ fontSize: 10, color: "var(--th-text-ghost)", marginTop: 6 }}>Be specific for better cards — "attention mechanism in transformers" beats "deep learning"</div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
                <div>
                  <label className="gen-label">number of cards</label>
                  <div style={{ display: "flex", gap: 6 }}>
                    {[3, 5, 8, 10].map(n => (
                      <button key={n} onClick={() => dispatchGen({ type: "SET", key: "count", value: n })} style={{ flex: 1, padding: "8px 0", borderRadius: 6, background: gen.count === n ? "var(--th-bg-active)" : "var(--th-bg)", border: `1px solid ${gen.count === n ? "var(--th-border-focus)" : "var(--th-border-soft)"}`, color: gen.count === n ? "#60a5fa" : "var(--th-text-subtle)", cursor: "pointer", fontFamily: "'DM Mono', monospace", fontSize: 12, transition: "all 0.15s" }}>{n}</button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="gen-label">add to deck</label>
                  <select className="select-field" value={gen.deckId} onChange={e => dispatchGen({ type: "SET", key: "deckId", value: e.target.value })}>
                    <option value="new">+ create new deck</option>
                    {decks.map(d => <option key={d.id} value={d.id}>{d.icon} {d.name}</option>)}
                  </select>
                </div>
              </div>

              {gen.deckId === "new" && (
                <div style={{ marginBottom: 20 }}>
                  <label className="gen-label">new deck name <span style={{ color: "var(--th-text-muted)", textTransform: "none", letterSpacing: 0 }}>(optional, defaults to topic)</span></label>
                  <input className="input-field" placeholder={gen.topic || "Deck name..."} value={gen.deckName} onChange={e => dispatchGen({ type: "SET", key: "deckName", value: e.target.value })} />
                </div>
              )}

              {gen.error && <div style={{ fontSize: 12, color: "#ef4444", background: "#ef444410", border: "1px solid #ef444430", borderRadius: 6, padding: "10px 14px", marginBottom: 16 }}>{gen.error}</div>}

              <button className="btn-primary" onClick={handleGenerate} disabled={!gen.topic.trim() || gen.generating} style={{ width: "100%", padding: "13px", fontSize: 13 }}>
                {gen.generating ? <><span className="spin" style={{ marginRight: 8 }}>◉</span>generating {gen.count} cards...</> : `generate ${gen.count} cards →`}
              </button>
            </div>

            <div style={{ marginTop: 28 }}>
              <div style={{ fontSize: 10, color: "var(--th-text-muted)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 14 }}>example topics</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {["Gradient descent variants", "Convolutional neural networks", "Bayesian inference", "RLHF & alignment", "Graph neural networks", "Diffusion models", "Mixture of Experts", "Contrastive learning"].map(t => (
                  <button key={t} onClick={() => dispatchGen({ type: "SET", key: "topic", value: t })} style={{ background: "var(--th-bg-card)", border: "1px solid var(--th-border-soft)", borderRadius: 20, padding: "6px 14px", color: "var(--th-text-subtle)", fontSize: 11, cursor: "pointer", fontFamily: "'DM Mono', monospace", transition: "all 0.15s" }}
                    onMouseOver={e => { e.currentTarget.style.borderColor = "var(--th-border-focus)"; e.currentTarget.style.color = "var(--th-text-muted)"; }}
                    onMouseOut={e => { e.currentTarget.style.borderColor = "var(--th-border-soft)"; e.currentTarget.style.color = "var(--th-text-subtle)"; }}>
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
              <div className="review-counter">{qIdx + 1} / {queue.length}</div>
              <button style={{ background: "transparent", border: "none", color: "var(--th-text-muted)", cursor: "pointer", fontSize: 14, padding: "2px 6px", transition: "color 0.2s" }} onClick={() => setView("deck")} onMouseOver={e => e.currentTarget.style.color = "var(--th-text)"} onMouseOut={e => e.currentTarget.style.color = "var(--th-text-muted)"}>✕</button>
            </div>

            <div style={{ marginBottom: 18, display: "flex", alignItems: "center", gap: 8 }}>
              <span className="tag" style={{ background: `${activeDeckData.color}15`, color: activeDeckData.color, border: `1px solid ${activeDeckData.color}25` }}>
                {activeDeckData.icon} {activeDeckData.name}
              </span>
              <span className="tag" style={{ background: reviewMode === "mc" ? "#a78bfa18" : "#0ea5e918", color: reviewMode === "mc" ? "#a78bfa" : "#0ea5e9", border: reviewMode === "mc" ? "1px solid #a78bfa30" : "1px solid #0ea5e930" }}>
                {reviewMode === "mc" ? "quiz" : "flip"}
              </span>
            </div>

            {reviewMode === "flip" ? (
              <FlipCard
                current={current}
                flipped={flipped}
                setFlipped={setFlipped}
                activeDeckData={activeDeckData}
                cardResources={cardResources}
                learnMoreOpen={learnMoreOpen}
                setLearnMoreOpen={setLearnMoreOpen}
                handleRate={handleRate}
              />
            ) : (
              <MCView
                current={current}
                mcState={mcState}
                handleMcSelect={handleMcSelect}
                handleMcNext={handleMcNext}
                loadMcChoices={loadMcChoices}
                activeDeck={activeDeck}
                getDeck={getDeck}
                sessionStats={sessionStats}
                qIdx={qIdx}
                queueLength={queue.length}
              />
            )}
          </div>
        )}

      </div>
    </div>
  );
}
