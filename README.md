# mlcards

A spaced-repetition flashcard app for machine learning, built as a single React component. Features AI-powered card generation, deck management, LaTeX rendering, and persistent storage.

![mlcards](https://img.shields.io/badge/react-18-61dafb?style=flat-square&logo=react) ![katex](https://img.shields.io/badge/katex-0.16.9-blue?style=flat-square) ![license](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Features

- **SM-2 Spaced Repetition** — cards are scheduled based on recall quality (Blackout / Hard / Good / Easy), with per-card easiness factor, interval, and repetition count
- **AI Card Generation** — describe any ML topic and Claude generates flashcards via the Anthropic API
- **Deck Management** — create named decks with custom colors and icons; track mastery per deck
- **LaTeX Rendering** — inline `$...$` and display `$$...$$` math rendered via KaTeX
- **Persistent Storage** — deck state (including SM-2 metadata) saved via `window.storage` across sessions
- **Flip Card UI** — 3D CSS perspective flip animation; rate recall after revealing the answer

---

## Getting Started

This app is designed to run as a **Claude.ai Artifact** (React JSX). To use it:

1. Open [Claude.ai](https://claude.ai)
2. Start a new conversation and paste the contents of `ml-flashcards.jsx` into a message, asking Claude to render it as an artifact
3. The app will run in the artifact sandbox with full access to `window.storage` and the Anthropic API

### Running Locally

To run outside of Claude.ai, you'll need to adapt two things:

**1. Replace `window.storage`** with `localStorage`:

```js
// Replace loadDecks()
async function loadDecks() {
  try {
    const raw = localStorage.getItem("mlcards:decks");
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

// Replace saveDecks()
async function saveDecks(decks) {
  localStorage.setItem("mlcards:decks", JSON.stringify(decks));
}
```

**2. Add an Anthropic API key** to the fetch call in `generateCards()`:

```js
headers: {
  "Content-Type": "application/json",
  "x-api-key": "YOUR_API_KEY",
  "anthropic-version": "2023-06-01",
  "anthropic-dangerous-direct-browser-access": "true",
},
```

Then scaffold with Vite:

```bash
npm create vite@latest mlcards -- --template react
cd mlcards
npm install
# Replace src/App.jsx with ml-flashcards.jsx contents
npm run dev
```

---

## Project Structure

```
mlcards/
├── ml-flashcards.jsx   # Full app — single self-contained React component
├── README.md
└── .gitignore
```

### Architecture

Everything lives in one JSX file, organized in sections:

| Section | Description |
|---|---|
| `LatexRenderer` | Parses and renders `$...$` / `$$...$$` via KaTeX |
| `sm2()` | SM-2 algorithm — computes next interval and EF from recall quality |
| `initCard()` | Seeds a card with default SM-2 state |
| `loadDecks / saveDecks` | `window.storage` persistence helpers |
| `SEED_DECKS` | Two starter decks with LaTeX-formatted cards |
| `generateCards()` | Anthropic API call — returns `n` AI-generated cards for a topic |
| `App` | Main component — all views (home, deck, review, generate) |

---

## SM-2 Algorithm

Cards are rated 0–3 after each review:

| Rating | Label | Meaning |
|---|---|---|
| 0 | Blackout | Complete blank |
| 1 | Hard | Barely recalled |
| 2 | Good | Recalled with effort |
| 3 | Easy | Instantly recalled |

The easiness factor (EF, default 2.5) and interval are updated per review:

```
EF' = max(1.3, EF + 0.1 − (3 − q) × (0.08 + (3 − q) × 0.02))
interval: 1 → 6 → round(prev × EF') → ...
```

Cards rated < 1 reset to interval = 1, repetitions = 0.

---

## LaTeX Syntax

Use standard LaTeX delimiters in card content:

```
Inline:  The loss is $L = -\sum y_i \log p_i$
Display: $$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
```

The AI generator is prompted to use LaTeX for all mathematical expressions automatically.

---

## Changelog

See git log for full history:

```bash
git log --oneline
```

| Commit | Description |
|---|---|
| `feat: LaTeX rendering via KaTeX` | KaTeX renderer, updated seed cards and AI prompt |
| `feat: deck management, AI generation, persistent storage` | Decks, AI generator, window.storage |
| `feat: initial SM-2 flashcard app` | Core app with SM-2, flip cards, category filter |

---

## License

MIT
