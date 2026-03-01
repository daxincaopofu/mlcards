# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mlcards** is a single-file React spaced-repetition flashcard app for machine learning education. The entire application lives in `ml-flashcards.jsx` (~1100 lines). It was originally designed to run as a **Claude.ai Artifact** but now ships with a full Vite local dev setup.

## Running Locally

```bash
npm install
npm run dev        # http://localhost:5173
```

Node.js is installed at `~/.local/bin/node` (added to `~/.zshrc`). Use `~/.local/bin/npm` if `npm` is not yet on PATH in the current shell.

**API key** (optional — only needed for AI card/distractor generation): copy `.env.example` to `.env.local` and fill in `VITE_ANTHROPIC_API_KEY`. All seed content works fully offline without a key.

There are no lint or test commands.

## Architecture

The app is a **monolithic React component** (`App`) with no sub-files. All logic is co-located in `ml-flashcards.jsx`.

### Core Modules (in order of definition)

1. **LaTeX Renderer** — Dynamically loads KaTeX 0.16.9 from CDN. `LatexRenderer` component parses `$...$` (inline) and `$$...$$` (display) delimiters and renders via `window.katex.renderToString()`.

2. **SM-2 Algorithm** — `sm2(card, quality)` implements spaced-repetition. Quality: 0=Blackout, 1=Hard, 2=Good, 3=Easy. Updates easiness factor, interval, and `nextReview`. `initCard()` creates fresh card metadata.

3. **Storage Layer** — `window.storage` (artifact API) with automatic fallback to `localStorage` via an adapter defined at module level. Keys:
   - `mlcards:decks` — full deck/card data
   - `mlcards:distractors` — `{ [cardId]: string[] }` user-cached distractors (merged on top of bundled ones)

4. **AI Generation** — `anthropicHeaders()` builds headers including `x-api-key` from `import.meta.env.VITE_ANTHROPIC_API_KEY`. Two API calls:
   - `generateCards(topic, count, deckName)` → `claude-sonnet-4-20250514` → `{cards: [{front, back}]}`
   - `generateDistractors(card, allCards)` → 3 plausible-but-wrong answers, auto-cached

5. **State Management** — All state in `useState` hooks in `App`. Key variables: `decks`, `view` (`"home"|"deck"|"review"|"generate"`), `reviewMode` (`"flip"|"mc"`), `queue`, `qIdx`, `flipped`, `mcState`, `distractorCache`, `cardResources`, `learnMoreOpen`, `sessionStats`.

6. **UI Views** — Four views rendered conditionally: Home (deck grid), Deck (card list + distractor manager), Review (flip or MC mode), Generate (AI form).

### Static JSON files (`public/`)

These are fetched at app startup and merged with any user-persisted data:

| File | Purpose |
|---|---|
| `public/distractors.json` | `{ [cardId]: string[] }` — 3 offline distractors per seed card. Merged with `mlcards:distractors` from storage; user entries take priority on key conflicts. |
| `public/resources.json` | `{ [cardId]: [{title, url, type}] }` — "Learn More" links per card. `type` is one of `paper`, `blog`, `wiki`, `notes`. Currently populated for the Deep Learning deck only. |

To add resources for new cards, add entries to `public/resources.json` keyed by card ID. To add offline distractors, add entries to `public/distractors.json`.

### Seed Decks (`SEED_DECKS` constant)

Four decks defined directly in `ml-flashcards.jsx`, used as fallback when storage is empty:

| ID | Name | Cards |
|---|---|---|
| `deck-1` | ML Fundamentals | 3 (c1–c3) |
| `deck-2` | Deep Learning | 13 (c4–c6, dl-1–dl-10) |
| `deck-3` | Classical ML | 10 (cml-1–cml-10) |
| `deck-4` | Probability & Statistics | 10 (ps-1–ps-10) |

Card IDs are stable keys used across `distractors.json` and `resources.json`. When adding new seed cards, assign a new prefixed ID (e.g. `cml-11`) and add matching entries to both JSON files if applicable.

**Note:** Returning users who already have decks persisted in storage will not see new seed decks automatically — they only load when storage is empty.

### Data Flow

```
App start → fetch distractors.json + resources.json → merge with storage
         → load decks (storage or SEED_DECKS fallback)
         → Home view → Deck view → Review (flip or MC) → SM-2 update → Save
                    └→ Generate view → AI cards → auto-cache distractors → Save
```

### "Learn More" UI

In the Review (flip) view, after the card is flipped, a `▸ learn more` toggle appears if `cardResources[current.id]` has entries. Expanding it shows links with color-coded type badges. `learnMoreOpen` resets to `false` via a `useEffect` on `qIdx`.

### Styling

Embedded CSS only (no framework). Dark theme (`#060a10` bg). Fonts from Google Fonts: DM Mono + Cormorant Garamond. 3D card flip via CSS `perspective` + `rotateY(180deg)`. Responsive grid with `minmax()`.
