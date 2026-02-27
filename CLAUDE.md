# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mlcards** is a single-file React spaced-repetition flashcard app for machine learning education. The entire application lives in `ml-flashcards.jsx` (≈950 lines). It is designed to run as a **Claude.ai Artifact**, where React 18, KaTeX, and `window.storage` are provided by the artifact environment.

## Running Locally

There is no `package.json`. To run locally, scaffold manually:

```bash
npm create vite@latest mlcards -- --template react
cd mlcards
npm install
# Replace src/App.jsx with ml-flashcards.jsx
npm run dev
```

When running locally, `window.storage` must be replaced with `localStorage` wrappers in `loadDecks()`/`saveDecks()`/`loadDistractorCache()`/`saveDistractorCache()`, and an Anthropic API key must be added to the `generateCards()` fetch headers.

There are no build, lint, or test commands in this repository.

## Architecture

The app is a **monolithic React component** (`App`) with no sub-files. All logic is co-located in `ml-flashcards.jsx`:

### Core Modules (in order of definition)

1. **LaTeX Renderer** — Dynamically loads KaTeX 0.16.9 from CDN. `LatexRenderer` component parses `$...$` (inline) and `$$...$$` (display) delimiters and renders via `window.katex.renderToString()`.

2. **SM-2 Algorithm** — `sm2(card, quality)` implements the standard spaced-repetition algorithm. Quality: 0=Blackout, 1=Hard, 2=Good, 3=Easy. Updates easiness factor, interval, and `nextReview` date. `initCard()` creates fresh card metadata.

3. **Storage Layer** — Uses `window.storage` (artifact API, not `localStorage`).
   - Decks: `mlcards:decks`
   - Distractor cache: `mlcards:distractors` (keyed by card ID, value = `string[]`)
   - Both support export/import as `.json` files.

4. **AI Generation** — Two Claude API calls:
   - `generateCards(topic, count, deckName)` → `POST /v1/messages` → `claude-sonnet-4-20250514` → `{cards: [{front, back}]}`
   - `generateDistractors(card, allCards)` → generates 3 plausible-but-wrong answers, uses sibling cards as context, auto-cached.

5. **State Management** — All state in `useState` hooks in `App`. Key variables: `decks`, `view` (`"home" | "deck" | "review" | "generate"`), `reviewMode` (`"flip" | "mc"`), `queue`, `qIdx`, `flipped`, `mcState`, `distractorCache`, `sessionStats`.

6. **UI Views** — Four views rendered conditionally: Home (deck grid), Deck (card list + distractor manager), Review (flip or MC mode), Generate (AI form).

### Data Flow

```
Persist decks → Home view → Deck view → Review (flip or MC) → SM-2 update → Save
                         └→ Generate view → AI cards → auto-cache distractors → Save
```

### Styling

Embedded CSS only (no framework). Dark theme (`#060a10` bg). Fonts loaded from Google Fonts: DM Mono + Cormorant Garamond. 3D card flip via CSS perspective + `rotateY(180deg)`. Responsive grid with `minmax()`.

### Seed Data

Two pre-loaded decks ("ML Fundamentals", "Deep Learning") defined in `loadDecks()` as fallback when storage is empty. Cards include LaTeX formulas.
