# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mlcards** is a modular React spaced-repetition flashcard app for machine learning education. It was originally designed to run as a **Claude.ai Artifact** and now ships with a full Vite local dev setup.

## Running Locally

```bash
npm install
npm run dev        # http://localhost:5173
```

Node.js is installed at `~/.local/bin/node` (added to `~/.zshrc`). Use `~/.local/bin/npm` if `npm` is not yet on PATH in the current shell.

**API key** (optional — only needed for AI card/distractor generation): copy `.env.example` to `.env.local` and fill in `VITE_ANTHROPIC_API_KEY`. All seed content works fully offline without a key.

There are no lint or test commands.

## Architecture

The app is split into focused modules under `src/`:

```
src/
  main.jsx                  Entry point — mounts App into #root
  App.jsx                   Main component: state, effects, handlers, all views
  utils/
    sm2.js                  SM-2 spaced-repetition algorithm (sm2, initCard)
    storage.js              Storage adapter + distractor cache helpers
    api.js                  anthropicHeaders, generateCards, generateDistractors, shuffle
  constants/
    index.js                SEED_DECKS (raw), DECK_COLORS, DECK_ICONS, QUALITY_BTNS
  components/
    LatexRenderer.jsx       KaTeX rendering (npm package, synchronous)
    FlipCard.jsx            Flip-mode review card + learn more + rating buttons
    MCView.jsx              Multiple-choice review mode
public/
  distractors.json          Offline distractors for all seed cards
  resources.json            "Learn More" links for all seed cards (all 4 decks)
```

### Core Modules

1. **`src/utils/sm2.js`** — Pure functions, no React. `sm2(card, quality)` implements spaced-repetition. Quality: 0=Blackout, 1=Hard, 2=Good, 3=Easy. `initCard(card)` creates fresh SM-2 metadata.

2. **`src/utils/storage.js`** — `window.storage` (artifact API) with automatic fallback to `localStorage`. `loadDecks(SEED_DECKS, initCard)` implements smarter seeding: merges new seed decks into returning users' data by comparing deck IDs. Keys:
   - `mlcards:decks` — full deck/card data
   - `mlcards:distractors` — `{ [cardId]: string[] }` user-cached distractors

3. **`src/utils/api.js`** — `anthropicHeaders()` builds headers with `x-api-key` from `import.meta.env.VITE_ANTHROPIC_API_KEY`. `generateCards()` and `generateDistractors()` call `claude-sonnet-4-20250514`.

4. **`src/constants/index.js`** — Raw `SEED_DECKS` (no `initCard` applied — `loadDecks` handles that), plus palette and quality button constants.

5. **`src/components/LatexRenderer.jsx`** — Imports `katex` from npm (bundled, synchronous — no CDN loading, no async state). Parses `$...$` (inline) and `$$...$$` (display) delimiters.

6. **`src/components/FlipCard.jsx`** — Props: `{ current, flipped, setFlipped, activeDeckData, cardResources, learnMoreOpen, setLearnMoreOpen, handleRate }`. Renders the 3D flip card, learn-more section, and quality rating buttons.

7. **`src/components/MCView.jsx`** — Props: `{ current, mcState, handleMcSelect, handleMcNext, loadMcChoices, activeDeck, getDeck, sessionStats, qIdx, queueLength }`. Renders the question box and multiple-choice answer buttons.

8. **`src/App.jsx`** — All state management, effects, handlers, and view routing. Generate form state uses `useReducer` (`genInitial`, `genReducer`, `dispatchGen`). The CSS `<style>` block lives here with named classes for flip card, MC, and generate view elements.

### Generate State (useReducer)

```js
const genInitial = { topic: "", count: 5, deckId: "new", deckName: "", generating: false, error: "" };
// Actions: SET { key, value } | START | ERROR { error } | RESET
const [gen, dispatchGen] = useReducer(genReducer, genInitial);
```

Access: `gen.topic`, `gen.count`, `gen.deckId`, `gen.deckName`, `gen.generating`, `gen.error`.
Dispatch: `dispatchGen({ type: "SET", key: "topic", value: v })` etc.

### Static JSON files (`public/`)

These are fetched at app startup and merged with any user-persisted data:

| File | Purpose |
|---|---|
| `public/distractors.json` | `{ [cardId]: string[] }` — 3 offline distractors per seed card. Merged with `mlcards:distractors` from storage; user entries take priority. |
| `public/resources.json` | `{ [cardId]: [{title, url, type}] }` — "Learn More" links per card. `type` is one of `paper`, `blog`, `wiki`, `notes`. Populated for all 4 seed decks. |

To add resources for new cards, add entries to `public/resources.json` keyed by card ID. To add offline distractors, add entries to `public/distractors.json`.

### Seed Decks (`SEED_DECKS` constant)

Four decks defined in `src/constants/index.js` as raw card objects (no SM-2 fields). `loadDecks()` applies `initCard` at load time:

| ID | Name | Cards |
|---|---|---|
| `deck-1` | ML Fundamentals | 3 (c1–c3) |
| `deck-2` | Deep Learning | 13 (c4–c6, dl-1–dl-10) |
| `deck-3` | Classical ML | 10 (cml-1–cml-10) |
| `deck-4` | Probability & Statistics | 10 (ps-1–ps-10) |

Card IDs are stable keys used across `distractors.json` and `resources.json`. When adding new seed cards, assign a new prefixed ID (e.g. `cml-11`) and add matching entries to both JSON files if applicable.

**Smarter seeding:** Returning users with existing storage will also receive new seed decks — `loadDecks` compares deck IDs and appends any missing seed decks.

### Data Flow

```
App start → fetch distractors.json + resources.json → merge with storage
         → loadDecks(SEED_DECKS, initCard) → smart merge with storage
         → Home view → Deck view → Review (flip or MC) → SM-2 update → Save
                    └→ Generate view → AI cards → auto-cache distractors → Save
```

### "Learn More" UI

In the Review (flip) view, after the card is flipped, a `▸ learn more` toggle appears if `cardResources[current.id]` has entries. Expanding it shows links with color-coded type badges. `learnMoreOpen` resets to `false` via a `useEffect` on `qIdx`.

### Theming (Light / Dark mode)

The app supports light and dark mode via a `lightMode` state variable (persisted to `localStorage` under `mlcards:theme`).

A `themeVars` object maps CSS custom property names (`--th-*`) to hex/rgba values for each mode. The root `<div>` spreads `...themeVars` as inline style, making all vars available to children. The embedded `<style>` block and JSX inline styles use `var(--th-*)` exclusively — no hardcoded theme colors in markup.

Key variable groups:
- `--th-bg`, `--th-bg-card`, `--th-bg-card-alt`, `--th-bg-nav`, `--th-bg-hover`, `--th-bg-active`, `--th-bg-overlay`
- `--th-border`, `--th-border-soft`, `--th-border-nav`, `--th-border-focus`
- `--th-text`, `--th-text-answer`, `--th-text-muted`, `--th-text-dim`, `--th-text-subtle`, `--th-text-faint`, `--th-text-ghost`
- `--th-grid`, `--th-scrollbar`, `--th-tooltip-bg`, `--th-tooltip-color`, `--th-glow1`, `--th-glow2`

Semantic colors (green for correct/mastered, red for errors/delete, orange for due, brand `#0ea5e9` accent) remain as hardcoded hex values since they are not theme-dependent.

### Styling

Embedded CSS in `App.jsx`'s `<style>` block (no framework). Named CSS classes for key UI regions:
- `.card-front-face`, `.card-question-label`, `.card-tap-hint`, `.card-sm2-stats` — flip card elements
- `.learn-more-btn`, `.resource-link` — learn more section
- `.mc-question-box` — MC question container
- `.recall-label`, `.recall-hint` — flip card post-flip UI
- `.review-counter` — progress counter
- `.generate-panel`, `.gen-label` — generate view form

Fonts from Google Fonts: DM Mono + Cormorant Garamond. 3D card flip via CSS `perspective` + `rotateY(180deg)`. Responsive grid with `minmax()`.

## Known Issues / Future Work

- **No manual card editor** — cards can only be created via AI generation or by editing seed data directly.
- **No card deletion** — individual cards cannot be deleted, only entire decks.
- **SM-2 on MC mode is coarse** — correct/wrong maps to quality 3/1. A finer-grained rating after MC reveal would be more accurate.
- **Distractor cache is not invalidated** — if a card's `back` is edited, its cached distractors become stale. Cache should be keyed on a hash of `card.back`, not just `card.id`.
- **`distractorCache` ref in `loadMcChoices`** — the function closes over `distractorCache` state. If called rapidly it could read stale cache. Use a ref to mirror the cache value if this becomes a problem.

## Data Shapes

```js
// Deck
{
  id: "deck-{timestamp}",
  name: string,
  color: string,   // hex, from DECK_COLORS
  icon: string,    // from DECK_ICONS
  created: number, // Date.now()
  cards: Card[]
}

// Card (includes SM-2 fields after initCard())
{
  id: string,
  front: string,   // may contain LaTeX $...$ or $$...$$
  back: string,    // may contain LaTeX, \n for line breaks
  ef: number,      // easiness factor, min 1.3, default 2.5
  interval: number,// days until next review
  repetitions: number,
  nextReview: number // Date.now() + interval * 86400000
}

// Distractor cache
{ [cardId: string]: string[] }  // 3 strings per card
```
