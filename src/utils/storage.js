export const storage = window.storage ?? {
  get: async (key) => { const v = localStorage.getItem(key); return v ? { value: v } : null; },
  set: async (key, value) => { localStorage.setItem(key, value); },
};

export async function loadDecks(SEED_DECKS, initCard) {
  const seeded = SEED_DECKS.map(d => ({ ...d, cards: d.cards.map(initCard) }));
  try {
    const r = await storage.get("mlcards:decks");
    if (!r) return seeded;
    const saved = JSON.parse(r.value);
    const existingIds = new Set(saved.map(d => d.id));
    const newDecks = seeded.filter(d => !existingIds.has(d.id));
    return newDecks.length ? [...saved, ...newDecks] : saved;
  } catch { return seeded; }
}

export async function saveDecks(decks) {
  try { await storage.set("mlcards:decks", JSON.stringify(decks)); } catch {}
}

export async function loadDistractorCache() {
  let bundled = {};
  try {
    const res = await fetch("/distractors.json");
    if (res.ok) bundled = await res.json();
  } catch {}
  try {
    const r = await storage.get("mlcards:distractors");
    const saved = r ? JSON.parse(r.value) : {};
    return { ...bundled, ...saved };
  } catch { return bundled; }
}

export async function saveDistractorCache(cache) {
  try { await storage.set("mlcards:distractors", JSON.stringify(cache)); } catch {}
}

export function exportDistractorCache(cache) {
  const json = JSON.stringify(cache, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "mlcards-distractors.json"; a.click();
  URL.revokeObjectURL(url);
}

export function importDistractorFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => {
      try { resolve(JSON.parse(e.target.result)); }
      catch { reject(new Error("Invalid JSON")); }
    };
    reader.onerror = () => reject(new Error("Read failed"));
    reader.readAsText(file);
  });
}
