export function sm2(card, quality) {
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

export function initCard(card) {
  return { ...card, ef: 2.5, interval: 1, repetitions: 0, nextReview: Date.now() };
}
