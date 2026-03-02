import { initCard } from "./sm2.js";

export function anthropicHeaders() {
  const key = typeof import.meta !== "undefined" && import.meta.env?.VITE_ANTHROPIC_API_KEY;
  return {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
    "anthropic-dangerous-direct-browser-access": "true",
    ...(key ? { "x-api-key": key } : {}),
  };
}

export async function generateCards(topic, count, deckName) {
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
    headers: anthropicHeaders(),
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

export async function generateDistractors(card, allCards) {
  const siblings = allCards
    .filter(c => c.id !== card.id)
    .sort(() => Math.random() - 0.5)
    .slice(0, 5)
    .map(c => c.back);

  const prompt = `You are an expert ML educator creating a multiple choice question.

Question: ${card.front}
Correct answer: ${card.back}

Generate exactly 3 plausible but INCORRECT answer choices for this question.
They must be:
- Realistic enough to challenge someone who partially understands the topic
- Clearly wrong to someone who truly knows the answer
- Similar in length and style to the correct answer
- For math questions, use LaTeX with $...$ and $$...$$ notation just like the correct answer
${siblings.length ? `\nContext (other cards in this deck for distractor inspiration):\n${siblings.map((s, i) => `${i + 1}. ${s}`).join('\n')}` : ''}

Return ONLY valid JSON, no other text:
{"distractors": ["wrong answer 1", "wrong answer 2", "wrong answer 3"]}`;

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: anthropicHeaders(),
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
  return parsed.distractors;
}

export function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
