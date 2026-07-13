// Quick check that the external LLM (Mistral by default) answers.
// Run WITHOUT putting your key in the file:
//   Git Bash / Linux :  LLM_API_KEY=xxxxx node test_llm.js
//   PowerShell       :  $env:LLM_API_KEY="xxxxx"; node test_llm.js
// Mirrors exactly the request server.js sends.

const {
  LLM_API_BASE = "https://api.mistral.ai/v1",
  LLM_API_KEY = "",
  LLM_MODEL = "mistral-small-latest",
  SYSTEM_PROMPT = "Tu es une assistante incarnée, vive et un peu taquine. Tu réponds en français, à l'oral, en 1 à 3 phrases courtes. Pas de listes, pas de markdown.",
} = process.env;

if (!LLM_API_KEY) {
  console.error("❌ LLM_API_KEY manquante. Ex: LLM_API_KEY=xxxxx node test_llm.js");
  process.exit(1);
}

const question = process.argv[2] || "Salut, présente-toi en une phrase.";

const t0 = Date.now();
const res = await fetch(`${LLM_API_BASE}/chat/completions`, {
  method: "POST",
  headers: { "content-type": "application/json", authorization: `Bearer ${LLM_API_KEY}` },
  body: JSON.stringify({
    model: LLM_MODEL,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: question },
    ],
    max_tokens: 220,
    temperature: 0.8,
  }),
});

console.log("HTTP", res.status, res.statusText, `(${Date.now() - t0} ms)`);
const data = await res.json();
if (!res.ok) {
  console.error("❌ Erreur:", JSON.stringify(data, null, 2));
  process.exit(1);
}
console.log("Modèle :", data.model);
console.log("Q:", question);
console.log("R:", data.choices?.[0]?.message?.content?.trim());
console.log("tokens:", JSON.stringify(data.usage));
