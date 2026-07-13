// Embodied voice-avatar backend.
//   /api/say  { text }        -> Piper TTS -> Rhubarb visemes -> { audio(base64 wav), cues }
//   /api/chat { text }        -> external LLM (OpenAI-compatible) -> reply -> say() -> { reply, audio, cues }
// The 3D avatar + STT (Web Speech API) live in the browser; this only does
// text -> voice -> visemes and the LLM call (key stays server-side).

import express from "express";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const execFileP = promisify(execFile);
const __dirname = dirname(fileURLToPath(import.meta.url));

const {
  PORT = 3000,
  PIPER_BIN = "piper",
  PIPER_VOICE = "/opt/voices/fr_FR-siwis-medium.onnx",
  RHUBARB_BIN = "rhubarb",
  FFMPEG_BIN = "ffmpeg",
  LLM_API_BASE = "https://api.mistral.ai/v1",
  LLM_API_KEY = "",
  LLM_MODEL = "mistral-small-latest",
  LLM_MAX_TOKENS = "220",
  SYSTEM_PROMPT = "Tu es une assistante incarnée, vive et un peu taquine. Tu réponds en français, à l'oral, en 1 à 3 phrases courtes. Pas de listes, pas de markdown.",
} = process.env;

// ---- text -> voice(wav) -> visemes(cues) --------------------------------
async function say(text) {
  const dir = await mkdtemp(join(tmpdir(), "say-"));
  try {
    const raw = join(dir, "raw.wav");
    const wav16 = join(dir, "v16.wav");
    // Piper: text on stdin -> wav
    await execFileP(PIPER_BIN, ["--model", PIPER_VOICE, "--output_file", raw],
      { input: text, maxBuffer: 64 * 1024 * 1024 });
    // normalise for Rhubarb (mono 16k PCM)
    await execFileP(FFMPEG_BIN, ["-y", "-i", raw, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav16]);
    // visemes
    const { stdout } = await execFileP(RHUBARB_BIN,
      ["-f", "json", "--extendedShapes", "GHX", wav16], { maxBuffer: 16 * 1024 * 1024 });
    const cues = JSON.parse(stdout).mouthCues;
    const audio = await readFile(raw);
    return { audio: audio.toString("base64"), mime: "audio/wav", cues };
  } finally {
    rm(dir, { recursive: true, force: true }).catch(() => {});
  }
}

// ---- external LLM (OpenAI-compatible chat/completions) -------------------
async function chat(userText, history = []) {
  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    ...history,
    { role: "user", content: userText },
  ];
  const res = await fetch(`${LLM_API_BASE}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${LLM_API_KEY}` },
    body: JSON.stringify({ model: LLM_MODEL, messages, max_tokens: Number(LLM_MAX_TOKENS), temperature: 0.8 }),
  });
  if (!res.ok) throw new Error(`LLM ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return data.choices?.[0]?.message?.content?.trim() || "…";
}

// ---- server -------------------------------------------------------------
const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(express.static(join(__dirname, "public")));

app.get("/api/health", (_req, res) => res.json({ ok: true }));

app.post("/api/say", async (req, res) => {
  try {
    res.json(await say(String(req.body.text || "").slice(0, 800)));
  } catch (e) { res.status(500).json({ error: String(e.message || e) }); }
});

app.post("/api/chat", async (req, res) => {
  try {
    const reply = await chat(String(req.body.text || "").slice(0, 1000), req.body.history || []);
    const voice = await say(reply);
    res.json({ reply, ...voice });
  } catch (e) { res.status(500).json({ error: String(e.message || e) }); }
});

app.listen(PORT, () => console.log(`avatar backend on :${PORT}`));
