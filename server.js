// Embodied voice-avatar backend.
//   /api/say  { text }        -> Piper TTS -> Rhubarb visemes -> { audio(base64 wav), cues }
//   /api/chat { text }        -> external LLM (OpenAI-compatible) -> reply -> say() -> { reply, audio, cues }
// The 3D avatar + STT (Web Speech API) live in the browser; this only does
// text -> voice -> visemes and the LLM call (key stays server-side).

import express from "express";
import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";

const execFileP = promisify(execFile);
const __dirname = dirname(fileURLToPath(import.meta.url));

const {
  PORT = 3000,
  PIPER_BIN = "piper",
  PIPER_VOICE = "/opt/voices/fr_FR-siwis-medium.onnx",
  RHUBARB_BIN = "rhubarb",
  RHUBARB_RECOGNIZER = "phonetic",
  FFMPEG_BIN = "ffmpeg",
  LLM_API_BASE = "https://api.mistral.ai/v1",
  LLM_API_KEY = "",
  LLM_MODEL = "mistral-small-latest",
  LLM_MAX_TOKENS = "220",
  SYSTEM_PROMPT = "Tu t'appelles Nora, une assistante incarnée dans un avatar 3D. Tu es vive, chaleureuse et un peu taquine. Tu réponds en français, à l'oral, en 1 à 3 phrases courtes et naturelles. Pas de listes, pas de markdown, pas d'emojis.",
} = process.env;

// ---- text -> voice(wav) -> visemes(cues) --------------------------------
function shortId() {
  return randomUUID().slice(0, 8);
}

function logStep(id, step, extra = "") {
  console.log(`[${id}] ${step}${extra ? ` ${extra}` : ""}`);
}

function spawnWithInput(file, args, input, { timeoutMs = 30000 } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(file, args, { stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`${file} timeout apres ${timeoutMs}ms`));
    }, timeoutMs);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", chunk => { stdout += chunk; });
    child.stderr.on("data", chunk => { stderr += chunk; });
    child.on("error", err => {
      clearTimeout(timer);
      reject(err);
    });
    child.on("close", code => {
      clearTimeout(timer);
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(`${file} exit ${code}: ${stderr || stdout}`));
    });
    child.stdin.end(input);
  });
}

async function say(text, id = shortId()) {
  const dir = await mkdtemp(join(tmpdir(), "say-"));
  const t0 = Date.now();
  const timings = {};
  try {
    const raw = join(dir, "raw.wav");
    const wav16 = join(dir, "v16.wav");
    // Piper: text on stdin -> wav
    logStep(id, "tts:piper:start", `chars=${text.length}`);
    await spawnWithInput(PIPER_BIN, ["--model", PIPER_VOICE, "--output_file", raw], text);
    timings.piperMs = Date.now() - t0;
    logStep(id, "tts:piper:ok", `ms=${timings.piperMs}`);
    // normalise for Rhubarb (mono 16k PCM)
    const ffmpegT0 = Date.now();
    logStep(id, "tts:ffmpeg:start");
    await execFileP(FFMPEG_BIN, ["-y", "-i", raw, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav16]);
    timings.ffmpegMs = Date.now() - ffmpegT0;
    logStep(id, "tts:ffmpeg:ok", `ms=${timings.ffmpegMs}`);
    // visemes
    const rhubarbT0 = Date.now();
    logStep(id, "lipsync:rhubarb:start", `recognizer=${RHUBARB_RECOGNIZER}`);
    const { stdout } = await execFileP(RHUBARB_BIN,
      ["-f", "json", "--recognizer", RHUBARB_RECOGNIZER, "--extendedShapes", "GHX", wav16],
      { maxBuffer: 16 * 1024 * 1024 });
    timings.rhubarbMs = Date.now() - rhubarbT0;
    const cues = JSON.parse(stdout).mouthCues;
    const audio = await readFile(raw);
    timings.totalMs = Date.now() - t0;
    logStep(id, "say:ok", `ms=${timings.totalMs} cues=${cues.length} audio=${audio.length}`);
    return { audio: audio.toString("base64"), mime: "audio/wav", cues, timings };
  } finally {
    rm(dir, { recursive: true, force: true }).catch(() => {});
  }
}

// ---- external LLM (OpenAI-compatible chat/completions) -------------------
async function chat(userText, history = [], id = shortId()) {
  if (!LLM_API_KEY.trim()) {
    throw new Error("LLM_API_KEY manquante cote serveur");
  }
  const t0 = Date.now();
  logStep(id, "llm:start", `model=${LLM_MODEL} base=${LLM_API_BASE}`);
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
  const reply = data.choices?.[0]?.message?.content?.trim() || "…";
  logStep(id, "llm:ok", `ms=${Date.now() - t0} chars=${reply.length}`);
  return reply;
}

// ---- server -------------------------------------------------------------
const app = express();
app.use(express.json({ limit: "1mb" }));
app.use((req, res, next) => {
  if (req.path === "/" || req.path.endsWith(".html")) {
    res.setHeader("Cache-Control", "no-store");
  }
  next();
});
app.use(express.static(join(__dirname, "public")));

app.get("/api/health", (_req, res) => res.json({
  ok: true,
  version: "diag-tts-timing-2026-07-13",
  llm: {
    configured: Boolean(LLM_API_KEY.trim()),
    apiBase: LLM_API_BASE,
    model: LLM_MODEL,
    keyPrefix: LLM_API_KEY.trim() ? `${LLM_API_KEY.trim().slice(0, 6)}...` : null,
  },
}));

app.get("/api/version", (_req, res) => res.json({
  ok: true,
  version: "diag-tts-timing-2026-07-13",
  endpoints: {
    health: "GET /api/health",
    chatText: "POST /api/chat-text",
    say: "POST /api/say",
    chat: "POST /api/chat",
    diagnosticPage: "GET /_diag.html",
  },
}));

app.get("/api/say", (_req, res) => res.status(405).json({
  error: "Utilise POST /api/say avec JSON {\"text\":\"Bonjour\"}",
}));

app.get("/api/chat", (_req, res) => res.status(405).json({
  error: "Utilise POST /api/chat avec JSON {\"text\":\"Bonjour\",\"history\":[]}",
}));

app.get("/api/chat-text", (_req, res) => res.status(405).json({
  error: "Utilise POST /api/chat-text avec JSON {\"text\":\"Bonjour\",\"history\":[]}",
}));

app.post("/api/say", async (req, res) => {
  const id = shortId();
  try {
    logStep(id, "api:say");
    res.json(await say(String(req.body.text || "").slice(0, 800), id));
  } catch (e) {
    logStep(id, "api:say:error", String(e.message || e));
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.post("/api/chat-text", async (req, res) => {
  const id = shortId();
  try {
    logStep(id, "api:chat-text");
    const reply = await chat(String(req.body.text || "").slice(0, 1000), req.body.history || [], id);
    res.json({ reply });
  } catch (e) {
    logStep(id, "api:chat-text:error", String(e.message || e));
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.post("/api/chat", async (req, res) => {
  const id = shortId();
  try {
    logStep(id, "api:chat");
    const reply = await chat(String(req.body.text || "").slice(0, 1000), req.body.history || [], id);
    const voice = await say(reply, id);
    res.json({ reply, ...voice });
  } catch (e) {
    logStep(id, "api:chat:error", String(e.message || e));
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.listen(PORT, () => console.log(`avatar backend on :${PORT}`));
