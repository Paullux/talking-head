// Embodied voice-avatar backend.
//   /api/say  { text }        -> Piper TTS -> Rhubarb visemes -> { audio(base64 wav), cues }
//   /api/chat { text }        -> external LLM (OpenAI-compatible) -> reply -> say() -> { reply, audio, cues }
// The 3D avatar + STT (Web Speech API) live in the browser; this only does
// text -> voice -> visemes and the LLM call (key stays server-side).

import express from "express";
import rateLimit from "express-rate-limit";
import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID, timingSafeEqual, createHash } from "node:crypto";

const execFileP = promisify(execFile);
const __dirname = dirname(fileURLToPath(import.meta.url));

const {
  PORT = 3000,
  PIPER_BIN = "piper",
  PIPER_VOICE = "/opt/voices/fr_FR-siwis-medium.onnx",
  RHUBARB_BIN = "rhubarb",
  RHUBARB_RECOGNIZER = "phonetic",
  LLM_API_BASE = "https://api.mistral.ai/v1",
  LLM_API_KEY = "",
  LLM_MODEL = "mistral-small-latest",
  LLM_VISION_MODEL = "",
  LLM_MAX_TOKENS = "220",
  SYSTEM_PROMPT = "Tu t'appelles Nora, une assistante incarnée dans un avatar 3D. Tu es vive, chaleureuse et un peu taquine. Tu réponds en français, à l'oral, en 1 à 3 phrases courtes et naturelles. Pas de listes, pas de markdown, pas d'emojis.",
  DIAG_PASSWORD = "",
} = process.env;
const VISION_MODEL = LLM_VISION_MODEL || LLM_MODEL;

// max simultaneous Piper/Rhubarb pipelines — this VPS has 1 vCPU, so
// running several at once degrades everyone instead of queuing fairly.
const MAX_CONCURRENT_SAY = 2;
let activeSay = 0;

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
    // Piper: text on stdin -> wav
    logStep(id, "tts:piper:start", `chars=${text.length}`);
    await spawnWithInput(PIPER_BIN, ["--model", PIPER_VOICE, "--output_file", raw], text);
    timings.piperMs = Date.now() - t0;
    logStep(id, "tts:piper:ok", `ms=${timings.piperMs}`);
    // visemes -- Rhubarb decodes WAV itself (any sample rate / channel count),
    // no ffmpeg pre-conversion needed. Verified directly: Piper's native output
    // (mono, its model's native rate, e.g. 22050Hz) feeds Rhubarb fine as-is.
    const rhubarbT0 = Date.now();
    logStep(id, "lipsync:rhubarb:start", `recognizer=${RHUBARB_RECOGNIZER}`);
    const { stdout } = await execFileP(RHUBARB_BIN,
      ["-f", "json", "--recognizer", RHUBARB_RECOGNIZER, "--extendedShapes", "GHX", raw],
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

async function compactMemory(summary = "", history = [], id = shortId()) {
  if (!LLM_API_KEY.trim()) {
    throw new Error("LLM_API_KEY manquante cote serveur");
  }
  const t0 = Date.now();
  logStep(id, "compact:start", `turns=${history.length}`);
  const res = await fetch(`${LLM_API_BASE}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${LLM_API_KEY}` },
    body: JSON.stringify({
      model: LLM_MODEL,
      max_tokens: 180,
      temperature: 0.2,
      messages: [
        {
          role: "system",
          content: "Résume en français, en moins de 900 caractères, uniquement les faits utiles pour continuer une conversation avec Nora: préférences de l'utilisateur, décisions de design, contexte durable. Ignore les hésitations, détails temporaires et données sensibles. Ne fais pas de markdown.",
        },
        {
          role: "user",
          content: JSON.stringify({
            resume_actuel: String(summary || "").slice(0, 1200),
            derniers_messages: history.slice(-12),
          }),
        },
      ],
    }),
  });
  if (!res.ok) throw new Error(`LLM ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const compact = (data.choices?.[0]?.message?.content || "").trim().slice(0, 1000);
  logStep(id, "compact:ok", `ms=${Date.now() - t0} chars=${compact.length}`);
  return compact;
}

function assertSafePublicUrl(rawUrl) {
  let parsed;
  try { parsed = new URL(rawUrl); }
  catch { throw new Error("URL invalide"); }
  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("Seules les URLs http(s) sont acceptées");
  }
  const host = parsed.hostname.toLowerCase();
  if (
    host === "localhost" ||
    host.endsWith(".localhost") ||
    host === "127.0.0.1" ||
    host === "::1" ||
    host.startsWith("10.") ||
    host.startsWith("192.168.") ||
    /^172\.(1[6-9]|2\d|3[0-1])\./.test(host) ||
    /^169\.254\./.test(host)
  ) {
    throw new Error("URL locale ou privée refusée");
  }
  return parsed.toString();
}

function normalizeImageDataUrl(dataUrl) {
  const value = String(dataUrl || "");
  if (!/^data:image\/(png|jpeg|jpg|webp|gif);base64,[a-z0-9+/=\r\n]+$/i.test(value)) {
    throw new Error("Image invalide: formats acceptés png, jpeg, webp ou gif");
  }
  if (value.length > 7_000_000) {
    throw new Error("Image trop grande: limite environ 5 Mo");
  }
  return value;
}

async function analyzeImage({ imageUrl, prompt = "", history = [] }, id = shortId()) {
  if (!LLM_API_KEY.trim()) {
    throw new Error("LLM_API_KEY manquante cote serveur");
  }
  const t0 = Date.now();
  const safePrompt = String(prompt || "").slice(0, 700) || "Analyse cette image en français.";
  logStep(id, "vision:start", `model=${VISION_MODEL}`);
  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    ...history,
    {
      role: "user",
      content: [
        { type: "text", text: `${safePrompt}\nRéponds comme Nora, en 1 à 3 phrases naturelles.` },
        { type: "image_url", image_url: imageUrl },
      ],
    },
  ];
  const res = await fetch(`${LLM_API_BASE}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${LLM_API_KEY}` },
    body: JSON.stringify({ model: VISION_MODEL, messages, max_tokens: Number(LLM_MAX_TOKENS), temperature: 0.6 }),
  });
  if (!res.ok) throw new Error(`LLM ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const reply = data.choices?.[0]?.message?.content?.trim() || "Je n'arrive pas à lire cette image.";
  logStep(id, "vision:ok", `ms=${Date.now() - t0} chars=${reply.length}`);
  return reply;
}

function htmlToPlainText(html) {
  return String(html || "")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, "\"")
    .replace(/&#39;/gi, "'")
    .replace(/\s+/g, " ")
    .trim();
}

async function analyzeLink({ url, prompt = "", history = [] }, id = shortId()) {
  const safeUrl = assertSafePublicUrl(url);
  logStep(id, "link:fetch", safeUrl.slice(0, 120));
  const response = await fetch(safeUrl, {
    redirect: "follow",
    signal: AbortSignal.timeout(10_000),
    headers: { "user-agent": "Nora/0.1 link analyzer" },
  });
  if (!response.ok) throw new Error(`URL HTTP ${response.status}`);
  const type = (response.headers.get("content-type") || "").toLowerCase();
  if (type.startsWith("image/")) {
    return analyzeImage({ imageUrl: safeUrl, prompt, history }, id);
  }
  const html = await response.text();
  const text = htmlToPlainText(html).slice(0, 40_000);
  if (!text) throw new Error("Aucun texte lisible trouvé dans ce lien");
  const userText = [
    String(prompt || "Analyse ce lien en français.").slice(0, 700),
    "Contenu du lien:",
    text,
  ].join("\n\n");
  return chat(userText.slice(0, 10_000), history, id);
}

// ---- server -------------------------------------------------------------
const app = express();
// behind Coolify/Traefik: trust the proxy's X-Forwarded-For so rate-limiting
// (and req.ip generally) sees the real client IP, not the proxy's.
app.set("trust proxy", 1);
app.use(express.json({ limit: "8mb" }));
app.use((req, res, next) => {
  if (req.path === "/" || req.path.endsWith(".html")) {
    res.setHeader("Cache-Control", "no-store");
  }
  next();
});

// ---- shared-secret gate for the diagnostics page (fail-closed) -----------
function timingSafeStrEqual(a, b) {
  const ha = createHash("sha256").update(a).digest();
  const hb = createHash("sha256").update(b).digest();
  return timingSafeEqual(ha, hb);
}
app.get("/_diag.html", (req, res, next) => {
  if (!DIAG_PASSWORD) return res.status(403).send("Diagnostics désactivés (DIAG_PASSWORD non configuré).");
  const auth = req.headers.authorization || "";
  const [scheme, encoded] = auth.split(" ");
  const [, pass] = scheme === "Basic" && encoded
    ? Buffer.from(encoded, "base64").toString("utf8").split(":")
    : [null, null];
  if (pass && timingSafeStrEqual(pass, DIAG_PASSWORD)) return next();
  res.set("WWW-Authenticate", 'Basic realm="Nora diagnostics"');
  res.status(401).send("Authentification requise.");
});

app.use(express.static(join(__dirname, "public")));

// ---- rate limits (per client IP) -----------------------------------------
// heavy = Piper + Rhubarb (/api/say, /api/chat); light = LLM only.
const heavyBurst = rateLimit({ windowMs: 30_000, max: 3, standardHeaders: true, legacyHeaders: false });
const heavySustained = rateLimit({ windowMs: 5 * 60_000, max: 10, standardHeaders: true, legacyHeaders: false });
const lightBurst = rateLimit({ windowMs: 30_000, max: 6, standardHeaders: true, legacyHeaders: false });
const lightSustained = rateLimit({ windowMs: 5 * 60_000, max: 20, standardHeaders: true, legacyHeaders: false });

// concurrency guard: reject immediately rather than queue (predictable UX)
function concurrencyGuard(req, res, next) {
  if (activeSay >= MAX_CONCURRENT_SAY) {
    return res.status(429).json({ error: "Nora est occupée, réessaie dans un instant." });
  }
  activeSay++;
  let released = false;
  const release = () => { if (!released) { released = true; activeSay = Math.max(0, activeSay - 1); } };
  res.on("finish", release);
  res.on("close", release);
  next();
}

app.get("/api/health", (_req, res) => res.json({
  ok: true,
  version: "social-assets-2026-07-13",
  llm: {
    configured: Boolean(LLM_API_KEY.trim()),
    apiBase: LLM_API_BASE,
    model: LLM_MODEL,
  },
}));

app.get("/api/version", (_req, res) => res.json({
  ok: true,
  version: "social-assets-2026-07-13",
  endpoints: {
    health: "GET /api/health",
    chatText: "POST /api/chat-text",
    compact: "POST /api/compact",
    analyzeImage: "POST /api/analyze-image",
    analyzeLink: "POST /api/analyze-link",
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

app.get("/api/compact", (_req, res) => res.status(405).json({
  error: "Utilise POST /api/compact avec JSON {\"summary\":\"...\",\"history\":[]}",
}));

app.get("/api/analyze-image", (_req, res) => res.status(405).json({
  error: "Utilise POST /api/analyze-image avec JSON {\"image\":\"data:image/...\",\"prompt\":\"...\",\"history\":[]}",
}));

app.get("/api/analyze-link", (_req, res) => res.status(405).json({
  error: "Utilise POST /api/analyze-link avec JSON {\"url\":\"https://...\",\"prompt\":\"...\",\"history\":[]}",
}));

app.post("/api/say", heavyBurst, heavySustained, concurrencyGuard, async (req, res) => {
  const id = shortId();
  try {
    logStep(id, "api:say");
    res.json(await say(String(req.body.text || "").slice(0, 800), id));
  } catch (e) {
    logStep(id, "api:say:error", String(e.message || e));
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.post("/api/chat-text", lightBurst, lightSustained, async (req, res) => {
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

app.post("/api/compact", lightBurst, lightSustained, async (req, res) => {
  const id = shortId();
  try {
    logStep(id, "api:compact");
    const summary = await compactMemory(
      String(req.body.summary || "").slice(0, 1200),
      Array.isArray(req.body.history) ? req.body.history : [],
      id,
    );
    res.json({ summary });
  } catch (e) {
    logStep(id, "api:compact:error", String(e.message || e));
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.post("/api/analyze-image", lightBurst, lightSustained, async (req, res) => {
  const id = shortId();
  try {
    logStep(id, "api:analyze-image");
    const reply = await analyzeImage({
      imageUrl: normalizeImageDataUrl(req.body.image),
      prompt: req.body.prompt,
      history: Array.isArray(req.body.history) ? req.body.history : [],
    }, id);
    res.json({ reply });
  } catch (e) {
    logStep(id, "api:analyze-image:error", String(e.message || e));
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.post("/api/analyze-link", lightBurst, lightSustained, async (req, res) => {
  const id = shortId();
  try {
    logStep(id, "api:analyze-link");
    const reply = await analyzeLink({
      url: req.body.url,
      prompt: req.body.prompt,
      history: Array.isArray(req.body.history) ? req.body.history : [],
    }, id);
    res.json({ reply });
  } catch (e) {
    logStep(id, "api:analyze-link:error", String(e.message || e));
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.post("/api/chat", heavyBurst, heavySustained, concurrencyGuard, async (req, res) => {
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
