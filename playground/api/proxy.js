// Vercel serverless proxy: browser -> /api/<path> -> this function -> Laya backend.
// Keeps the backend URL and API key server-side (set them in Vercel project env vars):
//   LAYA_BACKEND_URL      e.g. https://your-name-laya-playground.hf.space
//   LAYA_PLAYGROUND_KEY   same value as on the backend
// Locally, server.py serves /api itself; this file is only used on Vercel.
const ALLOWED = new Set(["health", "route", "predict"]);

module.exports = async (req, res) => {
  const path = [].concat(req.query.path || []).join("/");
  if (!ALLOWED.has(path)) return res.status(404).json({ detail: "not found" });
  const backend = (process.env.LAYA_BACKEND_URL || "").replace(/\/+$/, "");
  if (!backend) return res.status(500).json({ detail: "LAYA_BACKEND_URL is not set in Vercel" });

  const headers = { "content-type": "application/json" };
  if (process.env.LAYA_PLAYGROUND_KEY) headers.authorization = `Bearer ${process.env.LAYA_PLAYGROUND_KEY}`;
  try {
    const r = await fetch(`${backend}/api/${path}`, {
      method: req.method === "POST" ? "POST" : "GET",
      headers,
      body: req.method === "POST" ? JSON.stringify(req.body ?? {}) : undefined,
    });
    res.status(r.status);
    res.setHeader("content-type", r.headers.get("content-type") || "application/json");
    res.send(await r.text());
  } catch (e) {
    res.status(502).json({ detail: `Laya backend unreachable (${e.message}). Is it running / waking up?` });
  }
};
