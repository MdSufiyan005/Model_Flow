const PARTS = [
  "/scripts/core.jsx",
  "/scripts/ui.jsx",
  "/scripts/app.jsx",
];

(async function boot() {
  const sources = await Promise.all(
    PARTS.map(async (path) => {
      const r = await fetch(path);
      if (!r.ok) throw new Error(`Failed to load ${path}: ${r.status}`);
      return await r.text();
    })
  );

  eval(sources.join("\n\n"));
})();