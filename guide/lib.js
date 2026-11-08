// Helpers for guide chapters: reuse a paper-page tab (papers/<Key>/visual.js) as an inline chapter visual.
// The tab runs unchanged; it gets its data.json (if the paper has one) and the ctx fields the paper shell provides.
const cache = {};
async function load(key) {
  if (!cache[key]) cache[key] = (async () => {
    const [mod, meta] = await Promise.all([
      import(`../papers/${key}/visual.js`),
      fetch(`papers/${key}/meta.json`).then(r => r.json()).catch(() => ({})),
    ]);
    const data = meta.data ? await fetch(`papers/${key}/data.json`).then(r => r.json()) : null;
    return { tabs: mod.default.tabs, data };
  })();
  return cache[key];
}

/** A paper tab as a chapter visual. `over` replaces fields (e.g. title) without touching the dynamics. */
export async function paperTab(key, id, over = {}) {
  const { tabs, data } = await load(key);
  const t = tabs.find(x => x.id === id);
  if (!t) throw new Error(`[guide] papers/${key} has no tab "${id}"`);
  return Object.assign({}, t, { data, base: `papers/${key}/`, source: key }, over);
}
