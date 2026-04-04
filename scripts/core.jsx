const C = {
  bg: "#070502", panel: "#0e0a04", border: "#2a1a06",
  dimText: "#5a3e1a", midText: "#9a6e38", bodyText: "#d4a96a",
  bright: "#f0d090", amber: "#d4821a", gold: "#c8a040",
  orange: "#c86020", green: "#4a9e5c", red: "#b84040",
  blue: "#4a7ab8", purple: "#7a5aaa",
  tierLow: "#4a9e5c", tierMed: "#c8a040", tierHigh: "#c86020", tierRisky: "#b84040",
};

const TIER_C = { low: C.tierLow, medium: C.tierMed, high: C.tierHigh, risky: C.tierRisky };
const CMD_C = { LOAD: C.gold, EXECUTE: C.green, EVICT: C.red, IDLE: C.dimText, REPLACE: C.purple };
const CMD_ICON = { LOAD: "↑", EXECUTE: "▶", EVICT: "↓", IDLE: "—", REPLACE: "↔" };
const ROLE_C = { chatbot: C.gold, translator: C.green, coder: C.blue };
const ROLE_LBL = { chatbot: "CHAT", translator: "TRNSL", coder: "CODE" };

const TASKS = [
  { id: "single-load", label: "SINGLE-LOAD", desc: "Tests how well you keep needed models loaded vs evicting them" },
  { id: "multi-load", label: "MULTI-LOAD", desc: "Pack & swap models in RAM like a Tetris puzzle" },
  { id: "quality-limit", label: "QUALITY-LIMIT", desc: "Meet QoS targets for reasoning vs standard requests" },
  { id: "ram-pressure", label: "RAM-PRESSURE", desc: "Handle sudden memory spikes without crashing" },
];

const MODELS = [
  { id: "gemma-3-4b", role: "chatbot" },
  { id: "llama_1b", role: "translator" },
  { id: "qwen3.5-2b", role: "coder" },
];

const QUANTS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"];
const COMMANDS = ["LOAD", "EXECUTE", "EVICT", "IDLE", "REPLACE"];

const ROSTER_DATA = [
  { id: "gemma-3-4b", role: "chatbot", quants: [{ q: "Q4_K_M", tier: "low", mb: 2584 }, { q: "Q5_K_M", tier: "medium", mb: 3100 }, { q: "Q6_K", tier: "high", mb: 3800 }, { q: "Q8_0", tier: "risky", mb: 4820 }] },
  { id: "llama_1b", role: "translator", quants: [{ q: "Q4_K_M", tier: "low", mb: 836 }, { q: "Q5_K_M", tier: "medium", mb: 935 }, { q: "Q6_K", tier: "high", mb: 1040 }] },
  { id: "qwen3.5-2b", role: "coder", quants: [{ q: "Q4_K_M", tier: "low", mb: 1340 }, { q: "Q5_K_M", tier: "medium", mb: 1520 }, { q: "Q6_K", tier: "high", mb: 1650 }] },
];

const API = {
  async reset(task_name) {
    const r = await fetch("/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_name }),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
  async step(action) {
    const r = await fetch("/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
  async state() {
    const r = await fetch("/state");
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
};

function obsToState(obs, taskLabel) {
  const loaded = Object.entries(obs.loaded_models || {}).map(([key, v]) => ({
    key,
    model: v.model,
    tier: v.tier,
    size_mb: v.size_mb,
    gen_tps: v.gen_tps || 0,
    prompt_tps: v.prompt_tps || 0,
  }));

  const queue = (obs.queue || []).map(r => ({
    id: r.request_id,
    type: r.model_type,
    complexity: r.complexity,
    age: r.age_steps,
  }));

  const info = obs.info || {};

  return {
    task: taskLabel,
    ram_used: obs.ram_used_mb || 0,
    ram_limit: obs.ram_limit_mb || 8000,
    loaded,
    spike_mb: obs.pressure_spike_mb || 0,
    spike_steps: obs.spike_steps_remaining || 0,
    cum_reward: info.cumulative_reward || 0,
    completed: info.completed || 0,
    pending: info.pending || queue.length,
    total: (info.completed || 0) + (info.pending || queue.length),
    step_count: obs.step_count || 0,
    queue,
    grader: info.grader_metrics || {},
  };
}

const topBorder = col => ({
  background: C.panel,
  border: `1px solid ${C.border}`,
  borderTop: `2px solid ${col}`,
});

function SectionLabel({ text, col }) {
  return React.createElement(
    "div",
    {
      style: {
        fontFamily: "'Press Start 2P', monospace",
        fontSize: "8px",
        color: col || C.dimText,
        letterSpacing: "1.5px",
        marginBottom: "8px",
      },
    },
    text
  );
}

function Pane({ col, title, style, children }) {
  return React.createElement(
    "div",
    {
      style: {
        ...topBorder(col),
        padding: "12px",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        minHeight: 0,
        ...(style || {}),
      },
    },
    SectionLabel({ text: title, col }),
    children
  );
}