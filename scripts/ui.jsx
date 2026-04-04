function Roster({ loaded }) {
  const active = new Set((loaded || []).map(m => m.key));

  return React.createElement(
    "div",
    { style: { overflow: "auto", flex: 1, fontSize: "14px" } },
    ROSTER_DATA.map(m => {
      const rc = ROLE_C[m.role];

      return React.createElement(
        "div",
        { key: m.id, style: { marginBottom: "14px" } },
        React.createElement(
          "div",
          {
            style: {
              display: "flex",
              alignItems: "center",
              gap: "8px",
              borderBottom: `1px solid ${C.border}`,
              paddingBottom: "4px",
            },
          },
          React.createElement("span", { style: { color: rc } }, "◈"),
          React.createElement("span", { style: { color: rc, fontSize: "15px" } }, m.id),
          React.createElement(
            "span",
            {
              style: {
                marginLeft: "auto",
                fontFamily: "'Press Start 2P'",
                fontSize: "6px",
                color: C.dimText,
                padding: "2px 6px",
                border: `1px solid ${C.border}`,
              },
            },
            ROLE_LBL[m.role]
          )
        ),
        m.quants.map(q => {
          const key = `${m.id}-${q.q}`;
          const on = active.has(key);
          const tc = TIER_C[q.tier];

          return React.createElement(
            "div",
            {
              key: q.q,
              style: {
                display: "flex",
                alignItems: "center",
                gap: "8px",
                padding: "3px 8px",
                marginBottom: "3px",
                borderLeft: `3px solid ${on ? tc : C.border}`,
                background: on ? `${tc}15` : C.bg,
              },
            },
            React.createElement("span", { style: { color: on ? tc : C.dimText, minWidth: "62px" } }, q.q),
            React.createElement(
              "div",
              { style: { flex: 1, height: "3px", background: C.border } },
              React.createElement("div", {
                style: { height: "100%", width: `${(q.mb / 5000) * 100}%`, background: on ? tc : `${tc}44` },
              })
            ),
            React.createElement("span", { style: { color: on ? C.bodyText : C.dimText, fontSize: "13px" } }, `${q.mb}MB`)
          );
        })
      );
    })
  );
}

function ModelCard({ m, active }) {
  const tc = TIER_C[m.tier] || C.dimText;

  return React.createElement(
    "div",
    {
      style: {
        width: "172px",
        background: C.bg,
        border: `1px solid ${active ? tc : C.border}`,
        borderTop: `3px solid ${tc}`,
        padding: "12px",
        position: "relative",
      },
    },
    React.createElement(
      "div",
      {
        style: {
          position: "absolute",
          top: 6,
          right: 8,
          fontFamily: "'Press Start 2P'",
          fontSize: "6px",
          color: tc,
        },
      },
      (m.tier || "unknown").toUpperCase()
    ),
    React.createElement("div", { style: { fontFamily: "'Press Start 2P'", fontSize: "7px", color: C.midText } }, m.model),
    React.createElement("div", { style: { fontSize: "15px", color: tc, margin: "6px 0" } }, m.key.split("-").slice(-2).join(" · ")),
    React.createElement(
      "div",
      { style: { fontSize: "14px", lineHeight: "1.6" } },
      [
        ["MEM", `${m.size_mb} MB`, C.orange],
        ["GEN", `${Number(m.gen_tps || 0).toFixed(1)} t/s`, C.green],
        ["PMP", `${Number(m.prompt_tps || 0).toFixed(1)} t/s`, C.blue],
      ].map(([k, v, vc]) =>
        React.createElement(
          "div",
          { key: k, style: { display: "flex", justifyContent: "space-between" } },
          React.createElement("span", { style: { color: C.dimText } }, k),
          React.createElement("span", { style: { color: vc } }, v)
        )
      )
    )
  );
}

function EmptySlot() {
  return React.createElement(
    "div",
    {
      style: {
        width: "172px",
        height: "118px",
        border: `1px dashed ${C.border}`,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: C.border,
        fontSize: "20px",
      },
    },
    "empty"
  );
}

function Arena({ state, lastAction, feedback, error }) {
  if (!state) {
    return React.createElement(
      "div",
      {
        style: {
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: C.dimText,
          fontSize: "16px",
        },
      },
      "Press START to begin an episode"
    );
  }

  const loaded = state.loaded || [];
  const ramLimit = Math.max(Number(state.ram_limit || 0), 1);
  const vPct = (Number(state.ram_used || 0) / ramLimit) * 100;
  const sPct = (Number(state.spike_mb || 0) / ramLimit) * 100;
  const free = ramLimit - Number(state.ram_used || 0) - Number(state.spike_mb || 0);
  const vCol = vPct > 85 ? C.red : vPct > 65 ? C.orange : C.amber;
  const isOOM = !!(error && String(error).startsWith("OOM"));
  const aCol = lastAction ? (isOOM ? C.red : CMD_C[lastAction.command] || C.dimText) : C.dimText;

  const bannerText = error
    ? `ERROR: ${error}`
    : feedback
      ? `FEEDBACK: ${feedback}`
      : lastAction
        ? `LAST: ${lastAction.command}`
        : "No action yet";

  return React.createElement(
    "div",
    { style: { flex: 1, display: "flex", flexDirection: "column", gap: "12px" } },

    React.createElement(
      "div",
      { style: { flexShrink: 0 } },
      React.createElement(
        "div",
        {
          style: {
            display: "flex",
            justifyContent: "space-between",
            fontSize: "15px",
            marginBottom: "6px",
            color: C.midText,
          },
        },
        React.createElement("span", {}, "RAM ", React.createElement("span", { style: { color: vCol } }, `${state.ram_used}/${state.ram_limit} MB`)),
        React.createElement("span", { style: { color: free < 500 ? C.red : C.dimText } }, `${free} MB free`)
      ),
      React.createElement(
        "div",
        {
          style: {
            height: "22px",
            background: C.bg,
            border: `1px solid ${C.border}`,
            position: "relative",
            overflow: "hidden",
          },
        },
        [25, 50, 75].map(p =>
          React.createElement("div", {
            key: p,
            style: { position: "absolute", left: `${p}%`, top: 0, bottom: 0, width: "2px", background: C.border },
          })
        ),
        React.createElement("div", {
          style: {
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: `${Math.max(0, Math.min(vPct, 100))}%`,
            background: `repeating-linear-gradient(90deg,${vCol} 0,${vCol} 10px,${vCol}77 10px,${vCol}77 12px)`,
          },
        }),
        sPct > 0 &&
          React.createElement("div", {
            style: {
              position: "absolute",
              left: `${Math.max(0, Math.min(vPct, 100))}%`,
              top: 0,
              bottom: 0,
              width: `${Math.max(0, Math.min(sPct, 100 - vPct))}%`,
              background: `repeating-linear-gradient(90deg,${C.red}55 0,${C.red}55 6px,${C.red}18 6px,${C.red}18 8px)`,
              animation: "spike-pulse .9s infinite",
            },
          })
      )
    ),

    React.createElement(
      "div",
      {
        style: {
          padding: "10px 14px",
          border: `1px solid ${aCol}55`,
          background: `${aCol}12`,
          color: aCol,
          whiteSpace: "pre-wrap",
        },
      },
      bannerText
    ),

    React.createElement(
      "div",
      { style: { flex: 1, overflow: "auto" } },
      SectionLabel({ text: `LOADED (${loaded.length}/3)` }),
      React.createElement(
        "div",
        { style: { display: "flex", gap: "10px", flexWrap: "wrap" } },
        loaded.map(m =>
          React.createElement(ModelCard, {
            key: m.key,
            m,
            active: lastAction && lastAction.command === "EXECUTE" && lastAction.model_id === m.model,
          })
        ),
        Array.from({ length: Math.max(0, 3 - loaded.length) }, (_, i) =>
          React.createElement(EmptySlot, { key: i })
        )
      )
    ),

    React.createElement(
      "div",
      { style: { flexShrink: 0 } },
      React.createElement(
        "div",
        { style: { display: "flex", justifyContent: "space-between", fontSize: "15px", color: C.midText } },
        "Episode progress"
      ),
      React.createElement(
        "div",
        {
          style: {
            height: "10px",
            background: C.bg,
            border: `1px solid ${C.border}`,
            overflow: "hidden",
          },
        },
        React.createElement("div", {
          style: {
            height: "100%",
            width: `${(Number(state.completed || 0) / Math.max(Number(state.total || 1), 1)) * 100}%`,
            background: `repeating-linear-gradient(90deg,${C.green} 0,${C.green} 10px,${C.green}66 10px,${C.green}66 12px)`,
          },
        })
      )
    )
  );
}

function Queue({ queue, completed, total }) {
  const items = queue || [];

  if (!items.length) {
    return React.createElement(
      "div",
      {
        style: {
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: C.dimText,
          fontSize: "16px",
        },
      },
      "No pending requests"
    );
  }

  return React.createElement(
    "div",
    { style: { flex: 1, overflow: "auto", fontSize: "14px" } },
    SectionLabel({ text: `${items.length} PENDING / ${total || items.length} TOTAL` }),
    items.map(r =>
      React.createElement(
        "div",
        {
          key: r.id,
          style: {
            padding: "4px 8px",
            marginBottom: "3px",
            borderLeft: `3px solid ${ROLE_C[r.type] || C.border}`,
          },
        },
        `${r.id} • ${r.type} • ${r.complexity} • age ${r.age}`
      )
    )
  );
}

function LogRow({ entry, isLatest }) {
  return React.createElement(
    "div",
    {
      style: {
        display: "flex",
        gap: "10px",
        padding: "3px 10px",
        background: isLatest ? `${C.border}40` : "transparent",
        fontSize: "14px",
      },
    },
    React.createElement("span", {}, entry.step || "-"),
    entry.action ? React.createElement("span", {}, `${CMD_ICON[entry.action.command]} ${entry.action.command}`) : null,
    React.createElement("span", { style: { flex: 1, color: C.midText } }, entry.feedback || entry.error || "-")
  );
}

function StatusPopup({ onClose }) {
  const [checks, setChecks] = React.useState({
    state: "checking",
    schema: "checking",
    reset: "ready",
    step: "ready",
    dashboard: "ready",
  });

  React.useEffect(() => {
    let alive = true;

    async function runChecks() {
      const next = {
        state: "checking",
        schema: "checking",
        reset: "ready",
        step: "ready",
        dashboard: "ready",
      };

      try {
        const r = await fetch("/state");
        next.state = r.ok ? "online" : "offline";
      } catch {
        next.state = "offline";
      }

      try {
        const r = await fetch("/schema");
        next.schema = r.ok ? "online" : "offline";
      } catch {
        next.schema = "offline";
      }

      if (alive) setChecks(next);
    }

    runChecks();
    return () => {
      alive = false;
    };
  }, []);

  const badge = (value) => {
    if (value === "online") return { text: "Online", color: C.green };
    if (value === "offline") return { text: "Offline", color: C.red };
    if (value === "ready") return { text: "Available", color: C.blue };
    return { text: "Checking…", color: C.midText };
  };

  const row = (label, value) =>
    React.createElement(
      "div",
      {
        style: {
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "10px 12px",
          border: `1px solid ${C.border}`,
          background: C.bg,
          marginBottom: "8px",
        },
      },
      React.createElement("span", { style: { color: C.bodyText } }, label),
      React.createElement(
        "span",
        {
          style: {
            color: badge(value).color,
            fontFamily: "'Press Start 2P', monospace",
            fontSize: "8px",
          },
        },
        badge(value).text
      )
    );

  return React.createElement(
    "div",
    {
      style: {
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.72)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 9999,
        backdropFilter: "blur(4px)",
      },
    },
    React.createElement(
      "div",
      {
        style: {
          width: "min(92vw, 720px)",
          background: C.panel,
          border: `1px solid ${C.border}`,
          borderTop: `3px solid ${C.amber}`,
          borderRadius: "18px",
          padding: "22px",
          boxShadow: "0 30px 100px rgba(0,0,0,0.45)",
          color: C.bodyText,
          position: "relative",
          overflow: "hidden",
        },
      },
      React.createElement(
        "div",
        {
          style: {
            display: "flex",
            justifyContent: "space-between",
            alignItems: "start",
            gap: "12px",
          },
        },
        React.createElement(
          "div",
          null,
          React.createElement(
            "div",
            {
              style: {
                fontFamily: "'Press Start 2P', monospace",
                fontSize: "10px",
                color: C.amber,
                marginBottom: "10px",
              },
            },
            "MODELFLOW STATUS"
          ),
          React.createElement(
            "h2",
            { style: { margin: 0, fontSize: "28px", color: C.bright } },
            "All endpoints are live"
          ),
          React.createElement(
            "p",
            {
              style: {
                marginTop: "10px",
                color: C.midText,
                lineHeight: 1.6,
                maxWidth: "560px",
              },
            },
            "The dashboard is loaded and the API routes are reachable."
          )
        ),
        React.createElement(
          "button",
          {
            onClick: onClose,
            style: {
              background: "transparent",
              border: "none",
              color: C.midText,
              fontSize: "30px",
              cursor: "pointer",
              lineHeight: 1,
            },
            title: "Close",
          },
          "×"
        )
      ),

      React.createElement(
        "div",
        { style: { marginTop: "18px" } },
        row("GET /state", checks.state),
        row("GET /schema", checks.schema),
        row("POST /reset", checks.reset),
        row("POST /step", checks.step),
        row("GET /dashboard", checks.dashboard)
      ),

      React.createElement(
        "div",
        {
          style: {
            display: "flex",
            flexWrap: "wrap",
            gap: "10px",
            marginTop: "18px",
          },
        },
        React.createElement(
          "a",
          {
            href: "/docs",
            target: "_blank",
            style: {
              padding: "10px 14px",
              borderRadius: "12px",
              textDecoration: "none",
              background: `${C.gold}18`,
              color: C.bright,
              border: `1px solid ${C.gold}44`,
            },
          },
          "Open API Docs"
        ),
        React.createElement(
          "a",
          {
            href: "/dashboard",
            target: "_blank",
            style: {
              padding: "10px 14px",
              borderRadius: "12px",
              textDecoration: "none",
              background: `${C.blue}18`,
              color: C.bright,
              border: `1px solid ${C.blue}44`,
            },
          },
          "Open Dashboard"
        ),
        React.createElement(
          "a",
          {
            href: "/schema",
            target: "_blank",
            style: {
              padding: "10px 14px",
              borderRadius: "12px",
              textDecoration: "none",
              background: `${C.green}18`,
              color: C.bright,
              border: `1px solid ${C.green}44`,
            },
          },
          "View Schema"
        )
      )
    )
  );
}

function ActionConsole({ onStep, loading, hasEpisode, loaded = [], currentAction, setCurrentAction }) {
  const update = (fields) => setCurrentAction(prev => ({ ...prev, ...fields }));

  const cmd = currentAction.command;
  const showModel = ["LOAD", "EXECUTE", "EVICT", "REPLACE"].includes(cmd);
  const showQuant = ["LOAD", "EXECUTE", "EVICT", "REPLACE"].includes(cmd);
  const showBatch = cmd === "EXECUTE";
  const showEvict = cmd === "REPLACE";

  const selectStyle = {
    fontSize: "14px",
    padding: "6px 10px",
    background: C.bg,
    color: C.bodyText,
    border: `1px solid ${C.border}`,
    flex: 1,
  };
  const labelStyle = { color: C.dimText, fontSize: "12px", marginBottom: "4px", display: "block" };
  const fieldWrap = { display: "flex", flexDirection: "column", flex: 1 };

  return React.createElement(
    "div",
    { style: { ...topBorder(C.gold), padding: "12px", background: C.panel } },
    SectionLabel({ text: "ACTION CONSOLE", col: C.gold }),
    React.createElement(
      "div",
      { style: { display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "flex-end" } },

      React.createElement(
        "div",
        { style: fieldWrap },
        React.createElement("label", { style: labelStyle }, "COMMAND"),
        React.createElement(
          "select",
          {
            style: { ...selectStyle, borderColor: C.gold, color: CMD_C[cmd] || C.bodyText },
            value: cmd,
            onChange: e => update({ command: e.target.value }),
          },
          COMMANDS.map(c => React.createElement("option", { key: c, value: c }, c))
        )
      ),

      showModel &&
        React.createElement(
          "div",
          { style: fieldWrap },
          React.createElement("label", { style: labelStyle }, "MODEL"),
          React.createElement(
            "select",
            {
              style: selectStyle,
              value: currentAction.model_id || "",
              onChange: e => update({ model_id: e.target.value }),
            },
            React.createElement("option", { value: "" }, "Select Model"),
            MODELS.map(m => React.createElement("option", { key: m.id, value: m.id }, m.id))
          )
        ),

      showQuant &&
        React.createElement(
          "div",
          { style: fieldWrap },
          React.createElement("label", { style: labelStyle }, "QUANT"),
          React.createElement(
            "select",
            {
              style: selectStyle,
              value: currentAction.quant_type || "",
              onChange: e => update({ quant_type: e.target.value }),
            },
            React.createElement("option", { value: "" }, "Select Quant"),
            QUANTS.map(q => React.createElement("option", { key: q, value: q }, q))
          )
        ),

      showBatch &&
        React.createElement(
          "div",
          { style: fieldWrap },
          React.createElement("label", { style: labelStyle }, "BATCH"),
          React.createElement("input", {
            type: "number",
            min: 1,
            max: 8,
            style: selectStyle,
            value: currentAction.batch_size,
            onChange: e => update({ batch_size: parseInt(e.target.value, 10) }),
          })
        ),

      showEvict &&
        React.createElement(
          "div",
          { style: { ...fieldWrap, borderLeft: `1px solid ${C.border}`, paddingLeft: "10px" } },
          React.createElement("label", { style: labelStyle }, "EVICT MODEL"),
          React.createElement(
            "select",
            {
              style: selectStyle,
              value: currentAction.evict_model_id || "",
              onChange: e => update({ evict_model_id: e.target.value }),
            },
            React.createElement("option", { value: "" }, "Select Evict Model"),
            loaded.map(m => React.createElement("option", { key: m.key, value: m.model }, m.model))
          ),
          React.createElement("label", { style: { ...labelStyle, marginTop: "8px" } }, "EVICT QUANT"),
          React.createElement(
            "select",
            {
              style: selectStyle,
              value: currentAction.evict_quant_type || "",
              onChange: e => update({ evict_quant_type: e.target.value }),
            },
            React.createElement("option", { value: "" }, "Select Evict Quant"),
            QUANTS.map(q => React.createElement("option", { key: q, value: q }, q))
          )
        ),

      React.createElement(
        "button",
        {
          className: "btn",
          disabled: loading || !hasEpisode,
          onClick: () => onStep(currentAction),
          style: { background: C.gold, color: C.bg, height: "34px", marginLeft: "10px" },
        },
        "COMMIT ACTION"
      )
    )
  );
}