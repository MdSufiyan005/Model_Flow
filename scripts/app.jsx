function buildActionPayload(action) {
  const cmd = action.command || "IDLE";
  const payload = { command: cmd };

  if (cmd === "IDLE") return payload;

  if (["LOAD", "EXECUTE", "EVICT", "REPLACE"].includes(cmd)) {
    if (action.model_id) payload.model_id = action.model_id;
    if (action.quant_type) payload.quant_type = action.quant_type;
  }

  if (cmd === "EXECUTE") {
    payload.batch_size = Number(action.batch_size || 1);
  }

  if (cmd === "REPLACE") {
    if (action.evict_model_id) payload.evict_model_id = action.evict_model_id;
    if (action.evict_quant_type) payload.evict_quant_type = action.evict_quant_type;
  }

  return payload;
}

function App() {
  const [task, setTask] = React.useState("multi-load");
  const [state, setState] = React.useState(null);
  const [lastAction, setLastAction] = React.useState(null);
  const [feedback, setFeedback] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [apiErr, setApiErr] = React.useState(null);
  const [done, setDone] = React.useState(false);
  const [autoRunning, setAutoRunning] = React.useState(false);
  const [showStatusPopup, setShowStatusPopup] = React.useState(true);
  const [log, setLog] = React.useState([
    { step: 0, feedback: "Ready — choose task & press START" },
  ]);
  const [currentAction, setCurrentAction] = React.useState({
    command: "LOAD",
    model_id: "gemma-3-4b",
    quant_type: "Q4_K_M",
    batch_size: 4,
    evict_model_id: "gemma-3-4b",
    evict_quant_type: "Q4_K_M",
  });

  const logRef = React.useRef(null);
  const autoInterval = React.useRef(null);

  React.useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [log]);

  React.useEffect(() => {
    // Auto-clear visible feedback/error after a short delay.
    if (!feedback && !error) return;

    const timer = setTimeout(() => {
      setFeedback(null);
      setError(null);
    }, 1800);

    return () => clearTimeout(timer);
  }, [feedback, error]);

  React.useEffect(() => {
    API.state()
      .then(data => {
        const obs = data.observation;
        if (obs && obs.queue && obs.queue.length > 0) {
          setState(obsToState(obs, "CURRENT"));
          setLog([{ step: obs.step_count || 0, feedback: "Synced existing session state" }]);
        }
      })
      .catch(err => console.warn("Initial sync skipped:", err));

    return () => {
      if (autoInterval.current) {
        clearInterval(autoInterval.current);
        autoInterval.current = null;
      }
    };
  }, []);

  async function handleReset() {
    setLoading(true);
    try {
      const data = await API.reset(task);
      const obs = data.observation || data;

      setState(obsToState(obs, task.toUpperCase()));
      setLog([{ step: 0, feedback: `Started ${task.toUpperCase()}` }]);
      setDone(false);
      setFeedback(null);
      setError(null);
      setApiErr(null);
      setLastAction(null);
      setCurrentAction({
        command: "LOAD",
        model_id: "gemma-3-4b",
        quant_type: "Q4_K_M",
        batch_size: 4,
        evict_model_id: "gemma-3-4b",
        evict_quant_type: "Q4_K_M",
      });
    } catch (e) {
      setApiErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleStep(customAction) {
    if (!state || loading) return;
    setLoading(true);

    const action = buildActionPayload(customAction || currentAction);
    setLastAction(action);
    setApiErr(null);

    // Clear old banner immediately when a new action starts.
    setFeedback(null);
    setError(null);

    try {
      const data = await API.step(action);
      const obs = data.observation || data;
      const stepFeedback = obs.last_action_feedback || "";
      const stepError = obs.last_action_error || "";

      setState(obsToState(obs, task.toUpperCase()));
      setFeedback(stepFeedback || null);
      setError(stepError || null);
      setLog(prev => [
        ...prev,
        {
          step: obs.step_count || 0,
          action,
          feedback: stepFeedback,
          error: stepError,
        },
      ]);

      if (data.done || obs.done) setDone(true);
    } catch (e) {
      setApiErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  function toggleAuto() {
    const running = !autoRunning;
    setAutoRunning(running);

    if (running) {
      autoInterval.current = setInterval(() => {
        if (!state || done) return;

        const action =
          state.loaded.length === 0
            ? { command: "LOAD", model_id: "gemma-3-4b", quant_type: "Q4_K_M" }
            : { command: "EXECUTE", model_id: "gemma-3-4b", quant_type: "Q4_K_M", batch_size: 4 };

        handleStep(action);
      }, 1100);
    } else if (autoInterval.current) {
      clearInterval(autoInterval.current);
      autoInterval.current = null;
    }
  }

  const hasEpisode = !!state;
  const currentDesc = TASKS.find(t => t.id === task)?.desc || "";

  return React.createElement(
    "div",
    {
      style: {
        fontFamily: "'VT323', monospace",
        background: C.bg,
        color: C.bodyText,
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        padding: "10px",
        gap: "10px",
        overflow: "hidden",
        fontSize: "15px",
        position: "relative",
      },
    },

    React.createElement(
      "style",
      {},
      `
      .btn { font-size: 13px; padding: 8px 18px; }
      @keyframes spike-pulse { 0%,100% { opacity: 0.5 } 50% { opacity: 1 } }
      a { text-decoration: none; }
      `
    ),

    showStatusPopup && React.createElement(StatusPopup, { onClose: () => setShowStatusPopup(false) }),

    React.createElement(
      "div",
      {
        style: {
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 20px",
          ...topBorder(C.amber),
        },
      },
      React.createElement(
        "div",
        { style: { display: "flex", alignItems: "center", gap: "18px" } },
        React.createElement(
          "span",
          { style: { fontFamily: "'Press Start 2P'", fontSize: "14px", color: C.amber } },
          "◈ MODELFLOW"
        ),
        React.createElement(
          "span",
          { style: { fontSize: "20px", color: C.midText } },
          task.toUpperCase()
        )
      ),
      React.createElement(
        "div",
        { style: { display: "flex", alignItems: "center", gap: "10px" } },
        React.createElement(
          "select",
          {
            value: task,
            onChange: e => setTask(e.target.value),
            style: {
              fontSize: "16px",
              padding: "8px 12px",
              background: C.panel,
              color: C.bodyText,
              border: `1px solid ${C.border}`,
            },
          },
          TASKS.map(t =>
            React.createElement("option", { key: t.id, value: t.id }, t.label)
          )
        ),
        React.createElement(
          "button",
          {
            className: "btn",
            onClick: handleReset,
            disabled: loading,
            style: { background: C.amber, color: C.bg },
          },
          hasEpisode ? "RESTART" : "START"
        ),
        React.createElement(
          "button",
          {
            className: "btn",
            onClick: () => handleStep(),
            disabled: loading || !hasEpisode,
            style: { background: C.green, color: C.bg },
          },
          "STEP"
        ),
        React.createElement(
          "div",
          { style: { position: "relative" } },
          React.createElement(
            "button",
            {
              className: "btn",
              onClick: toggleAuto,
              style: { background: autoRunning ? C.red : C.purple, color: C.bg },
              title: "Internal Rule-based Auto: Not connected to an AI Agent",
            },
            autoRunning ? "STOP AUTO" : "AUTO"
          ),
          autoRunning &&
            React.createElement(
              "div",
              {
                style: {
                  position: "absolute",
                  bottom: "-20px",
                  left: "0",
                  right: "0",
                  fontSize: "10px",
                  color: C.purple,
                  textAlign: "center",
                  whiteSpace: "nowrap",
                },
              },
              "Rule-based (No AI)"
            )
        )
      ),
      React.createElement(
        "div",
        {
          style: {
            fontSize: "14px",
            color: C.midText,
            maxWidth: "340px",
            textAlign: "right",
          },
        },
        currentDesc
      )
    ),

    apiErr &&
      React.createElement(
        "div",
        {
          style: {
            padding: "12px 20px",
            color: C.red,
            border: `1px solid ${C.red}`,
          },
        },
        "⛔ " + apiErr
      ),

    React.createElement(
      "div",
      {
        style: {
          display: "grid",
          gridTemplateColumns: "210px 1fr 200px",
          gridTemplateRows: "1fr auto 120px",
          gap: "10px",
          flex: 1,
          overflow: "hidden",
        },
      },
      React.createElement(
        Pane,
        { col: C.amber, title: "MODEL ROSTER", style: { gridRow: "1 / 3" } },
        React.createElement(Roster, { loaded: state ? state.loaded : [] })
      ),
      React.createElement(
        Pane,
        { col: C.orange, title: "RAM ARENA", style: { gridRow: "1 / 2" } },
        React.createElement(Arena, { state, lastAction, feedback, error })
      ),
      React.createElement(
        Pane,
        { col: C.purple, title: "REQUEST QUEUE", style: { gridRow: "1 / 3" } },
        React.createElement(Queue, {
          queue: state ? state.queue : [],
          completed: state ? state.completed : 0,
          total: state ? state.total : 0,
        })
      ),
      React.createElement(
        "div",
        { style: { gridColumn: "2 / 3", gridRow: "2 / 3" } },
        React.createElement(ActionConsole, {
          onStep: handleStep,
          loading,
          hasEpisode,
          loaded: state ? state.loaded : [],
          currentAction,
          setCurrentAction,
        })
      ),
      React.createElement(
        Pane,
        {
          col: C.dimText,
          title: "ACTION LOG",
          style: { gridColumn: "1 / -1", gridRow: "3 / 4" },
        },
        React.createElement(
          "div",
          {
            ref: logRef,
            style: { overflow: "auto", flex: 1, fontSize: "14px" },
          },
          log.map((entry, i) =>
            React.createElement(LogRow, {
              key: i,
              entry,
              isLatest: i === log.length - 1,
            })
          )
        )
      )
    ),

    done &&
      state &&
      React.createElement(
        "div",
        {
          style: {
            position: "absolute",
            inset: 0,
            background: "rgba(7,5,2,0.85)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 100,
          },
        },
        React.createElement(
          "div",
          {
            style: {
              ...topBorder(C.green),
              background: C.panel,
              padding: "24px",
              minWidth: "320px",
              display: "flex",
              flexDirection: "column",
              gap: "16px",
            },
          },
          React.createElement(
            "div",
            {
              style: {
                fontFamily: "'Press Start 2P'",
                fontSize: "14px",
                color: C.green,
                textAlign: "center",
                marginBottom: "8px",
              },
            },
            "EPISODE COMPLETE"
          ),
          React.createElement(
            "div",
            { style: { fontSize: "32px", color: C.bright, textAlign: "center" } },
            `REWARD: ${(state.cum_reward || 0).toFixed(2)}`
          ),
          React.createElement(
            "div",
            {
              style: {
                borderTop: `1px dashed ${C.border}`,
                borderBottom: `1px dashed ${C.border}`,
                padding: "12px 0",
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "8px",
                fontSize: "16px",
                color: C.bodyText,
              },
            },
            React.createElement(
              "div",
              {},
              React.createElement("span", { style: { color: C.dimText } }, "Completed: "),
              `${state.completed}/${state.total}`
            ),
            React.createElement(
              "div",
              {},
              React.createElement("span", { style: { color: C.dimText } }, "Steps: "),
              state.step_count
            ),
            React.createElement(
              "div",
              {},
              React.createElement("span", { style: { color: C.dimText } }, "Loads: "),
              (state.grader && state.grader.loads) || 0
            ),
            React.createElement(
              "div",
              {},
              React.createElement("span", { style: { color: C.dimText } }, "Evictions: "),
              (state.grader && state.grader.evicts) || 0
            ),
            React.createElement(
              "div",
              {},
              React.createElement("span", { style: { color: C.dimText } }, "OOM Errors: "),
              (state.grader && state.grader.ooms) || 0
            ),
            React.createElement(
              "div",
              {},
              React.createElement("span", { style: { color: C.dimText } }, "Idles: "),
              (state.grader && state.grader.idles) || 0
            )
          ),
          React.createElement(
            "button",
            {
              className: "btn",
              onClick: handleReset,
              style: { background: C.amber, color: C.bg, fontSize: "16px", marginTop: "8px" },
            },
            "PLAY AGAIN"
          )
        )
      )
  );
}

if (typeof ReactDOM !== "undefined") {
  const root = ReactDOM.createRoot(document.getElementById("root"));
  root.render(React.createElement(App));
}