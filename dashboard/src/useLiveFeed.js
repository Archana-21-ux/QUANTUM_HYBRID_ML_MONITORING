import { useEffect, useReducer, useRef } from "react";

const initialState = {
  connected: false,
  health: [],
  predictions: [],
  reasoning: [],
  featureDrift: [], // latest window's per-feature drift rows
  deployed: null,
  simulation: { state: "idle" },
  candidateEvents: 0, // bumped when a candidate event arrives -> panels refetch
};

function reducer(state, event) {
  switch (event.type) {
    case "connected":
      return { ...state, connected: event.value };
    case "reset":
      return { ...initialState, connected: state.connected };
    case "health":
      return { ...state, health: [...state.health, event].slice(-300) };
    case "predictions": {
      const rows = event.rows.map((row, i) => ({
        ...row,
        window_id: event.window_id,
        key: `${event.window_id}-${i}`,
      }));
      return { ...state, predictions: [...rows, ...state.predictions].slice(0, 40) };
    }
    case "reasoning":
      return { ...state, reasoning: [event, ...state.reasoning].slice(0, 100) };
    case "feature_drift":
      return { ...state, featureDrift: event.rows };
    case "deployed":
      return { ...state, deployed: event.version, candidateEvents: state.candidateEvents + 1 };
    case "candidate":
      return { ...state, candidateEvents: state.candidateEvents + 1 };
    case "simulation": {
      const next = { ...state, simulation: event.status };
      if (event.deployed) next.deployed = event.deployed;
      return next;
    }
    default:
      return state;
  }
}

export function useLiveFeed() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const socketRef = useRef(null);

  useEffect(() => {
    let closed = false;

    function connect() {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const socket = new WebSocket(`${protocol}//${location.host}/ws`);
      socketRef.current = socket;

      socket.onopen = () => {
        dispatch({ type: "connected", value: true });
        dispatch({ type: "reset" }); // server replays the current run on connect
      };
      socket.onmessage = (message) => dispatch(JSON.parse(message.data));
      socket.onclose = () => {
        dispatch({ type: "connected", value: false });
        if (!closed) setTimeout(connect, 1500);
      };
      socket.onerror = () => socket.close();
    }

    connect();
    const keepalive = setInterval(() => {
      if (socketRef.current?.readyState === WebSocket.OPEN) socketRef.current.send("ping");
    }, 20000);

    return () => {
      closed = true;
      clearInterval(keepalive);
      socketRef.current?.close();
    };
  }, []);

  return state;
}
