// App.jsx
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import studentAvatar from "./assets/graduate.png";
import teacherAvatar from "./assets/teacher.png";
import attachIcon from "./assets/attach-file.png";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function nowTime() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function makeChatTitle(firstUserMsg) {
  const t = (firstUserMsg || "New chat").trim();
  return t.length > 28 ? t.slice(0, 28) + "…" : t;
}

function makeId() {
  return crypto?.randomUUID ? crypto.randomUUID() : String(Date.now());
}

export default function App() {
  const [chats, setChats] = useState(() => {
    const id = makeId();
    return [{ id, title: "New chat", messages: [] }];
  });
  const [activeChatId, setActiveChatId] = useState(chats[0].id);

  const activeChat = useMemo(
    () => chats.find((c) => c.id === activeChatId),
    [chats, activeChatId]
  );

  const [renamingChatId, setRenamingChatId] = useState(null);
  const [renameValue, setRenameValue] = useState("");

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const [lastTrace, setLastTrace] = useState([]);
  const [showTrace, setShowTrace] = useState(false);

  const [error, setError] = useState("");

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  // demo-only file attach
  const fileInputRef = useRef(null);
  const [selectedFiles, setSelectedFiles] = useState([]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages, loading, showTrace]);

  function updateActiveChat(updater) {
    setChats((prev) =>
      prev.map((c) => (c.id !== activeChatId ? c : updater(c)))
    );
  }

  function pushMessage(role, content, extra = {}) {
    updateActiveChat((c) => ({
      ...c,
      messages: [...c.messages, { role, content, ts: nowTime(), ...extra }],
    }));
  }

  function appendToLastAssistant(deltaText) {
    updateActiveChat((c) => {
      const msgs = [...(c.messages || [])];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") {
          msgs[i] = {
            ...msgs[i],
            content: (msgs[i].content || "") + deltaText,
          };
          return { ...c, messages: msgs };
        }
      }
      msgs.push({ role: "assistant", content: deltaText, ts: nowTime() });
      return { ...c, messages: msgs };
    });
  }

  function stopStreamingFlagOnLastAssistant() {
    updateActiveChat((c) => {
      const msgs = [...(c.messages || [])];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") {
          msgs[i] = { ...msgs[i], isStreaming: false };
          break;
        }
      }
      return { ...c, messages: msgs };
    });
  }

  // demo-only attach
  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function onFilesSelected(e) {
    const files = Array.from(e.target.files || []);
    setSelectedFiles(files);
    e.target.value = "";
  }

  function removeSelectedFile(idx) {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== idx));
  }

  function newChat() {
    const id = makeId();
    setChats((prev) => [{ id, title: "New chat", messages: [] }, ...prev]);
    setActiveChatId(id);
    setInput("");
    setShowTrace(false);
    setLastTrace([]);
    setError("");
    setRenamingChatId(null);
    setRenameValue("");
    setSelectedFiles([]);
    setTimeout(() => textareaRef.current?.focus(), 0);
  }

  function deleteChat(id) {
    setChats((prev) => prev.filter((c) => c.id !== id));

    if (id === activeChatId) {
      setTimeout(() => {
        setChats((prev) => {
          if (prev.length === 0) {
            const nid = makeId();
            setActiveChatId(nid);
            return [{ id: nid, title: "New chat", messages: [] }];
          }
          setActiveChatId(prev[0].id);
          return prev;
        });
      }, 0);
    }

    if (id === renamingChatId) {
      setRenamingChatId(null);
      setRenameValue("");
    }
  }

  function startRename(chat) {
    setRenamingChatId(chat.id);
    setRenameValue(chat.title || "");
  }
  function cancelRename() {
    setRenamingChatId(null);
    setRenameValue("");
  }
  function saveRename(chatId) {
    const newTitle = renameValue.trim() || "New chat";
    setChats((prev) =>
      prev.map((c) => (c.id === chatId ? { ...c, title: newTitle } : c))
    );
    cancelRename();
  }

  function buildHistoryPayload(chat, maxTurns = 20) {
    const msgs = (chat?.messages ?? []).slice(-maxTurns);
    return msgs.map((m) => ({ role: m.role, content: m.content }));
  }

  function parseSSE(buffer, onEvent) {
    const frames = buffer.split("\n\n");
    const remainder = frames.pop() || "";

    for (const frame of frames) {
      const lines = frame.split("\n");
      let eventName = "message";
      const dataLines = [];

      for (const line of lines) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }

      const dataStr = dataLines.join("\n");
      if (!dataStr) continue;

      try {
        onEvent(eventName, JSON.parse(dataStr));
      } catch {
        onEvent(eventName, { raw: dataStr });
      }
    }

    return remainder;
  }

  // ✅ UPDATED: allow selecting a model per message
  async function sendText(text, modelOverride = null) {
    const cleaned = (text ?? "").trim();
    if (!cleaned || loading) return;

    setError("");
    setShowTrace(false);
    setLastTrace([]);

    if (activeChat && activeChat.title === "New chat") {
      setChats((prev) =>
        prev.map((c) =>
          c.id === activeChatId ? { ...c, title: makeChatTitle(cleaned) } : c
        )
      );
    }

    const history = buildHistoryPayload(activeChat, 20);

    pushMessage("user", cleaned);
    setInput("");
    setLoading(true);

    // streaming assistant bubble
    pushMessage("assistant", "", { isStreaming: true });

    try {
      const res = await fetch(`${API_BASE}/chat_stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // ✅ pass model override to backend
        body: JSON.stringify({
          message: cleaned,
          history,
          model: modelOverride || undefined,
        }),
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(`Backend error (${res.status}): ${t}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("Streaming not supported in this browser.");

      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let sawAnyDelta = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        buffer = parseSSE(buffer, (eventName, data) => {
          if (eventName === "delta") {
            const d = data?.delta ?? "";
            if (d) {
              sawAnyDelta = true;
              appendToLastAssistant(d);
            }
          } else if (eventName === "error") {
            // backend sends {message, detail?}
            const msg =
              data?.message ||
              data?.detail ||
              "Streaming error";
            setError(msg);
          } else if (eventName === "done") {
            stopStreamingFlagOnLastAssistant();
          }
        });
      }

      if (!sawAnyDelta) {
        appendToLastAssistant("(no answer)");
      }
    } catch (e) {
      setError(String(e?.message ?? e));
      appendToLastAssistant(
        "\n\n(⚠️ I ran into an error calling the streaming backend. Check FastAPI is running on port 8000 and CORS is enabled.)"
      );
    } finally {
      stopStreamingFlagOnLastAssistant();
      setLoading(false);
      setSelectedFiles([]);
      textareaRef.current?.focus();
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || loading) return;
    await sendText(text);
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  const isFreshChat = (activeChat?.messages ?? []).length === 0;

  // ✅ NEW welcome starter actions (each has model + prompt)
  const starterActions = [
    {
      title: "🧠 Start a Guided Reflection",
      desc: "Reflect on today’s class, identify what’s unclear, and build your next learning step.",
      model: "agent:Prism Orchestrator",
      prompt:
        "Start a guided reflection with me about today’s class. Ask me a few focused questions to help me: (1) summarize what I learned, (2) identify what’s unclear, and (3) decide one concrete next step.",
    },
    {
      title: "💛 I’m Feeling Stuck or Overwhelmed",
      desc: "Get quick support and practical guidance to reset your learning mindset.",
      model: "agent:Emotion Support Agent",
      prompt:
        "I’m feeling stuck or overwhelmed. Help me reset my mindset with a short check-in, then suggest 2–3 practical next steps I can do right now.",
    },
    {
      title: "🔎 Find Resources on a Topic",
      desc: "Discover practical guides and explanations to strengthen your understanding.",
      model: "agent:Web Search Agent",
      prompt:
        "Help me find a few strong beginner-friendly resources on this topic. Ask me what topic and my current level, then recommend 3–5 links with 1–2 lines on why each is useful.",
    },
    {
      title: "📄 Ask About My Course Materials",
      desc: "Upload slides, notes, or PDFs and ask questions grounded in your own materials.",
      model: "agent:Course Materials Agent",
      prompt:
        "I want to ask questions about my course materials. First, ask me what I’m uploading (slides/notes/PDF) and what I want to understand, then guide me to paste key excerpts if needed.",
    },
  ];

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebarTop">
          <div className="brand">
            <div className="logoDot" />
            <div>
              <div className="brandTitle">Edu Multi-Agent</div>
              <div className="brandSub">Dev UI</div>
            </div>
          </div>

          <button className="btn primary full" onClick={newChat}>
            New chat
          </button>
        </div>

        <div className="chatList">
          {chats.map((c) => {
            const isActive = c.id === activeChatId;
            const isRenaming = c.id === renamingChatId;

            return (
              <div
                key={c.id}
                className={`chatItem ${isActive ? "active" : ""}`}
                onClick={() => {
                  if (!isRenaming) setActiveChatId(c.id);
                }}
                role="button"
                tabIndex={0}
              >
                <div className="chatItemMain">
                  {isRenaming ? (
                    <input
                      className="chatRenameInput"
                      value={renameValue}
                      autoFocus
                      onChange={(e) => setRenameValue(e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") saveRename(c.id);
                        if (e.key === "Escape") cancelRename();
                      }}
                      onBlur={() => saveRename(c.id)}
                    />
                  ) : (
                    <div
                      className="chatItemTitle"
                      title="Double-click to rename"
                      onDoubleClick={(e) => {
                        e.stopPropagation();
                        startRename(c);
                      }}
                    >
                      {c.title}
                    </div>
                  )}
                </div>

                <div className="chatItemBtns">
                  {!isRenaming && (
                    <button
                      className="iconBtn"
                      title="Rename"
                      onClick={(e) => {
                        e.stopPropagation();
                        startRename(c);
                      }}
                    >
                      ✎
                    </button>
                  )}

                  <button
                    className="iconBtn"
                    title="Delete"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteChat(c.id);
                    }}
                  >
                    ✕
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </aside>

      <section className="main">
        <header className="topbar">
          <div className="topbarLeft">
            <div className="activeTitle">{activeChat?.title ?? "Chat"}</div>
          </div>
        </header>

        <div className="chatArea">
          <div className="chatScroll">
            {isFreshChat && (
              <div className="welcomeWrap">
                <div className="welcomeCard">
                  <img
                    className="welcomeAvatar"
                    src={teacherAvatar}
                    alt="Teacher avatar"
                  />

                  {/* ✅ NEW: Welcome copy */}
                  <div className="welcomeTitle">Time2Reflect</div>
                  <div className="welcomeDesc">
                    Your Intelligent Reflection Partner for STEM Learning.
                    Reflect on what you learned, clarify confusion, explore deeper research,
                    and organize your thinking — all in one place.
                  </div>

                  {/* ✅ NEW: 4 starter actions */}
                  <div className="starterGrid">
                    {starterActions.map((a) => (
                      <button
                        key={a.title}
                        className="starterBtn"
                        onClick={() => sendText(a.prompt, a.model)}
                        disabled={loading}
                      >
                        <div className="starterTitle">{a.title}</div>
                        <div className="starterDesc">{a.desc}</div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {(activeChat?.messages ?? []).map((m, idx) => (
              <div
                key={idx}
                className={`msgRow ${m.role === "user" ? "user" : "assistant"}`}
              >
                <div className="avatar">
                  <img
                    className="avatarImg"
                    src={m.role === "user" ? studentAvatar : teacherAvatar}
                    alt={m.role === "user" ? "Student" : "Tutor"}
                  />
                </div>

                <div className="bubbleWrap">
                  <div className="bubble">
                    <div className="content">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {m.content}
                      </ReactMarkdown>

                      {m.role === "assistant" && m.isStreaming && (
                        <div className="inlineTyping" aria-label="Generating…">
                          <span className="dot" />
                          <span className="dot" />
                          <span className="dot" />
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="meta">{m.ts}</div>
                </div>
              </div>
            ))}

            <div ref={bottomRef} />
          </div>

          <div className="composer">
            {error && <div className="error">Error: {error}</div>}

            {selectedFiles.length > 0 && (
              <div className="fileChips">
                {selectedFiles.map((f, idx) => (
                  <div className="fileChip" key={`${f.name}-${idx}`}>
                    <span className="fileName" title={f.name}>
                      {f.name}
                    </span>
                    <button
                      className="fileRemove"
                      onClick={() => removeSelectedFile(idx)}
                      title="Remove"
                      type="button"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="composerInner">
              <input
                ref={fileInputRef}
                type="file"
                className="fileInputHidden"
                onChange={onFilesSelected}
                multiple
              />

              <textarea
                ref={textareaRef}
                className="input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Escape" && renamingChatId) cancelRename();
                  onKeyDown(e);
                }}
                placeholder="Message Time2Reflect…"
                rows={1}
              />

              <button
                className="iconBtn attachBtn"
                title="Attach file (demo)"
                onClick={openFilePicker}
                disabled={loading}
                type="button"
              >
                <img className="attachIcon" src={attachIcon} alt="Attach file" />
              </button>

              <button className="btn primary" onClick={send} disabled={loading}>
                Send
              </button>
            </div>

            <div className="hint">Enter to send, Shift+Enter for a new line</div>
          </div>
        </div>
      </section>
    </div>
  );
}
