import { Link, useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";

export default function Conversation() {
  const navigate = useNavigate();
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const [locked, setLocked] = useState(false);
  const greetedRef = useRef(false);
  const doneRef = useRef(false);
  const listRef = useRef(null);
  const recogRef = useRef(null);
  const transcriptRef = useRef("");
  const lastSpokenRef = useRef("");

  useEffect(() => {
    const userId = localStorage.getItem("user_id");
    const threadId = localStorage.getItem("thread_id");
    const checkDaily = async () => {
      try {
        if (!userId) {
          navigate("/login");
          return;
        }
        const res = await fetch(
          `http://localhost:8000/daily-status?user_id=${encodeURIComponent(userId)}`
        );
        if (!res.ok) {
          return;
        }
        const data = await res.json();
        if (data.completed_today && data.has_rubric) {
          if (doneRef.current) {
            return;
          }
          doneRef.current = true;
          const msg = "Hôm nay bạn đã học xong rồi. Hẹn bạn vào ngày mai nhé!";
          setLocked(true);
          setMessages((prev) => [...prev, { from: "coach", text: msg }]);
          speakText(msg);
          return;
        }
        if (!greetedRef.current) {
          greetedRef.current = true;
          const greeting =
            "Xin chào bạn, đây là chatbot hỗ trợ học tiếng Anh. Bạn vui lòng nói tiếng Anh trong suốt quá trình nhé.";
          setMessages((prev) => [...prev, { from: "coach", text: greeting }]);
          speakText(greeting);
        }
        if (!threadId) {
          navigate("/menu");
        }
      } catch {
        if (!greetedRef.current) {
          greetedRef.current = true;
          const greeting =
            "Xin chào bạn, đây là chatbot hỗ trợ học tiếng Anh. Bạn vui lòng nói tiếng Anh trong suốt quá trình nhé.";
          setMessages((prev) => [...prev, { from: "coach", text: greeting }]);
          speakText(greeting);
        }
      }
    };
    checkDaily();
  }, [navigate]);

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      return;
    }
    const recog = new SpeechRecognition();
    recog.lang = "en-US";
    recog.interimResults = true;
    recog.continuous = false;

    recog.onresult = (event) => {
      let full = "";
      for (let i = 0; i < event.results.length; i += 1) {
        full += event.results[i][0].transcript;
      }
      transcriptRef.current = full;
      setInput(full);
    };

    recog.onend = () => {
      setListening(false);
    };

    recog.onerror = () => {
      setListening(false);
    };

    recogRef.current = recog;
    return () => {
      try {
        recog.stop();
      } catch {
        // ignore
      }
    };
  }, []);

  const sendMessage = async (overrideText) => {
    const userId = localStorage.getItem("user_id");
    const threadId = localStorage.getItem("thread_id");
    const rawText = typeof overrideText === "string" ? overrideText : input;
    const text = rawText.trim();
    if (!text || !userId || !threadId || locked) {
      return;
    }
    setMessages((prev) => [...prev, { from: "user", text }]);
    if (typeof overrideText !== "string") {
      setInput("");
    }
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          thread_id: threadId,
          message: text,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setMessages((prev) => [
          ...prev,
          { from: "coach", text: "Lỗi gửi tin. Vui lòng thử lại." },
        ]);
        return;
      }
      const assistantText = data.assistant_message || "";
      setMessages((prev) => [...prev, { from: "coach", text: assistantText }]);
      speakText(assistantText);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { from: "coach", text: "Không kết nối được server." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!listRef.current) {
      return;
    }
    listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages, loading]);

  const speakText = (text) => {
    const t = (text || "").trim();
    if (!t) {
      return;
    }
    if (!window.speechSynthesis) {
      return;
    }
    if (lastSpokenRef.current === t) {
      return;
    }
    lastSpokenRef.current = t;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(t);
    utter.lang = "en-US";
    utter.rate = 1;
    window.speechSynthesis.speak(utter);
  };

  const toggleMic = () => {
    const recog = recogRef.current;
    if (locked) {
      return;
    }
    if (!recog) {
      setMessages((prev) => [
        ...prev,
        { from: "coach", text: "Trình duyệt chưa hỗ trợ Speech API." },
      ]);
      return;
    }
    if (listening) {
      recog.stop();
      return;
    }
    transcriptRef.current = "";
    setListening(true);
    try {
      recog.start();
    } catch {
      setListening(false);
    }
  };

  return (
    <div className="min-h-screen bg-ink text-fog">
      <div className="mx-auto grid min-h-screen max-w-6xl gap-6 px-6 py-10 lg:grid-cols-[0.35fr_0.65fr]">
        <aside className="glass rounded-3xl p-6 shadow-soft">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Phiên học</h2>
            <Link
              to="/menu"
              className="text-xs uppercase tracking-[0.3em] text-haze"
            >
              Quay lại
            </Link>
          </div>
          <div className="mt-6 space-y-4 text-sm text-haze">
            <div>
              <p className="text-base text-fog">Phiên học hôm nay</p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-haze">
                Mục tiêu
              </p>
              <p className="mt-2 text-base text-fog">Nghe + nói 10 phút</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-xs">
              Gợi ý: Nhấn micro để luyện nói (tính năng demo).
            </div>
          </div>
          <div className="mt-6 space-y-3">
            <button className="w-full rounded-2xl bg-pine px-4 py-3 text-sm font-medium text-white">
              Tải transcript
            </button>
          </div>
        </aside>

        <section className="flex min-h-0 flex-col gap-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-haze">
                Conversation
              </p>
              <h1 className="text-2xl font-semibold md:text-3xl">
                Phòng hội thoại
              </h1>
            </div>
            <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs text-haze">
              Trạng thái: đang luyện
            </div>
          </div>

          <div className="h-[58vh] rounded-3xl border border-white/10 bg-gradient-to-b from-white/5 to-transparent p-6 shadow-soft">
            <div className="chat-scroll h-full overflow-y-auto pr-1" ref={listRef}>
              <div className="space-y-4 pb-2">
              {messages.length === 0 ? (
                <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-haze">
                  Nhập câu chào để bắt đầu hội thoại.
                </div>
              ) : null}
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex ${
                    msg.from === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm ${
                      msg.from === "user"
                        ? "bg-tide text-white"
                        : "bg-white/10 text-fog"
                    }`}
                  >
                    {msg.text}
                  </div>
                </div>
              ))}
              {loading ? (
                <div className="flex justify-start">
                  <div className="rounded-2xl bg-white/10 px-4 py-3 text-sm text-fog">
                    <span className="wave-dots">
                      <span />
                      <span />
                      <span />
                    </span>
                  </div>
                </div>
              ) : null}
              </div>
            </div>
          </div>

          <div className="glass rounded-3xl p-4 shadow-soft">
            <div className="flex flex-col gap-3 md:flex-row md:items-center">
              <input
                type="text"
                placeholder="Nhập câu trả lời hoặc bấm micro..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    sendMessage();
                  }
                }}
                disabled={locked}
                className="flex-1 rounded-2xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-fog outline-none focus:border-tide"
              />
              <div className="flex gap-3">
                <button
                  onClick={toggleMic}
                  disabled={locked}
                  className={`rounded-2xl px-4 py-3 text-sm ${
                    listening
                      ? "border border-ember/60 bg-ember/20 text-ember"
                      : "border border-white/10 text-fog"
                  }`}
                >
                  {listening ? "Đang nghe..." : "Micro"}
                </button>
                <button
                  onClick={() => sendMessage()}
                  disabled={loading || locked}
                  className="rounded-2xl bg-ember px-4 py-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? "Đang gửi..." : "Gửi"}
                </button>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}


