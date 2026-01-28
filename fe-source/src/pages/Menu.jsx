import { Link, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";

export default function Menu() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState({
    completed_days: 0,
    current_day: 1,
  });

  useEffect(() => {
    const userId = localStorage.getItem("user_id");
    if (!userId) {
      return;
    }
    const controller = new AbortController();
    const loadProgress = async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/progress?user_id=${encodeURIComponent(userId)}`,
          { signal: controller.signal }
        );
        if (!res.ok) {
          return;
        }
        const data = await res.json();
        setProgress({
          completed_days: Number(data.completed_days || 0),
          current_day: Number(data.current_day || 1),
        });
      } catch (err) {
        if (err.name !== "AbortError") {
          return;
        }
      }
    };
    loadProgress();
    return () => controller.abort();
  }, []);

  const isNewUser = progress.completed_days === 0;
  const highlightUpTo = isNewUser
    ? 0
    : Math.min(progress.completed_days + 1, 4);

  const handleLogout = () => {
    localStorage.removeItem("user_id");
    localStorage.removeItem("account");
    localStorage.removeItem("thread_id");
    navigate("/login");
  };

  const handleStart = () => {
    const userId = localStorage.getItem("user_id");
    if (!userId) {
      navigate("/login");
      return;
    }
    const threadId =
      (typeof crypto !== "undefined" && crypto.randomUUID && crypto.randomUUID()) ||
      `thread_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    localStorage.setItem("thread_id", threadId);
    navigate("/conversation");
  };
  return (
    <div className="min-h-screen bg-ink text-fog">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-haze">
              Dashboard
            </p>
            <h1 className="text-3xl font-semibold md:text-4xl">
              Menu học tập
            </h1>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-haze">
              English Coach · Level A1
            </div>
            <button
              onClick={handleLogout}
              className="rounded-full border border-ember/50 bg-ember/10 px-4 py-2 text-sm text-ember"
            >
              Đăng xuất
            </button>
          </div>
        </header>

        <section className="mt-10 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="glass rounded-3xl p-8 shadow-soft">
            <h2 className="text-2xl font-semibold">Bắt đầu học</h2>
            <p className="mt-2 text-sm text-haze">
              Chọn chủ đề hôm nay để vào phòng luyện tập ngay.
            </p>
            <div className="mt-6 grid gap-3 sm:grid-cols-2">
              {[
                { title: "Giao tiếp hằng ngày", desc: "Chào hỏi, mua sắm, hỏi đường" },
                { title: "Du lịch", desc: "Sân bay, khách sạn, nhà hàng" },
                { title: "Công việc", desc: "Email, họp nhóm, báo cáo" },
                { title: "Học tập", desc: "Thuyết trình, hỏi bài, thảo luận" },
                { title: "Sức khoẻ", desc: "Bác sĩ, thuốc, tình trạng" },
                { title: "Giải trí", desc: "Phim ảnh, sở thích, bạn bè" },
              ].map((item) => (
                <div
                  key={item.title}
                  className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3"
                >
                  <p className="text-sm font-semibold text-fog">{item.title}</p>
                  <p className="mt-1 text-xs text-haze">{item.desc}</p>
                </div>
              ))}
            </div>
            <div className="mt-6 flex flex-wrap gap-3">
              <button
                onClick={handleStart}
                className="rounded-2xl bg-ember px-5 py-3 text-sm font-medium text-white shadow-soft"
              >
                Bắt đầu buổi học
              </button>
            </div>
          </div>

          <aside className="space-y-6">
            <div className="rounded-3xl border border-white/10 bg-gradient-to-br from-white/10 via-white/5 to-transparent p-6">
              <h3 className="text-lg font-semibold">Chuỗi ngày học</h3>
              <p className="mt-2 text-sm text-haze">
                4 ngày luyện tập · {progress.completed_days} ngày đã hoàn thành
              </p>
              <div className="mt-4 grid grid-cols-4 gap-2">
                {[1, 2, 3, 4].map((day) => (
                  <div
                    key={day}
                    className={`h-12 rounded-2xl border text-center text-xs leading-[3rem] ${
                      day <= highlightUpTo
                        ? "border-tide bg-tide/20 text-fog"
                        : isNewUser && day === 1
                        ? "border-tide/60 bg-tide/10 text-fog font-semibold"
                        : "border-tide/40 bg-transparent text-fog"
                    }`}
                  >
                    Ngày {day}
                  </div>
                ))}
              </div>
            </div>
            <div className="glass rounded-3xl p-6 shadow-soft">
              <h3 className="text-lg font-semibold">Ghi chú nhanh</h3>
              <ul className="mt-4 space-y-3 text-sm text-haze">
                <li>• Ưu tiên luyện nghe 10 phút mỗi ngày.</li>
                <li>• Chủ đề mới sẽ làm sau khi bạn chọn.</li>
                <li>• Nhấn vào hội thoại để bắt đầu.</li>
              </ul>
            </div>
          </aside>
        </section>
      </div>
    </div>
  );
}
