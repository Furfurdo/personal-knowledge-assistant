from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from flask import Flask, redirect, render_template_string, request, url_for
from werkzeug.utils import secure_filename

from config import create_embeddings, create_llm, load_settings
from index_builder import build_faiss_index, load_documents, split_documents
from qa_engine import answer_question, load_faiss_index


PAGE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>个人知识问答助手</title>
    <style>
      :root {
        --bg: #f7f6f2;
        --card: #ffffff;
        --line: #d8d8d0;
        --text: #1f2937;
        --muted: #6b7280;
        --primary: #0f766e;
        --primary-strong: #115e59;
        --accent: #d97706;
        --ok: #166534;
        --warn: #b45309;
        --err: #b91c1c;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        background:
          radial-gradient(circle at 90% 10%, #d8f3e8 0, transparent 35%),
          radial-gradient(circle at 8% 20%, #fff4d6 0, transparent 32%),
          var(--bg);
        color: var(--text);
        font-family: "Segoe UI", "PingFang SC", "Hiragino Sans GB", sans-serif;
      }
      .container {
        max-width: 1180px;
        margin: 0 auto;
        padding: 18px 16px 40px;
      }
      .hero {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
      }
      .hero h1 {
        margin: 0 0 6px;
        font-size: 30px;
        letter-spacing: 0.2px;
      }
      .sub {
        margin: 0;
        color: var(--muted);
      }
      .status-line {
        margin-top: 10px;
        font-weight: 600;
      }
      .ok { color: var(--ok); }
      .warn { color: var(--warn); }
      .err { color: var(--err); }
      .steps {
        margin-top: 14px;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
      }
      .step {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 10px;
        background: #fcfcfb;
      }
      .step strong {
        display: block;
        margin-bottom: 4px;
      }
      .step small {
        color: var(--muted);
      }
      .step.done {
        border-color: #86efac;
        background: #f0fdf4;
      }
      .layout {
        display: grid;
        grid-template-columns: 1.45fr 1fr;
        gap: 12px;
      }
      .card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
      }
      .card h2 {
        margin: 0 0 10px;
        font-size: 21px;
      }
      .tip {
        font-size: 14px;
        color: var(--muted);
      }
      label {
        display: block;
        margin: 10px 0 6px;
        font-weight: 600;
      }
      textarea, input, button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid var(--line);
        padding: 10px 12px;
        font-size: 15px;
      }
      textarea { min-height: 108px; resize: vertical; }
      .row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
      }
      .inline {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 8px;
      }
      .inline input { width: auto; }
      .btn {
        margin-top: 12px;
        border: 0;
        color: #fff;
        background: var(--primary);
        font-weight: 600;
        cursor: pointer;
      }
      .btn:hover { background: var(--primary-strong); }
      .btn-secondary {
        margin-top: 8px;
        background: #374151;
      }
      .btn-secondary:hover { background: #1f2937; }
      .btn-danger {
        margin-top: 8px;
        background: #b91c1c;
      }
      .btn-danger:hover {
        background: #991b1b;
      }
      .chips {
        margin-top: 8px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
      .chip {
        border: 1px solid #d6d3d1;
        border-radius: 999px;
        background: #fff;
        padding: 6px 10px;
        font-size: 13px;
        cursor: pointer;
      }
      .chip:hover {
        border-color: var(--accent);
        color: #92400e;
      }
      pre {
        white-space: pre-wrap;
        margin: 0;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px;
      }
      ul {
        margin: 8px 0;
        padding-left: 18px;
      }
      details {
        margin-top: 8px;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 8px;
        background: #fafafa;
      }
      .file-list {
        max-height: 260px;
        overflow: auto;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 8px;
        background: #fafafa;
      }
      .events {
        max-height: 220px;
        overflow: auto;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 8px;
        background: #fafafa;
      }
      .event-item {
        padding: 6px 0;
        border-bottom: 1px dashed #ddd;
      }
      .event-item:last-child { border-bottom: 0; }

      .modal {
        position: fixed;
        inset: 0;
        background: rgba(15, 23, 42, 0.46);
        display: none;
        align-items: center;
        justify-content: center;
        padding: 16px;
        z-index: 30;
      }
      .modal.show { display: flex; }
      .modal-card {
        width: min(560px, 100%);
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px;
      }
      .modal-card h3 {
        margin: 0 0 8px;
      }
      .modal-actions {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-top: 10px;
      }
      @media (max-width: 920px) {
        .layout { grid-template-columns: 1fr; }
        .steps { grid-template-columns: 1fr; }
      }
    </style>
    <script>
      function useQuestion(text) {
        const el = document.getElementById("question");
        if (el) el.value = text;
      }
      function openWipeModal() {
        const el = document.getElementById("wipeModal");
        if (el) el.classList.add("show");
      }
      function closeWipeModal() {
        const el = document.getElementById("wipeModal");
        if (el) el.classList.remove("show");
      }
      function confirmWipe() {
        const ok = window.confirm("请再次确认：清空后无法恢复，是否继续？");
        if (!ok) return;
        const form = document.getElementById("wipeForm");
        if (form) form.submit();
      }
    </script>
  </head>
  <body>
    <div class="container">
      <section class="hero">
        <h1>个人知识问答助手</h1>
        <p class="sub">上传资料后，你可以像聊天一样提问，系统会基于资料内容给出回答。</p>
        {% if status %}
          <div class="status-line {{ status_type }}">{{ status }}</div>
        {% endif %}

        <div class="steps">
          <div class="step {% if product.has_files %}done{% endif %}">
            <strong>1. 上传资料</strong>
            <small>{% if product.has_files %}已上传 {{ product.file_count }} 个文件{% else %}请先上传 .txt/.md/.pdf 文件{% endif %}</small>
          </div>
          <div class="step {% if product.index_ready %}done{% endif %}">
            <strong>2. 更新知识库</strong>
            <small>{% if product.index_ready %}知识库已就绪{% else %}可勾选“自动更新知识库”或手动更新{% endif %}</small>
          </div>
          <div class="step {% if product.ready_for_qa %}done{% endif %}">
            <strong>3. 开始提问</strong>
            <small>{% if product.ready_for_qa %}可以直接提问{% else %}完成前两步后即可提问{% endif %}</small>
          </div>
        </div>
      </section>

      <div class="layout">
        <main>
          <section class="card">
            <h2>提问区</h2>
            <p class="tip">建议问题尽量具体，例如“这份资料里关于 RAG 的优缺点是什么？”</p>
            <form method="post" action="{{ url_for('ask') }}">
              <label for="question">问题</label>
              <textarea id="question" name="question" placeholder="请输入你的问题...">{{ question or '' }}</textarea>

              <div class="chips">
                <button type="button" class="chip" onclick='useQuestion("这批资料的核心观点是什么？")'>核心观点</button>
                <button type="button" class="chip" onclick='useQuestion("请按条目总结这批资料。")'>条目总结</button>
                <button type="button" class="chip" onclick='useQuestion("这批资料中有什么可执行建议？")'>行动建议</button>
                <button type="button" class="chip" onclick='useQuestion("资料里有哪些关键概念？")'>关键概念</button>
              </div>

              <div class="row">
                <div>
                  <label for="k">参考资料数量</label>
                  <input id="k" name="k" type="number" min="1" max="12" value="{{ k }}" />
                </div>
                <div>
                  <label for="auto_index">自动更新知识库</label>
                  <div class="inline">
                    <input id="auto_index" name="auto_index" type="checkbox" {% if auto_index %}checked{% endif %} />
                    <span>有新资料时先自动整理再回答</span>
                  </div>
                </div>
              </div>

              <div class="inline">
                <input id="show_context" name="show_context" type="checkbox" {% if show_context %}checked{% endif %} />
                <label for="show_context" style="margin:0;">显示回答依据</label>
              </div>

              <button class="btn" type="submit">立即提问</button>
            </form>
          </section>

          {% if answer %}
          <section class="card">
            <h2>回答结果</h2>
            <pre>{{ answer }}</pre>

            <h3>来源文件</h3>
            {% if sources %}
              <ul>
                {% for src in sources %}
                  <li>{{ src }}</li>
                {% endfor %}
              </ul>
            {% else %}
              <p class="tip">本次没有可展示的来源。</p>
            {% endif %}

            {% if contexts %}
              <h3>回答依据</h3>
              {% for c in contexts %}
                <details>
                  <summary>{{ c.ref }} {{ c.source }}</summary>
                  <pre>{{ c.snippet }}</pre>
                </details>
              {% endfor %}
            {% endif %}
          </section>
          {% endif %}
        </main>

        <aside>
          <section class="card">
            <h2>上传资料</h2>
            <p class="tip">支持 .txt / .md / .pdf，支持一次上传多个文件。</p>
            <form method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data">
              <input type="file" name="files" multiple accept=".txt,.md,.pdf" />
              <div class="inline">
                <input id="upload_reindex" name="upload_reindex" type="checkbox" checked />
                <label for="upload_reindex" style="margin:0;">上传后自动更新知识库</label>
              </div>
              <button class="btn" type="submit">上传并保存</button>
            </form>
            <form method="post" action="{{ url_for('reindex') }}">
              <button class="btn btn-secondary" type="submit">手动更新知识库</button>
            </form>
            <button class="btn btn-danger" type="button" onclick="openWipeModal()">清空全部资料</button>
          </section>

          <section class="card">
            <h2>已收录资料（{{ knowledge_files|length }}）</h2>
            <div class="file-list">
              {% if knowledge_files %}
                <ul>
                  {% for f in knowledge_files %}
                    <li>{{ f }}</li>
                  {% endfor %}
                </ul>
              {% else %}
                <p class="tip">还没有资料，请先上传文件。</p>
              {% endif %}
            </div>
          </section>

          <section class="card">
            <h2>最近操作</h2>
            <div class="events">
              {% if events %}
                {% for e in events %}
                  <div class="event-item">{{ e }}</div>
                {% endfor %}
              {% else %}
                <p class="tip">暂无记录。</p>
              {% endif %}
            </div>
          </section>
        </aside>
      </div>
    </div>

    <div id="wipeModal" class="modal" onclick="if(event.target===this) closeWipeModal()">
      <div class="modal-card">
        <h3>确认清空资料</h3>
        <p>你即将删除所有已上传资料，并清空当前知识库索引。</p>
        <p><strong>该操作不可恢复。</strong></p>
        <div class="modal-actions">
          <button class="btn btn-secondary" type="button" onclick="closeWipeModal()">取消</button>
          <button class="btn btn-danger" type="button" onclick="confirmWipe()">确认清空</button>
        </div>
        <form id="wipeForm" method="post" action="{{ url_for('wipe') }}"></form>
      </div>
    </div>
  </body>
</html>
"""


ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _friendly_error_message(exc: Exception) -> str:
    raw = str(exc).lower()
    if "connection error" in raw or "timeout" in raw:
        return "当前网络不稳定，暂时无法完成操作，请稍后重试。"
    if "api key" in raw or "unauthorized" in raw or "authentication" in raw:
        return "服务配置有误，请检查密钥设置。"
    if "index not found" in raw:
        return "知识库还没准备好，请先更新知识库后再提问。"
    if "no txt/md/pdf files found" in raw:
        return "没有找到可用资料，请先上传文件。"
    if "dimension" in raw or "assert" in raw:
        return "知识库需要更新，请点击“手动更新知识库”。"
    return "处理失败，请稍后重试。"


def _push_event(state: Dict[str, Any], text: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    state["events"].insert(0, f"[{ts}] {text}")
    state["events"] = state["events"][:12]


def _index_ready(index_dir: str) -> bool:
    path = Path(index_dir)
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()


def _build_index(settings, docs_dir: str, index_dir: str, chunk_size: int, chunk_overlap: int) -> str:
    embeddings = create_embeddings(settings)
    raw_docs = load_documents(docs_dir)
    if not raw_docs:
        return "还没有可整理的资料，请先上传文件。"

    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    build_faiss_index(chunks, embeddings, index_dir)
    return f"知识库已更新，共整理 {len(raw_docs)} 份资料。"


def _is_allowed_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _next_available_path(base_dir: Path, filename: str) -> Path:
    original = Path(filename)
    ext = original.suffix.lower()
    stem = secure_filename(original.stem) or "upload"
    clean_name = f"{stem}{ext}"
    target = base_dir / clean_name
    if not target.exists():
        return target

    idx = 1
    while True:
        candidate = base_dir / f"{stem}_{idx}{ext}"
        if not candidate.exists():
            return candidate
        idx += 1


def _list_knowledge_files(docs_path: Path) -> list[str]:
    files = []
    for p in sorted(docs_path.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        files.append(str(p.relative_to(docs_path)))
    return files


def _wipe_docs_and_index(docs_path: Path, index_dir: str) -> tuple[int, int]:
    removed_docs = 0
    for p in sorted(docs_path.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink(missing_ok=True)
            removed_docs += 1
        elif p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass

    docs_path.mkdir(parents=True, exist_ok=True)

    index_path = Path(index_dir)
    removed_index = 0
    if index_path.exists():
        for p in sorted(index_path.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink(missing_ok=True)
                removed_index += 1
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        try:
            index_path.rmdir()
        except OSError:
            pass

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    return removed_docs, removed_index


def create_app(
    docs_dir: str = "knowledge_base",
    index_dir: str = "vector_store",
    default_k: int = 6,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> Flask:
    app = Flask(__name__)
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "settings": None,
        "embeddings": None,
        "llm": None,
        "index": None,
        "status": "",
        "status_type": "ok",
        "events": [],
    }

    def runtime() -> tuple[Any, Any, Any]:
        if state["settings"] is None:
            state["settings"] = load_settings()
        if state["embeddings"] is None:
            state["embeddings"] = create_embeddings(state["settings"])
        if state["llm"] is None:
            state["llm"] = create_llm(state["settings"])
        return state["settings"], state["embeddings"], state["llm"]

    def get_index(allow_build: bool) -> Any:
        settings, embeddings, _ = runtime()
        if not _index_ready(index_dir):
            if not allow_build:
                raise FileNotFoundError("index not found")
            msg = _build_index(settings, docs_dir, index_dir, chunk_size, chunk_overlap)
            state["status"] = msg
            state["status_type"] = "ok"
            _push_event(state, msg)
            state["index"] = None
        if state["index"] is None:
            state["index"] = load_faiss_index(index_dir, embeddings)
        return state["index"]

    def render(
        question: str = "",
        answer: str = "",
        sources: list[str] | None = None,
        contexts: list[dict] | None = None,
        k: int = default_k,
        auto_index: bool = True,
        show_context: bool = True,
    ):
        knowledge_files = _list_knowledge_files(docs_path)
        product = {
            "file_count": len(knowledge_files),
            "has_files": len(knowledge_files) > 0,
            "index_ready": _index_ready(index_dir),
        }
        product["ready_for_qa"] = product["has_files"] and product["index_ready"]

        return render_template_string(
            PAGE_TEMPLATE,
            status=state["status"],
            status_type=state["status_type"],
            question=question,
            answer=answer,
            sources=sources or [],
            contexts=contexts or [],
            knowledge_files=knowledge_files,
            events=state["events"],
            product=product,
            k=k,
            auto_index=auto_index,
            show_context=show_context,
        )

    @app.get("/")
    def home():
        return render()

    @app.post("/reindex")
    def reindex():
        try:
            settings, _, _ = runtime()
            msg = _build_index(settings, docs_dir, index_dir, chunk_size, chunk_overlap)
            state["status"] = msg
            state["status_type"] = "ok"
            _push_event(state, msg)
            state["index"] = None
        except Exception as exc:
            state["status"] = _friendly_error_message(exc)
            state["status_type"] = "err"
            _push_event(state, state["status"])
        return redirect(url_for("home"))

    @app.post("/wipe")
    def wipe():
        try:
            removed_docs, removed_index = _wipe_docs_and_index(docs_path, index_dir)
            state["index"] = None
            state["status"] = (
                f"已清空完成：删除资料 {removed_docs} 个，清理索引文件 {removed_index} 个。"
            )
            state["status_type"] = "ok"
            _push_event(state, "已执行清空资料")
        except Exception as exc:
            state["status"] = _friendly_error_message(exc)
            state["status_type"] = "err"
            _push_event(state, state["status"])
        return redirect(url_for("home"))

    @app.post("/upload")
    def upload():
        files = request.files.getlist("files")
        should_reindex = bool(request.form.get("upload_reindex"))

        if not files or all(not f.filename for f in files):
            state["status"] = "你还没有选择文件。"
            state["status_type"] = "warn"
            _push_event(state, state["status"])
            return redirect(url_for("home"))

        saved = []
        skipped = []
        for file in files:
            filename = file.filename or ""
            if not filename:
                continue
            if not _is_allowed_filename(filename):
                skipped.append(filename)
                continue

            target_path = _next_available_path(docs_path, filename)
            file.save(str(target_path))
            saved.append(target_path.name)

        if not saved:
            state["status"] = "上传失败：仅支持 .txt、.md、.pdf 文件。"
            state["status_type"] = "warn"
            _push_event(state, state["status"])
            return redirect(url_for("home"))

        try:
            if should_reindex:
                settings, _, _ = runtime()
                msg = _build_index(settings, docs_dir, index_dir, chunk_size, chunk_overlap)
                state["index"] = None
                state["status"] = f"已上传 {len(saved)} 个文件，并完成知识库更新。"
                _push_event(state, msg)
            else:
                state["status"] = f"已上传 {len(saved)} 个文件。"

            if skipped:
                state["status"] += " 部分文件格式不支持，已自动跳过。"
            state["status_type"] = "ok"
            _push_event(state, state["status"])
        except Exception as exc:
            state["status"] = f"文件已上传，但更新知识库失败：{_friendly_error_message(exc)}"
            state["status_type"] = "err"
            _push_event(state, state["status"])

        return redirect(url_for("home"))

    @app.post("/ask")
    def ask():
        question = (request.form.get("question") or "").strip()
        k_raw = (request.form.get("k") or str(default_k)).strip()
        auto_index = bool(request.form.get("auto_index"))
        show_context = bool(request.form.get("show_context"))

        try:
            k = max(1, min(12, int(k_raw)))
        except ValueError:
            k = default_k

        if not question:
            state["status"] = "请先输入问题。"
            state["status_type"] = "warn"
            _push_event(state, state["status"])
            return render(question=question, k=k, auto_index=auto_index, show_context=show_context)

        try:
            _, _, llm = runtime()
            index = get_index(allow_build=auto_index)
            result = answer_question(index, llm, question, k=k, include_context=show_context)
            state["status"] = "已完成回答。"
            state["status_type"] = "ok"
            _push_event(state, f"已回答问题：{question[:30]}")
            return render(
                question=question,
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
                contexts=result.get("contexts", []),
                k=k,
                auto_index=auto_index,
                show_context=show_context,
            )
        except Exception as exc:
            state["status"] = _friendly_error_message(exc)
            state["status_type"] = "err"
            _push_event(state, state["status"])
            return render(question=question, k=k, auto_index=auto_index, show_context=show_context)

    return app
