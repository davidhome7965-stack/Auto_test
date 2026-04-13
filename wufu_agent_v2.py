"""
🐉 WUFU - AI Code Agent (Ollama Version)
=========================================
Requirements:
    pip install streamlit requests gitpython

Ollama install: https://ollama.com/download
Model pull:     ollama pull qwen2.5-coder:3b

Run:
    streamlit run wufu_agent.py
"""

import streamlit as st
import os, sys, shutil, threading, subprocess, time, requests, re, json, zipfile
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty

# ─────────────────────────────────────────────────────────────
# CONFIG  (change karo agar zaroorat ho)
# ─────────────────────────────────────────────────────────────
OLLAMA_BASE  = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")

TEMP_ROOT = Path(os.getenv("TEMP", "/tmp"))
WORK_ROOT = TEMP_ROOT / "wufu_workspace"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

SKIP_DIRS = {".git","__pycache__","node_modules",".venv","venv","env",
             ".next","dist","build",".mypy_cache",".idea",".vscode"}
TEXT_EXTS = {".py",".js",".ts",".jsx",".tsx",".html",".css",".json",
             ".yaml",".yml",".txt",".md",".sh",".toml",".cfg",".ini",
             ".xml",".rs",".go",".rb",".java",".cpp",".c",".h",".php"}
MAX_TOTAL = 8000

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wufu AI Agent",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

.main { background-color: #0d1117; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1100px; }

.wufu-header {
    text-align: center; padding: 1.5rem 0 1rem;
    border-bottom: 1px solid #21262d; margin-bottom: 1.5rem;
}
.wufu-header h1 { font-size: 2.2rem; color: #00ff88; margin: 0; letter-spacing: -0.02em; }
.wufu-header p  { color: #484f58; font-size: 0.9rem; margin-top: 4px; }

.terminal-box {
    background: #010409; color: #00ff88;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 0.82rem; padding: 14px 16px;
    border-radius: 6px; border: 1px solid #21262d;
    min-height: 300px; max-height: 450px;
    overflow-y: auto; line-height: 1.65;
    white-space: pre-wrap; word-break: break-all;
}
.info-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
}
.info-card h4 {
    color: #388bfd; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 0 0 6px; font-family: 'JetBrains Mono', monospace;
}
.info-card p { color: #c9d1d9; font-size: 0.88rem; margin: 0; }
.tag {
    display: inline-block; padding: 2px 9px;
    border-radius: 999px; font-size: 0.72rem;
    margin: 2px; font-family: 'JetBrains Mono', monospace;
}
.tag-blue   { background:#0d2240; color:#58a6ff; border:1px solid #1f4070; }
.tag-green  { background:#0d2010; color:#3fb950; border:1px solid #1a4020; }
.tag-amber  { background:#2a1a00; color:#d29922; border:1px solid #4a3000; }
.tag-purple { background:#1e0d2a; color:#bc8cff; border:1px solid #3a1a5a; }

.step-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 12px; border-radius: 8px;
    border: 1px solid #21262d; margin-bottom: 6px;
    background: #161b22;
}
.step-dot {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}
.dot-pending { background:#21262d; color:#484f58; }
.dot-running { background:#0d2240; color:#58a6ff; animation: blink 0.9s infinite; }
.dot-done    { background:#0d2010; color:#3fb950; }
.dot-error   { background:#2a0d0d; color:#f85149; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }
.step-name { font-size: 0.88rem; color: #c9d1d9; }
.step-sub  { font-size: 0.72rem; color: #484f58; margin-top: 1px;
             font-family: 'JetBrains Mono', monospace; }

.ollama-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 999px;
    background: #0d2a1a; border: 1px solid #1a5030;
    color: #3fb950; font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
}
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: #161b22 !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important; border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "logs": [], "workspace": None, "project_name": "",
        "analysis": None, "running": False, "finished": False,
        "log_queue": Queue(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
LEVEL_COLORS = {
    "SYS":   "#58a6ff", "OK":    "#3fb950", "ERROR": "#f85149",
    "WARN":  "#d29922", "AI":    "#bc8cff", "OUT":   "#00ff88",
    "INFO":  "#8b949e", "DIM":   "#30363d",
}

def push_log(msg: str, level: str = "INFO"):
    ts    = datetime.now().strftime("%H:%M:%S")
    color = LEVEL_COLORS.get(level.upper(), "#c9d1d9")
    safe  = msg.replace("<","&lt;").replace(">","&gt;")
    entry = (f'<span style="color:#30363d">[{ts}]</span> '
             f'<span style="color:{color}">{level:5s}</span>  {safe}')
    st.session_state.logs.append(entry)

def render_terminal():
    body = "<br>".join(st.session_state.logs) if st.session_state.logs \
           else '<span style="color:#30363d">--- Wufu Terminal Ready ---</span>'
    st.markdown(f'<div class="terminal-box">{body}</div>', unsafe_allow_html=True)

def drain_queue():
    q = st.session_state.log_queue
    while True:
        try:
            msg = q.get_nowait()
            if "--- PROCESS FINISHED ---" in msg:
                st.session_state.running  = False
                st.session_state.finished = True
                push_log("Process finished.", "OK")
            else:
                push_log(msg, "OUT")
        except Empty:
            break

# ─────────────────────────────────────────────────────────────
# OLLAMA HELPERS
# ─────────────────────────────────────────────────────────────
def ollama_is_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def ollama_models() -> list:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []

def ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 800}
    }
    r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")

# ─────────────────────────────────────────────────────────────
# FILE COLLECTION
# ─────────────────────────────────────────────────────────────
def collect_files(root: Path) -> str:
    root    = Path(root)
    chunks  = []
    total   = 0
    # priority files pehle
    priority = ["requirements.txt","package.json","app.py","main.py",
                "index.js","index.ts","manage.py","server.py","run.py",
                "Dockerfile","docker-compose.yml","pyproject.toml","setup.py"]
    for fname in priority:
        for p in root.rglob(fname):
            if any(d in SKIP_DIRS for d in p.parts): continue
            try:
                text  = p.read_text(errors="replace")[:3000]
                chunk = f"### {p.relative_to(root)}\n{text}\n"
                if total + len(chunk) < MAX_TOTAL:
                    chunks.append(chunk); total += len(chunk)
            except Exception: pass
    # baaki files
    for p in sorted(root.rglob("*")):
        if any(d in SKIP_DIRS for d in p.parts): continue
        if not p.is_file(): continue
        if p.suffix.lower() not in TEXT_EXTS: continue
        if p.name in priority: continue
        try:
            text  = p.read_text(errors="replace")[:2000]
            chunk = f"### {p.relative_to(root)}\n{text}\n"
            if total + len(chunk) > MAX_TOTAL: break
            chunks.append(chunk); total += len(chunk)
        except Exception: pass
    return "\n".join(chunks) or "No readable files found."

# ─────────────────────────────────────────────────────────────
# AI ANALYSIS
# ─────────────────────────────────────────────────────────────
def analyze_with_ollama(code_text: str, hint: str = "") -> dict:
    prompt = f"""You are a code analysis expert. Analyze these project files and return ONLY a JSON object. No markdown. No explanation. Just JSON.

{f'User hint: {hint}' if hint else ''}

Return EXACTLY this JSON structure:
{{
  "projectType": "Flask API / React App / Django / FastAPI / Node.js / ML Script / CLI Tool",
  "language": "Python / JavaScript / TypeScript / etc",
  "framework": "Flask / Django / FastAPI / Express / React / Next.js / none",
  "dependencies": ["dep1","dep2","dep3"],
  "installCommand": "pip install -r requirements.txt",
  "runCommand": "python app.py",
  "entryFile": "app.py",
  "port": "5000",
  "description": "2 sentences about what this project does."
}}

PROJECT FILES:
{code_text}

Return ONLY the JSON object. Nothing else."""

    raw = ollama_generate(prompt)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    m   = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try: return json.loads(m.group(0))
        except json.JSONDecodeError: pass
    try: return json.loads(raw)
    except Exception: pass
    push_log("AI response parse nahi hua, basic detection use kar raha hoon...", "WARN")
    return _basic_detect(code_text, hint)

def _basic_detect(code_text: str, hint: str = "") -> dict:
    ct = code_text.lower() + hint.lower()
    if "flask" in ct:
        return {"projectType":"Flask App","language":"Python","framework":"Flask",
                "dependencies":["flask"],"installCommand":"pip install flask",
                "runCommand":"python app.py","entryFile":"app.py","port":"5000",
                "description":"Flask web application."}
    if "fastapi" in ct:
        return {"projectType":"FastAPI App","language":"Python","framework":"FastAPI",
                "dependencies":["fastapi","uvicorn"],
                "installCommand":"pip install fastapi uvicorn",
                "runCommand":"uvicorn main:app --reload","entryFile":"main.py",
                "port":"8000","description":"FastAPI application."}
    if "django" in ct:
        return {"projectType":"Django App","language":"Python","framework":"Django",
                "dependencies":["django"],"installCommand":"pip install django",
                "runCommand":"python manage.py runserver","entryFile":"manage.py",
                "port":"8000","description":"Django web application."}
    if "express" in ct or "require('express')" in ct:
        return {"projectType":"Node.js App","language":"JavaScript","framework":"Express",
                "dependencies":["express"],"installCommand":"npm install",
                "runCommand":"node index.js","entryFile":"index.js","port":"3000",
                "description":"Express.js application."}
    if "react" in ct:
        return {"projectType":"React App","language":"JavaScript","framework":"React",
                "dependencies":["react","react-dom"],"installCommand":"npm install",
                "runCommand":"npm start","entryFile":"src/index.js","port":"3000",
                "description":"React application."}
    if "import" in ct or ".py" in ct:
        return {"projectType":"Python Script","language":"Python","framework":"none",
                "dependencies":[],"installCommand":"",
                "runCommand":"python main.py","entryFile":"main.py","port":"null",
                "description":"Python script."}
    return {"projectType":"Unknown","language":"Unknown","framework":"none",
            "dependencies":[],"installCommand":"",
            "runCommand":"python app.py","entryFile":"app.py","port":"null",
            "description":"Could not auto-detect. Check run command manually."}

# ─────────────────────────────────────────────────────────────
# INSTALL DEPENDENCIES
# ─────────────────────────────────────────────────────────────
def install_deps(ana: dict, cwd: Path) -> bool:
    cmd = (ana.get("installCommand") or "").strip()
    if not cmd:
        push_log("Install command nahi hai — skip.", "WARN")
        return True
    if cmd.startswith("pip "):
        cmd = f'"{sys.executable}" -m {cmd}'
    push_log(f"Installing: {cmd}", "SYS")
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=str(cwd),
            capture_output=True, text=True, timeout=300,
            env={**os.environ}
        )
        for line in (result.stdout + result.stderr).splitlines()[-15:]:
            if line.strip(): push_log(line, "OUT")
        if result.returncode != 0:
            push_log("Install failed (non-zero). Aage badh rahe hain...", "WARN")
            return False
        push_log("Dependencies install ho gayi ✓", "OK")
        return True
    except Exception as e:
        push_log(f"Install error: {e}", "ERROR")
        return False

# ─────────────────────────────────────────────────────────────
# BACKGROUND PROCESS RUNNER
# ─────────────────────────────────────────────────────────────
def _run_thread(cmd: str, cwd: Path, q: Queue):
    try:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env={**os.environ}
        )
        for line in iter(proc.stdout.readline, ""):
            if line.strip(): q.put(line.rstrip())
        proc.stdout.close()
        proc.wait()
    except Exception as e:
        q.put(f"[ERROR] {e}")
    finally:
        q.put("--- PROCESS FINISHED ---")

def launch_project(cmd: str, cwd: Path):
    st.session_state.running  = True
    st.session_state.finished = False
    t = threading.Thread(
        target=_run_thread,
        args=(cmd, cwd, st.session_state.log_queue),
        daemon=True
    )
    t.start()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def find_entry(workspace: Path, entry_name: str):
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if entry_name in files:
            return Path(root) / entry_name
    return None

def build_run_cmd(ana: dict, entry_path: Path) -> str:
    run_cmd = ana.get("runCommand", f"python {entry_path.name}")
    py = sys.executable
    if run_cmd.startswith("python "):
        return run_cmd.replace("python ", f'"{py}" ', 1)
    if run_cmd.startswith("python3 "):
        return run_cmd.replace("python3 ", f'"{py}" ', 1)
    return run_cmd

def step_html(num, name, sub, status="pending"):
    cls  = {"pending":"dot-pending","running":"dot-running",
            "done":"dot-done","error":"dot-error"}[status]
    icon = {"pending":str(num),"running":"●","done":"✓","error":"✗"}[status]
    return (f'<div class="step-row"><div class="step-dot {cls}">{icon}</div>'
            f'<div><div class="step-name">{name}</div>'
            f'<div class="step-sub">{sub}</div></div></div>')

# ═════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════
st.markdown("""
<div class="wufu-header">
  <h1>🐉 WUFU — AI Code Agent</h1>
  <p>Project do → Ollama se analyze karo → Dependencies install karo → Auto-run karo</p>
</div>
""", unsafe_allow_html=True)

# ── Ollama Status Bar ──
col_s1, col_s2, col_s3 = st.columns([2, 2, 4])
with col_s1:
    alive = ollama_is_running()
    if alive:
        st.markdown('<div class="ollama-badge">🟢 Ollama Running</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="ollama-badge" style="color:#f85149;border-color:#4a1a1a;background:#2a0d0d">'
            '🔴 Ollama Offline</div>', unsafe_allow_html=True)
with col_s2:
    models = ollama_models()
    if models:
        chosen = st.selectbox("Model:", models,
                              index=models.index(OLLAMA_MODEL) if OLLAMA_MODEL in models else 0,
                              label_visibility="collapsed")
        OLLAMA_MODEL = chosen
    else:
        st.caption(f"Model: `{OLLAMA_MODEL}`")
with col_s3:
    st.caption(f"`{OLLAMA_BASE}`  ·  Pull: `ollama pull {OLLAMA_MODEL}`")

if not alive:
    st.error("⚠️ Ollama chal nahi raha! Terminal mein chalaao:  `ollama serve`")

st.divider()

# ═════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["① SOURCE", "② ANALYSIS", "③ PIPELINE", "④ TERMINAL"])

# ─────────────────────────────────────────────────────────────
# TAB 1 — SOURCE
# ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Project Source")
    src_type = st.radio("", ["📂 Local Folder", "🌐 GitHub URL", "🗜️ ZIP File"],
                        horizontal=True, label_visibility="collapsed")
    hint = st.text_input("Project hint (optional):",
                         placeholder="e.g. Flask REST API, React todo app, Django blog...")

    # Local Folder
    if src_type == "📂 Local Folder":
        folder_path = st.text_input("Folder path:",
            value="D:/Open_Code/Auto/Test_Project",
            placeholder="D:/Projects/MyApp  ya  /home/user/myapp")
        if st.button("📂 Load Folder", use_container_width=True, type="primary"):
            p = Path(folder_path.strip())
            if p.exists() and p.is_dir():
                dest = WORK_ROOT / "project"
                if dest.exists(): shutil.rmtree(dest, ignore_errors=True)
                shutil.copytree(str(p), str(dest), dirs_exist_ok=True)
                st.session_state.workspace    = dest
                st.session_state.project_name = p.name
                st.session_state.analysis     = None
                st.session_state.finished     = False
                fc = sum(1 for _ in dest.rglob("*") if _.is_file())
                push_log(f"Loaded: {p.name}  ({fc} files)", "OK")
                st.success(f"✅ `{p.name}` load ho gaya!  ({fc} files)")
            else:
                st.error("❌ Folder nahi mila. Path check karo.")

    # GitHub
    elif src_type == "🌐 GitHub URL":
        gh_url = st.text_input("GitHub URL:", placeholder="https://github.com/user/repo")
        branch  = st.text_input("Branch:", value="main")
        if st.button("🌐 Clone", use_container_width=True, type="primary"):
            if gh_url.strip():
                dest = WORK_ROOT / "project"
                if dest.exists(): shutil.rmtree(dest, ignore_errors=True)
                with st.spinner("Cloning..."):
                    cmd = f'git clone --depth=1 --branch {branch} "{gh_url.strip()}" "{dest}"'
                    r   = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
                if r.returncode == 0 and dest.exists():
                    st.session_state.workspace    = dest
                    st.session_state.project_name = gh_url.strip().rstrip("/").split("/")[-1]
                    st.session_state.analysis     = None
                    fc = sum(1 for _ in dest.rglob("*") if _.is_file())
                    push_log(f"Cloned: {st.session_state.project_name}  ({fc} files)", "OK")
                    st.success("✅ Clone ho gaya!")
                else:
                    st.error(f"❌ Clone failed:\n{r.stderr[:400]}")
            else:
                st.error("URL daalo pehle.")

    # ZIP
    else:
        uploaded = st.file_uploader("ZIP upload karo:", type=["zip"])
        if uploaded and st.button("🗜️ Extract", use_container_width=True, type="primary"):
            dest = WORK_ROOT / "project"
            if dest.exists(): shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            zp = WORK_ROOT / uploaded.name
            zp.write_bytes(uploaded.read())
            with zipfile.ZipFile(str(zp), "r") as zf:
                zf.extractall(str(dest))
            st.session_state.workspace    = dest
            st.session_state.project_name = uploaded.name.replace(".zip","")
            st.session_state.analysis     = None
            fc = sum(1 for _ in dest.rglob("*") if _.is_file())
            push_log(f"ZIP extracted: {uploaded.name}  ({fc} files)", "OK")
            st.success("✅ ZIP extract ho gaya!")

    if st.session_state.workspace:
        ws = st.session_state.workspace
        fc = sum(1 for _ in ws.rglob("*") if _.is_file())
        st.info(f"📁 **{st.session_state.project_name}**  ·  {fc} files  ·  `{ws}`")

# ─────────────────────────────────────────────────────────────
# TAB 2 — ANALYSIS
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### AI Code Analysis (Ollama)")

    if not st.session_state.workspace:
        st.info("Pehle **① SOURCE** mein project load karo.")
    else:
        ca, cb = st.columns([3,1])
        with ca:
            st.write(f"Project: **{st.session_state.project_name}**")
        with cb:
            do_analyze = st.button("🤖 Analyze", use_container_width=True, type="primary")

        if do_analyze:
            if not ollama_is_running():
                st.error("❌ Ollama chal nahi raha! `ollama serve` run karo.")
            else:
                with st.spinner(f"🤖 {OLLAMA_MODEL} analysis kar raha hai..."):
                    push_log("Files collect ho rahi hain...", "SYS")
                    code_text = collect_files(st.session_state.workspace)
                    push_log(f"~{len(code_text)} chars collected", "SYS")
                    push_log(f"Ollama ({OLLAMA_MODEL}) bhej raha hoon...", "AI")
                    try:
                        result = analyze_with_ollama(code_text, hint)
                        st.session_state.analysis = result
                        push_log(f"Analysis done: {result.get('projectType','?')}", "OK")
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
                        push_log(f"Error: {e}", "ERROR")

        ana = st.session_state.analysis
        if ana:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="info-card"><h4>Project Type</h4>'
                    f'<p><span class="tag tag-blue">{ana.get("projectType","?")}</span>'
                    f'&nbsp;<span class="tag tag-amber">{ana.get("language","?")}</span></p></div>',
                    unsafe_allow_html=True)
            with c2:
                fw = ana.get("framework","none") or "none"
                st.markdown(
                    f'<div class="info-card"><h4>Framework</h4>'
                    f'<p><span class="tag tag-green">{fw}</span></p></div>',
                    unsafe_allow_html=True)
            with c3:
                port = ana.get("port") or "—"
                st.markdown(
                    f'<div class="info-card"><h4>Port</h4>'
                    f'<p><span class="tag tag-amber">{port}</span></p></div>',
                    unsafe_allow_html=True)

            deps     = ana.get("dependencies") or []
            dep_html = "".join(f'<span class="tag tag-blue">{d}</span>' for d in deps) or "None"
            st.markdown(
                f'<div class="info-card"><h4>Dependencies</h4><p>{dep_html}</p></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="info-card"><h4>Install Command</h4>'
                f'<p style="font-family:\'JetBrains Mono\',monospace;color:#d29922">'
                f'$ {ana.get("installCommand","—")}</p></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="info-card"><h4>Run Command</h4>'
                f'<p style="font-family:\'JetBrains Mono\',monospace;color:#3fb950">'
                f'$ {ana.get("runCommand","—")}</p></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="info-card"><h4>Description</h4>'
                f'<p>{ana.get("description","—")}</p></div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 3 — PIPELINE
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Automated Pipeline")
    drain_queue()

    if not st.session_state.workspace:
        st.info("Pehle **① SOURCE** mein project load karo.")
    elif not st.session_state.analysis:
        st.info("Pehle **② ANALYSIS** se AI analysis karwao.")
    else:
        ana = st.session_state.analysis

        # NOT STARTED
        if not st.session_state.running and not st.session_state.finished:
            col_go, col_info = st.columns([1, 2])
            with col_go:
                go = st.button("🚀 LAUNCH PIPELINE", use_container_width=True, type="primary")
            with col_info:
                st.caption(f"Run: `{ana.get('runCommand','?')}`  |  Entry: `{ana.get('entryFile','?')}`")

            st.markdown(
                step_html(1,"Load Project",        "Workspace ready ✓",              "done")
              + step_html(2,"AI Analysis",          f"{ana.get('projectType','?')} ✓","done")
              + step_html(3,"Install Dependencies", ana.get("installCommand","—"),    "pending")
              + step_html(4,"Launch Project",       ana.get("runCommand","—"),        "pending"),
                unsafe_allow_html=True)

            if go:
                push_log("=== PIPELINE START ===", "SYS")

                # Step 3
                install_deps(ana, st.session_state.workspace)

                # Step 4
                entry_name = ana.get("entryFile","app.py")
                entry_path = find_entry(st.session_state.workspace, entry_name)
                if not entry_path:
                    entry_path = find_entry(st.session_state.workspace, "main.py")

                if not entry_path:
                    push_log(f"Entry file '{entry_name}' nahi mila!", "ERROR")
                    st.error(f"❌ `{entry_name}` nahi mila workspace mein.")
                else:
                    cwd = entry_path.parent
                    cmd = build_run_cmd(ana, entry_path)
                    push_log(f"Entry: {entry_path}", "SYS")
                    push_log(f"CWD:   {cwd}", "SYS")
                    push_log(f"CMD:   {cmd}", "SYS")
                    launch_project(cmd, cwd)
                    port = ana.get("port","")
                    if port and port not in ("null","None",""):
                        push_log(f"App: http://localhost:{port}", "OK")

                st.rerun()

        # RUNNING
        elif st.session_state.running:
            drain_queue()
            st.markdown(
                step_html(1,"Load Project",        "Done ✓","done")
              + step_html(2,"AI Analysis",          "Done ✓","done")
              + step_html(3,"Install Dependencies", "Done ✓","done")
              + step_html(4,"Launch Project",       "Running... (Terminal tab dekho)","running"),
                unsafe_allow_html=True)

            port = ana.get("port","")
            if port and port not in ("null","None",""):
                st.success(f"🌐 App chal raha hai: http://localhost:{port}")

            if st.button("🔴 Stop Process"):
                st.session_state.running  = False
                st.session_state.finished = True
                push_log("Stopped by user.", "WARN")

            time.sleep(1)
            st.rerun()

        # FINISHED
        else:
            st.markdown(
                step_html(1,"Load Project",        "Done ✓","done")
              + step_html(2,"AI Analysis",          "Done ✓","done")
              + step_html(3,"Install Dependencies", "Done ✓","done")
              + step_html(4,"Launch Project",       "Finished","done"),
                unsafe_allow_html=True)
            st.success("✅ Pipeline complete!")

            if st.button("🔄 Run Again"):
                st.session_state.finished = False
                st.session_state.logs     = []
                while True:
                    try: st.session_state.log_queue.get_nowait()
                    except Empty: break
                st.rerun()

# ─────────────────────────────────────────────────────────────
# TAB 4 — TERMINAL
# ─────────────────────────────────────────────────────────────
with tab4:
    drain_queue()
    ct1, ct2 = st.columns([6, 1])
    with ct1:
        st.markdown("### Live Terminal Output")
    with ct2:
        if st.button("🗑 Clear"):
            st.session_state.logs = []
            st.rerun()

    render_terminal()

    if st.session_state.running:
        st.caption("🟢 Running... auto-refresh every 1s")
        time.sleep(1)
        st.rerun()
    elif st.session_state.finished:
        st.caption("⚪ Process finished.")
    else:
        st.caption("⚫ Pipeline abhi shuru nahi hua.")
