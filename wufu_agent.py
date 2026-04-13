import streamlit as st
import os, sys, shutil, threading, subprocess, time, requests, re
from pathlib import Path
from datetime import datetime
from queue import Queue

# --- Configuration ---
OLLAMA_MODEL = "qwen2.5-coder:3b"
OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
TEMP_ROOT = Path(os.getenv("TEMP", "/tmp"))
WORK_ROOT = TEMP_ROOT / "wufu_workspace"

st.set_page_config(page_title="Wufu AI Agent", page_icon="🐉", layout="wide")

# --- UI Styling ---
st.markdown("""
<style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .log-window { background-color: #000000; color: #00ff88; font-family: 'Consolas', monospace; padding: 15px; border-radius: 5px; height: 400px; overflow-y: auto; border: 1px solid #30363d; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# --- Session State & Queue Initialization ---
if "log_queue" not in st.session_state:
    st.session_state.log_queue = Queue()

def _init_state():
    defaults = {
        "workspace": None, "project_name": "", "analysis": "", 
        "running": False, "finished": False, "logs": [], 
        "run_commands": []
    }
    for key, val in defaults.items():
        if key not in st.session_state: st.session_state[key] = val

_init_state()

def push_log(msg, level="INFO"):
    timestamp = datetime.now().strftime('%H:%M:%S')
    formatted_msg = f"[{timestamp}] {level}: {msg}"
    st.session_state.logs.append(formatted_msg)

# --- Thread-Safe Logger ---
def run_command_thread(cmd, cwd, q):
    try:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in iter(proc.stdout.readline, ""):
            if line: q.put(line.strip())
        proc.stdout.close()
        proc.wait()
        q.put("--- PROCESS FINISHED ---")
    except Exception as e:
        q.put(f"ERROR: {str(e)}")

def start_execution():
    if not st.session_state.workspace: 
        st.error("Workspace not found!")
        return
    
    st.session_state.running = True
    st.session_state.logs = []
    
    # 1. Sabse pehle 'app.py' ki sahi location dhoondo
    workspace_path = st.session_state.workspace
    app_file = None
    
    # Ye poore workspace mein 'app.py' dhoondega
    for root, dirs, files in os.walk(workspace_path):
        if "app.py" in files:
            app_file = Path(root) / "app.py"
            break
    
    if not app_file:
        push_log("ERROR: 'app.py' not found in any subdirectory!", "SYSTEM")
        st.session_state.running = False
        return

    # 2. Sahi folder (CWD) set karo jahan app.py hai
    cwd = app_file.parent
    cmd = f'python "{app_file.name}"' # File ka naam quote mein rakha hai safe side
    
    push_log(f"Found app.py at: {app_file}", "SYSTEM")
    push_log(f"Launching: {cmd} in {cwd}", "SYSTEM")
    
    # 3. Thread chalu karein
    thread = threading.Thread(
        target=run_command_thread, 
        args=(cmd, cwd, st.session_state.log_queue), 
        daemon=True
    )
    thread.start()

# --- Main UI ---
st.markdown("<h1 style='text-align: center; color: #00ff88;'>🐉 WUFU - AI Code Agent</h1>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["① SOURCE", "② ANALYSIS", "③ EXECUTE", "④ LIVE LOGS"])

with tab1:
    path_input = st.text_input("Local Project Path:", value="D:/Open_Code/Auto/Test_Project")
    if st.button("LOAD PROJECT"):
        src = Path(path_input)
        if src.exists():
            dest = WORK_ROOT / "project"
            if dest.exists(): shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            st.session_state.workspace = WORK_ROOT
            st.session_state.project_name = src.name
            push_log(f"Project {src.name} loaded.", "OK")
            st.success("Project Loaded!")
        else: st.error("Path not found!")

with tab2:
    if st.session_state.workspace:
        st.write(f"Project: **{st.session_state.project_name}**")
        if st.button("RUN AI ANALYSIS"):
            st.session_state.analysis = "Flask project detected. Command: python app.py"
            st.session_state.run_commands = ["python app.py"]
            st.info(st.session_state.analysis)
    else: st.info("Load project first.")

with tab3:
    if st.session_state.workspace:
        if st.button("🚀 LAUNCH AGENT", disabled=st.session_state.running):
            start_execution()
            st.rerun()
    else: st.info("Load project first.")

with tab4:
    # Queue se logs nikal kar session state mein dalein
    while not st.session_state.log_queue.empty():
        msg = st.session_state.log_queue.get()
        push_log(msg, "OUT")
        if "FINISHED" in msg: st.session_state.running = False

    st.markdown("#### Real-time Terminal Output")
    log_content = "\n".join(st.session_state.logs)
    st.markdown(f'<div class="log-window">{log_content}</div>', unsafe_allow_html=True)
    
    if st.session_state.running:
        time.sleep(1)
        st.rerun()