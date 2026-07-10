import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from flask.json.provider import DefaultJSONProvider

sys.path.insert(0, os.path.dirname(__file__))

from llm import ask_llm
from nl2sql import query_all_domains, get_schema, get_cross_reference_summary
from rag_engine import retrieve_context
from governance import masker, rbac
from dashboard_engine import build_dashboard
from predictor import find_similar_question


class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return super().default(obj)


app = Flask(__name__)
app.json = NumpyJSONProvider(app)


def convert_for_json(obj):
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)

DB_KEYWORDS = [
    "employee", "employees", "department", "ticket", "tickets",
    "device", "devices", "asset", "assets", "finance", "salary",
    "maintenance", "sensor", "machine", "iot", "email", "count",
    "total", "how many", "average", "avg", "maximum", "minimum",
    "highest", "lowest", "top", "list", "show", "display", "find",
    "sales", "revenue", "cost", "order", "orders", "customer",
    "hr", "attrition", "turnover", "failure", "predict"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Intelligence</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/remixicon@4.5.0/fonts/remixicon.min.css" rel="stylesheet">
    <style>
        :root { --bg: #f7f7f8; --surface: #ffffff; --border: #e8e8ea; --border2: #d0d0d8; --text: #0d0d0d; --text2: #5c5c6a; --text3: #8e8ea0; --text4: #b0b0b8; --hover: #f0f0f1; --card-shadow: 0 1px 3px rgba(0,0,0,0.04); --input-shadow: 0 2px 8px rgba(0,0,0,0.04); --green: #10a37f; --green-light: #ecfdf5; --green-border: #d1fae5; --red: #ef4444; --chart-grid: #f0f0f1; --ar: #d0d0d0; }
        .dark { --bg: #0d0d0d; --surface: #1a1a1a; --border: #2a2a2a; --border2: #333; --text: #f0f0f0; --text2: #a0a0a8; --text3: #6a6a72; --text4: #555; --hover: #222; --card-shadow: 0 1px 3px rgba(0,0,0,0.2); --input-shadow: 0 2px 8px rgba(0,0,0,0.2); --green: #10a37f; --green-light: #0a2a1f; --green-border: #1a3a2f; --red: #f87171; --ar: #444; }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; letter-spacing: -0.01em; overflow: hidden; transition: background 0.2s, color 0.2s; }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--ar); border-radius: 3px; }

        .app-layout { display: flex; flex: 1; overflow: hidden; }
        .sidebar { width: 56px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; align-items: center; padding: 14px 0; gap: 6px; flex-shrink: 0; }
        .sidebar-logo { width: 34px; height: 34px; background: var(--green); border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700; color: #fff; margin-bottom: 14px; }
        .sidebar-btn { width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: var(--text3); font-size: 19px; cursor: pointer; transition: all 0.15s; border: none; background: transparent; }
        .sidebar-btn:hover { color: var(--text); background: var(--hover); }
        .sidebar-btn.active { color: var(--green); background: var(--green-light); }

        .main-area { flex: 1; display: flex; flex-direction: column; min-width: 0; }

        .topbar { height: 52px; background: var(--surface); border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; padding: 0 24px; flex-shrink: 0; }
        .topbar-left { display: flex; align-items: center; gap: 12px; }
        .topbar-left h1 { font-size: 18px; font-weight: 600; color: var(--text); }
        .topbar-badge { font-size: 11px; background: var(--hover); padding: 4px 10px; border-radius: 6px; color: var(--text3); border: 1px solid var(--border); font-weight: 500; }
        .topbar-right { display: flex; align-items: center; gap: 14px; }
        .topbar-right .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); }
        .topbar-right span { font-size: 13px; color: var(--text3); }
        .theme-toggle { background: var(--hover); border: 1px solid var(--border); border-radius: 8px; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; cursor: pointer; color: var(--text3); font-size: 16px; transition: all 0.15s; }
        .theme-toggle:hover { color: var(--text); border-color: var(--border2); }

        .chat-area { flex: 1; overflow-y: auto; padding: 28px 24px; display: flex; flex-direction: column; gap: 22px; }
        .message { max-width: 860px; padding: 0; line-height: 1.8; font-size: 17px; animation: fadeIn 0.2s ease; margin: 0 auto; width: 100%; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
        .user { background: var(--hover); border: 1px solid var(--border); border-radius: 14px; padding: 14px 18px; align-self: flex-end; color: var(--text); max-width: 70%; font-size: 16px; line-height: 1.7; }
        .assistant { align-self: flex-start; width: 100%; }
        .assistant .label-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .assistant .label-row .avatar { width: 28px; height: 28px; border-radius: 8px; background: var(--green); display: flex; align-items: center; justify-content: center; font-size: 12px; color: #fff; font-weight: 700; }
        .assistant .label-row .name { font-size: 15px; font-weight: 600; color: var(--text); }
        .sql-block { font-size: 14px; font-family: 'JetBrains Mono', 'Fira Code', monospace; background: var(--bg); border: 1px solid var(--border); padding: 12px 16px; border-radius: 10px; margin: 10px 0; color: #818cf8; overflow-x: auto; white-space: pre-wrap; line-height: 1.5; }
        .assistant pre { font-size: 17px; line-height: 1.8; color: var(--text); white-space: pre-wrap; margin: 4px 0; }
        .source-tags { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
        .source-tags span { font-size: 12px; color: var(--text3); background: var(--hover); padding: 4px 12px; border-radius: 6px; border: 1px solid var(--border); }

        .input-area { border-top: 1px solid var(--border); padding: 16px 24px 20px; background: var(--surface); flex-shrink: 0; }
        .input-row { display: flex; gap: 12px; max-width: 820px; margin: 0 auto; background: var(--bg); border: 1px solid var(--border2); border-radius: 16px; padding: 6px; transition: border 0.2s, box-shadow 0.2s; box-shadow: var(--input-shadow); }
        .input-row:focus-within { border-color: var(--green); box-shadow: 0 2px 16px rgba(16,163,127,0.15); }
        .input-row input { flex: 1; padding: 14px 18px; border: none; background: transparent; color: var(--text); font-size: 17px; outline: none; font-family: inherit; }
        .input-row input::placeholder { color: var(--text4); }
        .input-row button { padding: 12px 24px; background: var(--green); color: #fff; border: none; border-radius: 10px; font-weight: 600; font-size: 15px; cursor: pointer; transition: background 0.15s; font-family: inherit; }
        .input-row button:hover { background: #0d8c6d; }

        .empty-state { text-align: center; padding: 100px 20px; margin: auto; }
        .empty-state .icon-box { width: 56px; height: 56px; margin: 0 auto 18px; background: var(--green-light); border-radius: 16px; display: flex; align-items: center; justify-content: center; font-size: 24px; color: var(--green); border: 1px solid var(--green-border); }
        .empty-state h2 { font-size: 24px; font-weight: 600; color: var(--text); margin-bottom: 8px; }
        .empty-state p { font-size: 15px; color: var(--text3); line-height: 1.5; }
        .empty-state .examples { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-top: 20px; }
        .empty-state .examples button { font-size: 14px; color: var(--text3); background: var(--hover); border: 1px solid var(--border); padding: 10px 18px; border-radius: 10px; cursor: pointer; font-family: inherit; transition: all 0.15s; }
        .empty-state .examples button:hover { color: var(--green); border-color: var(--green); background: var(--green-light); }

        .dashboard { margin-top: 20px; border-top: 1px solid var(--border); padding-top: 20px; }
        .dash-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
        .dash-header h3 { font-size: 14px; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: 0.5px; }
        .dash-header .row-count { font-size: 13px; color: var(--text3); background: var(--hover); padding: 4px 12px; border-radius: 6px; border: 1px solid var(--border); }

        .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 20px; }
        .kpi-card { background: var(--surface); border: 1px solid var(--border); padding: 18px 20px; border-radius: 14px; box-shadow: var(--card-shadow); }
        .kpi-card .kpi-label { font-size: 12px; color: var(--text3); font-weight: 500; text-transform: uppercase; letter-spacing: 0.4px; }
        .kpi-card .kpi-value { font-size: 32px; font-weight: 700; color: var(--text); margin-top: 4px; }

        .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 20px; }
        .chart-box { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px; box-shadow: var(--card-shadow); }
        .chart-box h4 { font-size: 13px; color: var(--text3); font-weight: 500; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.3px; padding-left: 4px; }
        .chart-box img { width: 100%; height: auto; border-radius: 8px; display: block; }

        .insight-box { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 20px; box-shadow: var(--card-shadow); }
        .insight-box h4 { font-size: 16px; font-weight: 600; color: var(--text); margin-bottom: 8px; }
        .insight-box p { font-size: 16px; color: var(--text2); line-height: 1.8; }
        .insight-box ul { padding-left: 22px; margin-top: 10px; }
        .insight-box li { font-size: 16px; color: var(--text2); line-height: 1.8; margin-bottom: 4px; }

        .error { color: var(--red); }
        .loading { opacity: 0.4; }
    </style>
</head>
<body>
    <div class="app-layout">
        <div class="sidebar">
            <div class="sidebar-logo">EI</div>
            <button class="sidebar-btn active" title="Chat"><i class="ri-chat-3-line"></i></button>
            <button class="sidebar-btn" title="Explore"><i class="ri-database-2-line"></i></button>
            <button class="sidebar-btn" title="History"><i class="ri-history-line"></i></button>
            <button class="sidebar-btn" title="Training Data" onclick="alert('Check /learn/stats for training stats')"><i class="ri-bar-chart-2-line"></i></button>
            <div style="margin-top:auto;">
                <button class="sidebar-btn" title="Theme" onclick="toggleTheme()"><i id="themeIcon" class="ri-moon-line"></i></button>
            </div>
        </div>
        <div class="main-area">
            <div class="topbar">
                <div class="topbar-left">
                    <h1>Enterprise Intelligence</h1>
                    <span class="topbar-badge">NL2SQL + RAG</span>
                </div>
                <div class="topbar-right">
                    <span class="status-dot"></span>
                    <span id="status">Ready</span>
                    <div class="theme-toggle" onclick="toggleTheme()"><i id="themeIcon2" class="ri-moon-line"></i></div>
                </div>
            </div>
            <div class="chat-area" id="chat">
                <div class="empty-state" id="empty">
                    <div class="icon-box"><i class="ri-lightbulb-flash-line" style="font-size:20px;"></i></div>
                    <h2>Ask anything about your data</h2>
                    <p>Natural language queries powered by AI</p>
                    <div class="examples" style="flex-direction:column;align-items:stretch;max-width:500px;margin:20px auto 0;">
                        <div style="font-size:13px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:0.3px;margin-bottom:6px;">HR</div>
                        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
                            <button onclick="fillExample('Show attrition rate by department')">Attrition by dept</button>
                            <button onclick="fillExample('Average monthly income by job role')">Avg salary by role</button>
                            <button onclick="fillExample('Employees with overtime and performance rating')">Overtime &amp; performance</button>
                        </div>
                        <div style="font-size:13px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:0.3px;margin-bottom:6px;">Tickets</div>
                        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
                            <button onclick="fillExample('How many high priority tickets are unresolved?')">Open high-priority</button>
                            <button onclick="fillExample('Average resolution time by channel and region')">Resolution by channel</button>
                            <button onclick="fillExample('Count of tickets by customer segment')">Tickets by segment</button>
                        </div>
                        <div style="font-size:13px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:0.3px;margin-bottom:6px;">Maintenance</div>
                        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
                            <button onclick="fillExample('How many machines have failed?')">Machine failures</button>
                            <button onclick="fillExample('Average tool wear for failed vs non-failed machines')">Tool wear analysis</button>
                            <button onclick="fillExample('Which failure type is most common?')">Common failures</button>
                        </div>
                        <div style="font-size:13px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:0.3px;margin-bottom:6px;">Sales</div>
                        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
                            <button onclick="fillExample('Total sales by category')">Sales by category</button>
                            <button onclick="fillExample('Top 10 customers by total sales')">Top customers</button>
                            <button onclick="fillExample('Sales by category and region')">Sales by region</button>
                        </div>
                        <div style="font-size:13px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:0.3px;margin-bottom:6px;">Cross-Domain</div>
                        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
                            <button onclick="fillExample('Compare total sales by region against ticket volume by region')">Sales vs tickets by region</button>
                            <button onclick="fillExample('How does employee attrition relate to ticket volume and resolution time?')">Attrition &amp; tickets</button>
                            <button onclick="fillExample('Give me a unified enterprise overview across all departments')">Enterprise overview</button>
                        </div>
                    </div>
                </div>
            </div>
            <div class="input-area">
                <div class="input-row">
                    <input type="text" id="query" placeholder="Ask a question..." autofocus>
                    <button onclick="sendQuery()">Send</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const empty = document.getElementById('empty');
        const input = document.getElementById('query');
        const status = document.getElementById('status');
        let chartInstances = {};
        let isDark = localStorage.getItem('theme') === 'dark';

        if (isDark) document.body.classList.add('dark');
        updateThemeIcons();

        function toggleTheme() {
            isDark = !isDark;
            document.body.classList.toggle('dark', isDark);
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
            updateThemeIcons();
        }
        function updateThemeIcons() {
            const icon = isDark ? 'ri-sun-line' : 'ri-moon-line';
            document.getElementById('themeIcon').className = icon;
            document.getElementById('themeIcon2').className = icon;
        }

        input.addEventListener('keydown', e => { if (e.key === 'Enter') sendQuery(); });

        function fillExample(text) {
            input.value = text;
            sendQuery();
        }

        function addMessage(content, cls) {
            empty.style.display = 'none';
            const div = document.createElement('div');
            div.className = 'message ' + cls;
            div.innerHTML = content;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        function renderDashboard(container, dash) {
            if (!dash || !dash.charts || dash.charts.length === 0) return;
            const dashDiv = document.createElement('div');
            dashDiv.className = 'dashboard';
            let html = '<div class="dash-header"><h3>Dashboard</h3><span class="row-count">' + (dash.row_count || 0) + ' records</span></div>';
            if (dash.kpis && dash.kpis.length) {
                html += '<div class="kpi-row">';
                dash.kpis.forEach(k => { html += '<div class="kpi-card"><div class="kpi-label">' + k.label + '</div><div class="kpi-value">' + k.value + '</div></div>'; });
                html += '</div>';
            }
            if (dash.charts && dash.charts.length) {
                html += '<div class="chart-grid">';
                dash.charts.forEach((c, i) => { html += '<div class="chart-box"><h4>' + c.title + '</h4><img src="data:image/png;base64,' + c.img + '" alt="' + c.title + '"></div>'; });
                html += '</div>';
            }
            if (dash.insights) {
                html += '<div class="insight-box">';
                if (dash.insights.summary) html += '<h4>Analysis</h4><p>' + dash.insights.summary + '</p>';
                if (dash.insights.pain_points && dash.insights.pain_points.length) html += '<h4 style="margin-top:12px;color:var(--red);">Pain Points</h4><ul>' + dash.insights.pain_points.map(p => '<li>' + p + '</li>').join('') + '</ul>';
                if (dash.insights.recommendations && dash.insights.recommendations.length) html += '<h4 style="margin-top:12px;color:var(--green);">Recommendations</h4><ul>' + dash.insights.recommendations.map(r => '<li>' + r + '</li>').join('') + '</ul>';
                html += '</div>';
            }
            dashDiv.innerHTML = html;
            container.appendChild(dashDiv);
        }

        function addFeedback(container, question) {
            const fb = document.createElement('div');
            fb.style.cssText = 'display:flex;gap:8px;margin-top:10px;align-items:center;';
            fb.innerHTML = '<span style="font-size:13px;color:var(--text3);">Helpful?</span>' +
                '<button onclick="sendFeedback(\\'' + question.replace(/'/g, "\\'") + '\\',1)" style="background:var(--hover);border:1px solid var(--border);border-radius:6px;padding:6px 14px;cursor:pointer;font-size:13px;color:var(--text2);">Yes</button>' +
                '<button onclick="sendFeedback(\\'' + question.replace(/'/g, "\\'") + '\\',0)" style="background:var(--hover);border:1px solid var(--border);border-radius:6px;padding:6px 14px;cursor:pointer;font-size:13px;color:var(--text2);">No</button>';
            container.appendChild(fb);
        }

        function sendFeedback(question, score) {
            fetch('/feedback', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question,score}) })
            .then(r=>r.json()).then(d=>{});
        }

        function sendQuery() {
            const q = input.value.trim();
            if (!q) return;
            addMessage('<div style="opacity:0.85">' + q + '</div>', 'user');
            input.value = '';
            status.textContent = 'Processing...';
            const loading = document.createElement('div');
            loading.className = 'message assistant loading';
            loading.innerHTML = '<div class="label-row"><div class="avatar">F4</div><span class="name">FATAL4</span></div><div style="color:var(--text3);font-size:15px">Thinking...</div>';
            chat.appendChild(loading);
            chat.scrollTop = chat.scrollHeight;
            fetch('/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q, user: 'user', role: 'analyst' }) })
            .then(r => r.json())
            .then(data => {
                loading.remove();
                status.textContent = 'Ready';
                if (data.error) { addMessage('<div class="label-row"><div class="avatar">F4</div><span class="name">FATAL4</span></div><div class="error">' + data.error + '</div>', 'assistant'); return; }
                let html = '<div class="label-row"><div class="avatar">F4</div><span class="name">FATAL4</span></div>';
                if (data.sql) html += '<div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#818cf8;margin-top:6px;margin-bottom:4px;"><i class="ri-terminal-box-line"></i> SQL Query</div><div class="sql-block">' + data.sql + '</div>';
                html += '<pre>' + data.answer + '</pre>';
                if (data.retrieved_sources && data.retrieved_sources.length) html += '<div class="source-tags">' + data.retrieved_sources.map(s => '<span><i class="ri-file-text-line" style="font-size:11px;margin-right:3px"></i>' + s + '</span>').join('') + '</div>';
                if (data.cached) html += '<div style="margin-top:6px;font-size:12px;color:var(--green);"><i class="ri-database-2-line"></i> Answered from cache (match: ' + (data.match_score * 100).toFixed(0) + '%)</div>';
                const msgDiv = addMessage(html, 'assistant');
                if (data.dashboard) renderDashboard(msgDiv, data.dashboard);
                addFeedback(msgDiv, q);
            })
            .catch(err => { loading.remove(); status.textContent = 'Error'; addMessage('<div class="label-row"><div class="avatar">F4</div><span class="name">FATAL4</span></div><div class="error">Network Error: ' + err.message + '</div>', 'assistant'); });
        }
    </script>
</body>
</html>
"""


def is_database_query(question):
    q = question.lower()
    return any(kw in q for kw in DB_KEYWORDS)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/schema")
def schema():
    return jsonify({"schema": get_schema()})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    user = data.get("user", "anonymous")
    role = data.get("role", "analyst")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not rbac.check_access(user, role, "query"):
        return jsonify({"error": "Insufficient permissions"}), 403

    cached = find_similar_question(question, threshold=0.72)
    if cached:
        from learn import log_query
        log_query(question, user, role, cached["sql"], cached["answer"], None, ["cache"], success=True)
        return jsonify(convert_for_json({
            "answer": cached["answer"],
            "sql": cached["sql"],
            "retrieved_sources": ["cache"],
            "dashboard": None,
            "cached": True,
            "match_score": cached["score"],
        }))

    try:
        answer = None
        sql = None
        retrieved_sources = []
        contexts = retrieve_context(question)
        if contexts:
            retrieved_sources = list(set(c["source"] for c in contexts))

        sql = None
        sql_result_text = None

        query_results, query_errors, sql_generated = query_all_domains(question)
        if query_results:
            sql = sql_generated
            part_summaries = []
            result_parts = []
            for label, df in query_results:
                masked_df = masker.mask_sql_result(df.copy())
                part_summaries.append(f"  - {label}: {len(df)} rows")
                text = masked_df.head(5).to_string(index=False)
                result_parts.append(f"{label} ({len(df)} rows)\n{text}")
            sql_result_text = "\n\n".join(result_parts)
            sql_summary = "\n".join(part_summaries)
        if query_errors:
            error_text = "\n".join(query_errors)
            if sql_result_text:
                sql_result_text += f"\n\n-- Notes:\n{error_text}"

        dashboard = None
        if query_results:
            best_df = max(query_results, key=lambda x: len(x[1]))[1]
            best_df = masker.mask_sql_result(best_df.copy())
            dashboard = build_dashboard(question, best_df, sql)
        if dashboard is None and contexts:
            dummy_df = pd.DataFrame({"result": ["Data from documents"]})
            dashboard = build_dashboard(question, dummy_df, sql or "RAG query")

        rag_context = "\n\n".join(
            f"[{c['source']} #{c['doc_id']}]: {c['content'][:500]}"
            for c in contexts[:6]
        ) if contexts else ""

        cross_ref_info = get_cross_reference_summary()

        has_data = bool(rag_context) or bool(sql_result_text)
        if has_data:
            source_summary_parts = []
            if query_results:
                srcs = list(set(lbl.split("]")[0].strip("[- ") for lbl, _ in query_results))
                source_summary_parts.append(f"SQL data from: {', '.join(srcs)}")
            if contexts:
                srcs = list(set(c["source"] for c in contexts))
                source_summary_parts.append(f"Document context from: {', '.join(srcs)}")
            source_summary = " | ".join(source_summary_parts)

            prompt = f"""You have data from ALL 5 enterprise domains — HR employees (1470 records), IT support tickets (100k records), machine maintenance (10k records), sales orders (9800 records), and emails (517k records) — all interconnected in one database.

DATA SOURCES USED: {source_summary}

ENTITY CROSS-REFERENCE (how entities link across domains):
{cross_ref_info}

QUESTION: {question}

DATABASE RESULTS (all domains):
{sql_summary if query_results else "No database results."}

{sql_result_text if query_results else ""}

DOCUMENT RESULTS:
{f"Found {len(contexts[:6])} relevant documents across HR, tickets, maintenance, sales, and email domains." if contexts else "No document results."}

{rag_context}

Analyze the data across ALL domains and find interconnections between them. For example: link HR attrition to ticket volumes, machine failures to overtime patterns, sales performance to support quality, email patterns about escalations, etc. Use specific numbers. Cite which domain each data point comes from. Answer in 3-5 sentences."""

            answer = ask_llm([
                {"role": "system", "content": "You are a data analyst with access to ALL 5 enterprise data domains (HR, tickets, maintenance, sales, emails). Cross-reference data across ALL domains — HR employees with support tickets, machine failures with overtime, sales with ticket resolution, email sentiment with attrition, etc. Find meaningful connections between ALL datasets. Use specific numbers and cite which domain each data point comes from. Answer comprehensively using actual data."},
                {"role": "user", "content": prompt}
            ])
        else:
            try:
                answer = ask_llm([
                    {"role": "system", "content": "You are an enterprise data assistant. Answer from your general knowledge about HR, support tickets, machine maintenance, sales, and emails."},
                    {"role": "user", "content": question}
                ])
            except Exception:
                answer = ("Please ask a question about employees, HR attrition, support tickets, "
                          "machine maintenance, sales, or email communications.")

        return jsonify(convert_for_json({
            "answer": answer,
            "sql": sql,
            "retrieved_sources": retrieved_sources,
            "dashboard": dashboard,
        }))

    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import webbrowser
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Enterprise Intelligence Engine running on http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
