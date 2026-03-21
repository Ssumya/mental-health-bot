# frontend.py — SafeSpace AI Mental Health Therapist
# Streamlit frontend with login, register, chat history, JWT auth, mood tracker

import streamlit as st
import requests
import os
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SafeSpace · AI Therapist",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.stApp { background-color: #F5F3EF; }

[data-testid="stSidebar"] { background-color: #2C2925; }
[data-testid="stSidebar"] * { color: #E8E4DE !important; }
[data-testid="stSidebar"] .stButton button {
    background: #7A9E7E; color: white !important;
    border: none; border-radius: 8px; width: 100%; font-weight: 500;
}
[data-testid="stSidebar"] .stButton button:hover { background: #6A8E6E; }

[data-testid="stChatMessage"] {
    background: white; border-radius: 12px;
    margin-bottom: 8px; border: 1px solid #EAE6E0; padding: 4px;
}

.tool-badge {
    display: inline-block; background: #EAF2EB; color: #4A7A50;
    border: 1px solid #B8D4BB; border-radius: 20px;
    padding: 2px 10px; font-size: 12px; margin-top: 6px;
}
.emergency-badge { background: #FEF0ED; color: #C0392B; border: 1px solid #F5C0B0; }

.history-item {
    background: #3A3632; border-radius: 8px; padding: 8px 10px;
    margin-bottom: 4px; font-size: 13px; color: #B8B4AE !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.history-item.user { color: #A8C8AB !important; }

.welcome-card {
    background: white; border: 1px solid #EAE6E0; border-radius: 16px;
    padding: 2.5rem; text-align: center; max-width: 520px;
    margin: 3rem auto; box-shadow: 0 4px 24px rgba(44,41,37,0.07);
}

.mood-card {
    background: white; border: 1px solid #EAE6E0; border-radius: 16px;
    padding: 1.5rem; margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(44,41,37,0.06);
}

.mood-log-item {
    background: #F8F6F2; border-radius: 10px; padding: 0.75rem 1rem;
    margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ──────────────────────────────────────────────────
for key, default in {
    "token":      None,
    "user":       None,
    "messages":   [],
    "page":       "chat",  # "chat" | "mood"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── API helpers ─────────────────────────────────────────────────────────────
def api_post(endpoint: str, payload: dict, token: str = None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.post(f"{BACKEND}{endpoint}", json=payload, headers=headers, timeout=120)
        return r.status_code, r.json()
    except requests.exceptions.ConnectionError:
        return 503, {"detail": "Cannot connect to backend. Is main.py running?"}
    except Exception as e:
        return 500, {"detail": str(e)}


def api_get(endpoint: str, token: str) -> tuple[int, dict]:
    try:
        r = requests.get(
            f"{BACKEND}{endpoint}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        return r.status_code, r.json()
    except requests.exceptions.ConnectionError:
        return 503, {"detail": "Cannot connect to backend."}
    except Exception as e:
        return 500, {"detail": str(e)}


def do_logout():
    st.session_state.token    = None
    st.session_state.user     = None
    st.session_state.messages = []
    st.session_state.page     = "chat"
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  NOT LOGGED IN → show auth screen
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.token:

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 2rem 0 1rem;'>
            <span style='font-size:2.8rem;'>🌿</span>
            <h1 style='font-family:Georgia,serif; font-weight:400;
                       color:#2C2925; font-size:2.2rem; margin:0.3rem 0;'>
                SafeSpace
            </h1>
            <p style='color:#8C8680; font-size:0.9rem; letter-spacing:0.08em;'>
                AI MENTAL HEALTH SUPPORT
            </p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_reg = st.tabs(["🔑  Sign In", "✨  Create Account"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", key="li_user", placeholder="your username")
            password = st.text_input("Password", key="li_pass",
                                     placeholder="••••••••", type="password")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Sign In", key="btn_login", use_container_width=True):
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    with st.spinner("Signing in…"):
                        status, data = api_post("/auth/login",
                                                {"username": username, "password": password})
                    if status == 200:
                        st.session_state.token = data["token"]
                        st.session_state.user  = data["user"]
                        _, hist = api_get("/history", data["token"])
                        for m in hist.get("history", []):
                            st.session_state.messages.append({
                                "role":        m["role"],
                                "content":     m["message"],
                                "tool_called": m.get("tool_called", "None"),
                            })
                        st.rerun()
                    else:
                        st.error(data.get("detail", "Login failed."))

        with tab_reg:
            st.markdown("<br>", unsafe_allow_html=True)
            r_user  = st.text_input("Username", key="reg_user",  placeholder="choose a username")
            r_email = st.text_input("Email",    key="reg_email", placeholder="you@example.com")
            r_pass  = st.text_input("Password", key="reg_pass",
                                    placeholder="min 6 characters", type="password")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Create Account", key="btn_reg", use_container_width=True):
                if not r_user or not r_email or not r_pass:
                    st.error("Please fill in all fields.")
                elif len(r_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    with st.spinner("Creating account…"):
                        status, data = api_post("/auth/register",
                                                {"username": r_user,
                                                 "email":    r_email,
                                                 "password": r_pass})
                    if status == 200:
                        st.session_state.token = data["token"]
                        st.session_state.user  = data["user"]
                        st.success("Account created! Welcome 🌿")
                        st.rerun()
                    else:
                        st.error(data.get("detail", "Registration failed."))

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGED IN → main app
# ══════════════════════════════════════════════════════════════════════════════
user     = st.session_state.user
token    = st.session_state.token
username = user.get("username", "User")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:0.5rem 0 1rem;'>
        <div style='font-size:1.5rem; font-family:Georgia,serif;
                    font-weight:300; color:#E8E4DE;'>🌿 SafeSpace</div>
        <div style='margin-top:0.75rem; background:#3A3632; border-radius:10px;
                    padding:0.6rem 0.85rem; display:flex; align-items:center; gap:0.6rem;'>
            <span style='background:#7A9E7E; border-radius:50%; width:30px; height:30px;
                         display:inline-flex; align-items:center; justify-content:center;
                         font-weight:600; font-size:0.9rem; color:white; flex-shrink:0;'>
                {username[0].upper()}
            </span>
            <span style='font-size:0.875rem; color:#E8E4DE;'>{username}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("<p style='font-size:0.72rem; letter-spacing:0.1em; color:#6A6460;'>NAVIGATION</p>",
                unsafe_allow_html=True)

    if st.button("💬  Chat", key="nav_chat", use_container_width=True):
        st.session_state.page = "chat"
        st.rerun()

    if st.button("😊  Mood Tracker", key="nav_mood", use_container_width=True):
        st.session_state.page = "mood"
        st.rerun()

    st.divider()

    if st.button("🚪  Log Out", key="logout"):
        do_logout()

    st.divider()

    # Chat history in sidebar
    st.markdown("<p style='font-size:0.72rem; letter-spacing:0.1em; color:#6A6460;'>CHAT HISTORY</p>",
                unsafe_allow_html=True)

    if st.session_state.messages:
        for m in reversed(st.session_state.messages[-20:]):
            role_icon = "🧑" if m["role"] == "user" else "🤖"
            preview   = m["content"][:50] + ("…" if len(m["content"]) > 50 else "")
            css_class = "user" if m["role"] == "user" else ""
            st.markdown(
                f"<div class='history-item {css_class}'>{role_icon} {preview}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<p style='color:#4A4540; font-size:0.8rem;'>Your conversations will appear here.</p>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MOOD TRACKER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "mood":

    st.markdown("""
    <div style='display:flex; align-items:center; gap:0.6rem;
                padding:0.75rem 0; border-bottom:1px solid #E5E0D8; margin-bottom:1.5rem;'>
        <span style='font-family:Georgia,serif; font-size:1.3rem; color:#2C2925;'>
            😊 Mood Tracker
        </span>
        <span style='margin-left:auto; font-size:0.75rem; color:#8C8680;'>
            Track your daily wellbeing
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Log today's mood ────────────────────────────────────────────────────
    st.markdown("### How are you feeling today?")

    MOODS = {
        "😄 Great":     {"score": 5, "color": "#52B788", "emoji": "😄"},
        "🙂 Good":      {"score": 4, "color": "#74C69D", "emoji": "🙂"},
        "😐 Neutral":   {"score": 3, "color": "#F4A261", "emoji": "😐"},
        "😔 Low":       {"score": 2, "color": "#E07A5F", "emoji": "😔"},
        "😢 Very Low":  {"score": 1, "color": "#C1121F", "emoji": "😢"},
    }

    cols = st.columns(5)
    selected_mood = None
    for i, (mood_label, mood_data) in enumerate(MOODS.items()):
        with cols[i]:
            if st.button(
                f"{mood_data['emoji']}\n{mood_label.split(' ', 1)[1]}",
                key=f"mood_{i}",
                use_container_width=True
            ):
                selected_mood = mood_label

    note = st.text_area(
        "Add a note (optional)",
        placeholder="What's making you feel this way?",
        key="mood_note",
        height=80
    )

    if selected_mood:
        with st.spinner("Saving mood..."):
            status, data = api_post(
                "/mood/log",
                {
                    "mood":  selected_mood,
                    "score": MOODS[selected_mood]["score"],
                    "note":  note or ""
                },
                token=token
            )
        if status == 200:
            st.success(f"Mood logged! {selected_mood} ✅")
        else:
            st.error("Could not save mood. Try again.")

    st.divider()

    # ── Mood history ────────────────────────────────────────────────────────
    st.markdown("### Your Mood History")

    status, data = api_get("/mood/history", token)

    if status == 200 and data.get("logs"):
        logs = data["logs"]

        # Chart
        import pandas as pd
        df = pd.DataFrame(logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        st.line_chart(
            df.set_index("timestamp")["score"],
            color="#7A9E7E",
            height=200
        )

        st.markdown("### Recent Logs")

        # Show logs
        for log in reversed(logs[-10:]):
            ts = datetime.fromisoformat(log["timestamp"]).strftime("%b %d · %I:%M %p")
            mood_emoji = log["mood"].split(" ")[0]
            mood_name  = log["mood"].split(" ", 1)[1] if " " in log["mood"] else log["mood"]
            note_text  = f" — {log['note']}" if log.get("note") else ""

            st.markdown(f"""
            <div class='mood-log-item'>
                <span style='font-size:1.5rem;'>{mood_emoji}</span>
                <div>
                    <div style='font-weight:500; color:#2C2925;'>{mood_name}</div>
                    <div style='font-size:0.8rem; color:#8C8680;'>{ts}{note_text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No mood logs yet. Log your first mood above! 😊")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CHAT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='display:flex; align-items:center; gap:0.6rem;
            padding:0.75rem 0; border-bottom:1px solid #E5E0D8; margin-bottom:1rem;'>
    <span style='width:9px; height:9px; border-radius:50%;
                 background:#52B788; display:inline-block;
                 box-shadow:0 0 0 3px rgba(82,183,136,0.2);'></span>
    <span style='font-family:Georgia,serif; font-size:1.1rem; color:#2C2925;'>
        Dr. Emily Hartman &middot; AI Therapist
    </span>
    <span style='margin-left:auto; font-size:0.75rem; color:#8C8680;'>
        Session · {username}
    </span>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class='welcome-card'>
        <div style='font-size:2.5rem; margin-bottom:0.75rem;'>🌿</div>
        <h2 style='font-family:Georgia,serif; font-weight:400;
                   color:#2C2925; font-size:1.6rem; margin-bottom:0.5rem;'>
            Hello, I'm here for you
        </h2>
        <p style='color:#8C8680; font-size:0.9rem; line-height:1.7;'>
            This is a safe, supportive space. Share what's on your mind —
            your thoughts, feelings, or anything you'd like to talk about.
        </p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        tool = msg.get("tool_called", "None")
        if tool and tool != "None":
            badge_class = "emergency-badge" if "emergency" in tool else "tool-badge"
            label = tool.replace("_", " ")
            icon  = "🚨" if "emergency" in tool else "🔧"
            st.markdown(
                f"<span class='{badge_class} tool-badge'>{icon} {label}</span>",
                unsafe_allow_html=True,
            )

user_input = st.chat_input("Share what's on your mind…")

if user_input:
    st.session_state.messages.append({
        "role": "user", "content": user_input, "tool_called": "None"
    })
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Dr. Emily is thinking…"):
            status, data = api_post("/ask", {"message": user_input}, token=token)

        if status == 401:
            st.error("Session expired. Please log in again.")
            do_logout()
        elif status != 200:
            st.error(data.get("detail", "Something went wrong. Please try again."))
        else:
            response    = data.get("response", "")
            tool_called = data.get("tool_called", "None")
            st.write(response)

            if tool_called and tool_called != "None":
                badge_class = "emergency-badge" if "emergency" in tool_called else "tool-badge"
                label = tool_called.replace("_", " ")
                icon  = "🚨" if "emergency" in tool_called else "🔧"
                st.markdown(
                    f"<span class='{badge_class} tool-badge'>{icon} {label}</span>",
                    unsafe_allow_html=True,
                )
                if "emergency" in tool_called:
                    st.error(
                        "🚨 **Emergency alert triggered.** Please reach out to someone you trust "
                        "or call your local emergency number immediately.",
                        icon="🚨",
                    )

            st.session_state.messages.append({
                "role": "assistant", "content": response, "tool_called": tool_called,
            })

st.markdown("""
<div style='text-align:center; color:#A8A4A0; font-size:0.72rem; margin-top:2rem;
            padding-top:1rem; border-top:1px solid #E5E0D8;'>
    Not a substitute for professional care &nbsp;·&nbsp;
    If you're in crisis, call your local emergency services
</div>
""", unsafe_allow_html=True)
