# Step1: Setup Ollama with Medgemma tool
import ollama

def query_medgemma(prompt: str) -> str:
    """
    Calls MedGemma model with a therapist personality profile.
    Returns responses as an empathic mental health professional.
    """
    system_prompt = """You are Dr. Emily Hartman, a warm and experienced clinical psychologist. 
    Respond to patients with:

    1. Emotional attunement ("I can sense how difficult this must be...")
    2. Gentle normalization ("Many people feel this way when...")
    3. Practical guidance ("What sometimes helps is...")
    4. Strengths-focused support ("I notice how you're...")

    Key principles:
    - Never use brackets or labels
    - Blend elements seamlessly
    - Vary sentence structure
    - Use natural transitions
    - Mirror the user's language level
    - Always keep the conversation going by asking open ended questions to dive into the root cause of patients problem
    """
    
    try:
        response = ollama.chat(
            model='alibayram/medgemma:4b',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={
                'num_predict': 350,
                'temperature': 0.7,
                'top_p': 0.9
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        return "I'm having technical difficulties, but I want you to know your feelings matter. Please try again shortly."


# Step2: Local Emergency Alarm (no Twilio needed)
import os
import sys
import datetime
import threading

EMERGENCY_LOG = "emergency_log.txt"

def _beep():
    """Play alert sound cross-platform."""
    if sys.platform == "win32":
        # Windows — use winsound (built-in, no install needed)
        import winsound
        for _ in range(6):
            winsound.Beep(1000, 600)   # 1000 Hz tone, 600ms each
    elif sys.platform == "darwin":
        # Mac — use afplay with system alert sound
        for _ in range(6):
            os.system("afplay /System/Library/Sounds/Sosumi.aiff")
    else:
        # Linux — try paplay, fallback to terminal bell
        sound_played = False
        linux_sounds = [
            "/usr/share/sounds/ubuntu/stereo/dialog-warning.ogg",
            "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga",
        ]
        for sound in linux_sounds:
            if os.path.exists(sound):
                os.system(f"paplay {sound} 2>/dev/null")
                sound_played = True
                break
        if not sound_played:
            # Terminal bell fallback
            for _ in range(6):
                print("\a", end="", flush=True)
                import time; time.sleep(0.5)


def _show_popup():
    """Show a popup alert window cross-platform."""
    msg = "🚨 MENTAL HEALTH EMERGENCY DETECTED\n\nA user may be in crisis.\nCheck emergency_log.txt immediately."

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, "⚠️ EMERGENCY ALERT", 0x00000010)

    elif sys.platform == "darwin":
        safe_msg = msg.replace("'", "\\'").replace("\n", "\\n")
        os.system(f"osascript -e 'display dialog \"{safe_msg}\" with title \"EMERGENCY\" buttons {{\"OK\"}} with icon stop'")

    else:
        # Linux — try zenity, then kdialog, then xmessage
        safe_msg = msg.replace('"', '\\"')
        if os.system("which zenity > /dev/null 2>&1") == 0:
            os.system(f'zenity --error --title="EMERGENCY ALERT" --text="{safe_msg}" 2>/dev/null')
        elif os.system("which kdialog > /dev/null 2>&1") == 0:
            os.system(f'kdialog --error "{safe_msg}" --title "EMERGENCY ALERT" 2>/dev/null')
        elif os.system("which xmessage > /dev/null 2>&1") == 0:
            os.system(f'xmessage -center "{safe_msg}" 2>/dev/null')
        else:
            print(f"\n{'='*50}\n{msg}\n{'='*50}\n")


def _log_emergency():
    """Write emergency event to log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"\n{'='*60}\n"
        f"  🚨 EMERGENCY TRIGGERED\n"
        f"  Time    : {timestamp}\n"
        f"  Action  : User expressed suicidal ideation or crisis\n"
        f"  Status  : Alarm sounded + popup shown\n"
        f"{'='*60}\n"
    )
    with open(EMERGENCY_LOG, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # Also print to terminal
    print(log_entry)


def call_emergency():
    """
    Local emergency alert — no Twilio needed.
    Triggers simultaneously:
      1. Writes a timestamped entry to emergency_log.txt
      2. Plays a loud beep/alert sound
      3. Shows a popup dialog window
    All three run in parallel so nothing blocks the response.
    """
    # Log immediately (synchronous — always runs)
    _log_emergency()

    # Sound + popup in background threads (non-blocking)
    threading.Thread(target=_beep,        daemon=True).start()
    threading.Thread(target=_show_popup,  daemon=True).start()


# Step3: Location tool
