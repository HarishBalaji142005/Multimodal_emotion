# ── Suppress all warnings before anything else ───────────────────
import os
import logging

os.environ["TF_ENABLE_ONEDNN_OPTS"]          = "0"   # suppress oneDNN warning
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1" # suppress transformers warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]           = "3"   # suppress TF C++ logs (0=all,3=none)

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("deepface").setLevel(logging.ERROR)

# ── Standard Library ─────────────────────────────────────────────
import time, warnings, tempfile, io, wave, struct
from collections import Counter
from datetime import datetime
warnings.filterwarnings("ignore")

# ── Core ──────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
from PIL import Image

# ── Optional imports with graceful fallback ───────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from deepface import DeepFace
    DF_OK = True
except Exception:
    DF_OK = False

try:
    import librosa
    import soundfile as sf
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False

try:
    from textblob import TextBlob
    TB_OK = True
except ImportError:
    TB_OK = False

try:
    from transformers import pipeline as hf_pipeline
    HF_OK = True
except ImportError:
    HF_OK = False

try:
    import pandas as pd
    PD_OK = True
except ImportError:
    PD_OK = False

# ── Microphone recorder (streamlit-audiorecorder) ─────────────────
try:
    from audiorecorder import audiorecorder
    MIC_OK = True
except ImportError:
    MIC_OK = False

# ── sounddevice fallback for mic ──────────────────────────────────
try:
    import sounddevice as sd
    SD_OK = True
except ImportError:
    SD_OK = False

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MultiModal Emotion AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;700&family=Syne:wght@700;800&display=swap');

:root{--bg:#05090d;--surface:#0b1520;--surf2:#0f1e2e;--border:#1a2f45;
      --accent:#00e5ff;--green:#00ff88;--pink:#ff3f80;--purple:#9f6fff;
      --amber:#ffb830;--text:#c5d8e8;--dim:#4a6070;--red:#ff3f80;}

html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);}
.stApp{background:var(--bg);}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
.block-container{padding-top:1.4rem!important;}

[data-testid="metric-container"]{background:var(--surf2);border:1px solid var(--border);border-radius:10px;padding:14px!important;}
[data-testid="stMetricValue"]{font-family:'Share Tech Mono',monospace!important;color:var(--accent)!important;font-size:24px!important;}
[data-testid="stMetricLabel"]{color:var(--dim)!important;font-family:'Share Tech Mono',monospace!important;font-size:10px!important;letter-spacing:2px!important;text-transform:uppercase!important;}

.stButton>button{background:var(--accent);color:#000;font-family:'Syne',sans-serif;font-weight:800;
  border:none;border-radius:8px;padding:11px 20px;font-size:14px;width:100%;transition:all .2s;}
.stButton>button:hover{background:#00b8cc;transform:translateY(-1px);}

[data-testid="stTabs"] button{font-family:'Syne',sans-serif;font-weight:700;
  color:var(--dim)!important;border-bottom:2px solid transparent!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;}

.stTextArea textarea,.stTextInput input{background:var(--surf2)!important;border:1px solid var(--border)!important;
  color:var(--text)!important;border-radius:8px!important;}
[data-testid="stFileUploader"]{background:var(--surf2);border:1px dashed var(--border);border-radius:10px;padding:12px;}
hr{border-color:var(--border)!important;margin:1rem 0!important;}

.mcard{background:var(--surf2);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:12px;}
.mcard-title{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:3px;color:var(--dim);margin-bottom:12px;text-transform:uppercase;}

.emo-big{background:var(--surf2);border:1px solid var(--border);border-radius:14px;padding:24px;text-align:center;margin-bottom:14px;}
.emo-emoji{font-size:80px;line-height:1;display:block;margin-bottom:8px;}
.emo-name{font-family:'Syne',sans-serif;font-size:34px;font-weight:800;letter-spacing:-1px;}
.emo-conf{font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--dim);margin-top:4px;}

.badge{display:inline-block;font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:2px;
  padding:3px 10px;border-radius:20px;margin:2px;font-weight:700;}
.badge-face  {background:rgba(0,229,255,.12);color:#00e5ff;border:1px solid rgba(0,229,255,.3);}
.badge-speech{background:rgba(0,255,136,.12);color:#00ff88;border:1px solid rgba(0,255,136,.3);}
.badge-text  {background:rgba(159,111,255,.12);color:#9f6fff;border:1px solid rgba(159,111,255,.3);}
.badge-fused {background:rgba(255,184,48,.12);color:#ffb830;border:1px solid rgba(255,184,48,.3);}
.badge-mic   {background:rgba(255,63,128,.12);color:#ff3f80;border:1px solid rgba(255,63,128,.3);}

.bar-row{display:flex;align-items:center;gap:8px;margin-bottom:8px;}
.bar-label{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);width:72px;text-align:right;letter-spacing:1px;}
.bar-track{flex:1;background:rgba(255,255,255,.05);border-radius:4px;height:10px;overflow:hidden;}
.bar-fill{height:100%;border-radius:4px;transition:width .4s ease;}
.bar-pct{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);width:34px;text-align:right;}

.logbox{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px;
  font-family:'Share Tech Mono',monospace;font-size:11px;max-height:180px;overflow-y:auto;line-height:1.9;}
.log-ts{color:#00e5ff;margin-right:8px;}
.log-face{color:#00e5ff;}.log-sph{color:#00ff88;}.log-txt{color:#9f6fff;}
.log-fus{color:#ffb830;}.log-mic{color:#ff3f80;}

.app-header{background:linear-gradient(135deg,var(--surface),var(--surf2));
  border:1px solid var(--border);border-radius:14px;padding:24px 28px;margin-bottom:20px;}
.header-tag{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--accent);letter-spacing:4px;margin-bottom:6px;}
.header-title{font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:#fff;letter-spacing:-1px;line-height:1;}
.header-title span{color:var(--accent);}

.pipeline{display:flex;align-items:center;background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:14px 18px;overflow-x:auto;flex-wrap:wrap;gap:4px;margin-bottom:4px;}
.pipe-node{text-align:center;min-width:65px;font-family:'Share Tech Mono',monospace;font-size:10px;}
.pipe-icon{font-size:20px;display:block;margin-bottom:3px;}
.pipe-label{color:var(--dim);letter-spacing:1px;line-height:1.4;}
.pipe-arrow{color:var(--border);font-size:16px;margin:0 4px;}

.insight-card{background:linear-gradient(135deg,rgba(0,229,255,.06),rgba(159,111,255,.06));
  border:1px solid var(--border);border-radius:12px;padding:16px;margin-top:10px;}
.insight-title{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:3px;color:#ffb830;margin-bottom:8px;}
.insight-body{font-size:13px;color:var(--text);line-height:1.7;}

/* MIC recording card */
.mic-card{background:var(--surf2);border:2px solid var(--border);border-radius:14px;padding:24px;text-align:center;}
.mic-card.active{border-color:#ff3f80;box-shadow:0 0 20px rgba(255,63,128,.2);}
.mic-icon{font-size:56px;display:block;margin-bottom:10px;}
.mic-status{font-family:'Share Tech Mono',monospace;font-size:12px;letter-spacing:2px;}

/* Waveform bar animation */
.wave-bars{display:flex;align-items:center;justify-content:center;gap:3px;height:40px;margin:10px 0;}
.wave-bar{width:4px;border-radius:2px;background:#ff3f80;animation:wave 0.8s ease-in-out infinite;}
.wave-bar:nth-child(1){animation-delay:0s;height:10px;}
.wave-bar:nth-child(2){animation-delay:0.1s;height:20px;}
.wave-bar:nth-child(3){animation-delay:0.2s;height:35px;}
.wave-bar:nth-child(4){animation-delay:0.3s;height:25px;}
.wave-bar:nth-child(5){animation-delay:0.4s;height:38px;}
.wave-bar:nth-child(6){animation-delay:0.3s;height:22px;}
.wave-bar:nth-child(7){animation-delay:0.2s;height:32px;}
.wave-bar:nth-child(8){animation-delay:0.1s;height:15px;}
.wave-bar:nth-child(9){animation-delay:0s;height:8px;}
@keyframes wave{0%,100%{transform:scaleY(1);}50%{transform:scaleY(0.3);}}

/* Rec dot */
.rec-dot{width:10px;height:10px;background:#ff3f80;border-radius:50%;display:inline-block;
  margin-right:6px;animation:blink 1s infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0;}}

/* Method toggle */
.method-box{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:14px;}
.method-label{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);letter-spacing:2px;margin-bottom:10px;}

/* Audio waveform display */
.audio-info{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:12px;font-family:Share Tech Mono,monospace;font-size:11px;line-height:2;}

.cam-ph{background:#000;border:1px solid var(--border);border-radius:10px;height:300px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;}
.cam-ph-icon{font-size:52px;}
.cam-ph-text{font-family:'Share Tech Mono',monospace;color:var(--dim);font-size:12px;letter-spacing:2px;}

::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# EMOTION CONFIG
# ─────────────────────────────────────────────────────────────────
EMOTIONS = {
    "happy":    {"emoji":"😄","color":"#FFD700","desc":"Joyful & positive"},
    "sad":      {"emoji":"😢","color":"#6ab0ff","desc":"Sorrowful & low energy"},
    "angry":    {"emoji":"😠","color":"#ff3f80","desc":"Frustrated & tense"},
    "fear":     {"emoji":"😨","color":"#bf7fff","desc":"Anxious & fearful"},
    "surprise": {"emoji":"😮","color":"#ff9f40","desc":"Astonished & alert"},
    "disgust":  {"emoji":"🤢","color":"#7fff6e","desc":"Aversion & discomfort"},
    "neutral":  {"emoji":"😐","color":"#8ba3b0","desc":"Calm & balanced"},
}
EMOTION_KEYS = list(EMOTIONS.keys())

TEXT_HAPPY    = {"happy","joy","love","excited","great","wonderful","amazing","fantastic","awesome","pleased","glad","cheerful","delighted","thrilled"}
TEXT_SAD      = {"sad","depressed","unhappy","miserable","lonely","disappointed","hurt","cry","grief","sorrow","hopeless"}
TEXT_ANGRY    = {"angry","furious","rage","mad","annoyed","irritated","frustrated","hate","bitter","hostile"}
TEXT_FEAR     = {"fear","scared","afraid","terrified","anxious","worried","nervous","panic","dread","horror"}
TEXT_SURPRISE = {"surprised","shocked","astonished","amazed","unexpected","wow","incredible"}
TEXT_DISGUST  = {"disgusted","gross","revolting","awful","terrible","horrible","nasty","sick"}

INSIGHTS = {
    "happy":    "Subject appears in a **positive emotional state**. Comfort and satisfaction are present. Ideal for creative and social tasks.",
    "sad":      "Indicators suggest **low mood or sadness**. Empathetic communication and reduced cognitive load are recommended.",
    "angry":    "Signs of **frustration or anger** detected. De-escalation and calm communication are advised.",
    "fear":     "**Anxiety or fear-related signals** present. A reassuring, structured environment may help.",
    "surprise": "**Surprise** detected — may be positive or negative. Further context needed to determine valence.",
    "disgust":  "**Aversion or discomfort** signals present. Identifying the trigger may improve engagement.",
    "neutral":  "Subject is in a **calm, neutral state** — ideal for focused tasks and decision-making.",
}

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "log":            [],
        "face_result":    None,
        "speech_result":  None,
        "text_result":    None,
        "fused_result":   None,
        "session_count":  0,
        "history":        [],
        "cam_running":    False,
        "face_frames":    0,
        "mic_recording":  False,
        "mic_audio_data": None,   # numpy array of recorded audio
        "mic_result":     None,   # emotion result from mic
        "mic_duration":   5,      # seconds to record
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────
def add_log(msg, kind="info"):
    ts  = datetime.now().strftime("%H:%M:%S")
    cls = {"face":"log-face","speech":"log-sph","text":"log-txt",
           "fused":"log-fus","mic":"log-mic"}.get(kind,"")
    st.session_state.log.insert(0,
        f"<span class='log-ts'>{ts}</span><span class='{cls}'>{msg}</span>")
    if len(st.session_state.log) > 40:
        st.session_state.log.pop()

# ─────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_face_cascade():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)

@st.cache_resource(show_spinner=False)
def warmup_deepface():
    if not DF_OK: return False
    try:
        blank = np.zeros((100,100,3), dtype=np.uint8)
        DeepFace.analyze(blank, actions=["emotion"], enforce_detection=False, silent=True)
        return True
    except Exception:
        return False

@st.cache_resource(show_spinner=False)
def get_hf_model():
    if not HF_OK: return None
    try:
        return hf_pipeline("text-classification",
                           model="j-hartmann/emotion-english-distilroberta-base",
                           top_k=None,
                           model_kwargs={"ignore_mismatched_sizes": True})
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────
# MODALITY 1 — FACE
# ─────────────────────────────────────────────────────────────────
def analyze_face_image(img_bgr):
    if not DF_OK:
        vals   = np.random.dirichlet(np.ones(7))
        scores = {e: round(float(v)*100,2) for e,v in zip(EMOTION_KEYS,vals)}
        return {"dominant":max(scores,key=scores.get),"scores":scores,"simulated":True}
    try:
        r = DeepFace.analyze(img_bgr, actions=["emotion"],
                              enforce_detection=False, silent=True)
        if isinstance(r, list): r = r[0]
        scores   = {k.lower(): float(v) for k,v in r.get("emotion",{}).items()}
        dominant = r.get("dominant_emotion","neutral").lower()
        return {"dominant":dominant,"scores":scores,"simulated":False}
    except Exception:
        return {"dominant":"neutral","scores":{e:0.0 for e in EMOTION_KEYS},"simulated":True}

def draw_face_overlay(frame, faces, dominant, scores):
    hx    = EMOTIONS.get(dominant, EMOTIONS["neutral"])["color"]
    color = (int(hx[5:7],16), int(hx[3:5],16), int(hx[1:3],16))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cs = 16
        for seg in [[(x,y+cs),(x,y),(x+cs,y)],[(x+w-cs,y),(x+w,y),(x+w,y+cs)],
                    [(x,y+h-cs),(x,y+h),(x+cs,y+h)],[(x+w-cs,y+h),(x+w,y+h),(x+w,y+h-cs)]]:
            for i in range(len(seg)-1):
                cv2.line(frame,seg[i],seg[i+1],color,3,cv2.LINE_AA)
        label = f"{dominant.upper()} {scores.get(dominant,0):.0f}%"
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
        ly = max(y-10,40)
        cv2.rectangle(frame,(x-2,ly-th-10),(x+tw+10,ly+4),color,-1)
        cv2.putText(frame,label,(x+4,ly-2),cv2.FONT_HERSHEY_SIMPLEX,0.6,(10,10,10),2,cv2.LINE_AA)
        for pt in [(x+w//2,y+int(h*.60)),(x+int(w*.27),y+int(h*.32)),
                   (x+int(w*.73),y+int(h*.32)),(x+int(w*.35),y+int(h*.82)),
                   (x+int(w*.65),y+int(h*.82))]:
            cv2.circle(frame,pt,3,(0,220,220),-1,cv2.LINE_AA)
    return frame

# ─────────────────────────────────────────────────────────────────
# AUDIO FEATURE EXTRACTION  (shared by upload + mic)
# ─────────────────────────────────────────────────────────────────
def extract_audio_features(y, sr):
    """Extract acoustic features from audio signal y at sample rate sr."""
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                       fmax=librosa.note_to_hz('C7'))
    pitch_mean = float(np.nanmean(f0[voiced_flag])) if voiced_flag.any() else 0
    pitch_std  = float(np.nanstd(f0[voiced_flag]))  if voiced_flag.any() else 0

    rms         = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std  = float(np.std(rms))

    tempo, _  = librosa.beat.beat_track(y=y, sr=sr)
    tempo     = float(tempo)
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var  = float(np.mean(np.var(mfcc, axis=1)))
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    features = {
        "Pitch Mean (Hz)":    round(pitch_mean,1),
        "Pitch Std (Hz)":     round(pitch_std,1),
        "Energy Mean":        round(energy_mean,4),
        "Tempo (BPM)":        round(tempo,1),
        "Zero Crossing Rate": round(zcr,4),
        "MFCC Variance":      round(mfcc_var,2),
        "Spectral Centroid":  round(centroid,1),
    }

    # Rule-based emotion scoring
    scores = {e: 5.0 for e in EMOTION_KEYS}
    if pitch_mean > 250:
        if energy_mean > 0.08: scores["angry"] += 35; scores["surprise"] += 15
        else:                  scores["surprise"] += 30; scores["happy"] += 15
    elif pitch_mean < 150:
        if energy_mean < 0.03: scores["sad"] += 40; scores["neutral"] += 15
        else:                  scores["neutral"] += 30
    else:
        scores["neutral"] += 20; scores["happy"] += 10

    if tempo > 120:   scores["happy"] += 15; scores["angry"] += 10
    elif tempo < 70:  scores["sad"] += 20
    if zcr > 0.15:    scores["fear"] += 20; scores["angry"] += 10
    if energy_std > 0.05: scores["angry"] += 10; scores["surprise"] += 8
    if centroid > 3000:   scores["happy"] += 12; scores["surprise"] += 8
    elif centroid < 1500: scores["sad"] += 10; scores["disgust"] += 8

    total  = sum(scores.values())
    scores = {k: round(v/total*100,2) for k,v in scores.items()}
    return {"dominant":max(scores,key=scores.get),"scores":scores,
            "features":features,"simulated":False}

def analyze_audio_file(audio_path):
    """Analyze emotion from audio file path."""
    if not AUDIO_OK:
        vals   = np.random.dirichlet(np.ones(7))
        scores = {e: round(float(v)*100,2) for e,v in zip(EMOTION_KEYS,vals)}
        return {"dominant":max(scores,key=scores.get),"scores":scores,"features":{},"simulated":True}
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        return extract_audio_features(y, sr)
    except Exception as ex:
        scores = {e2: round(100/7,2) for e2 in EMOTION_KEYS}
        return {"dominant":"neutral","scores":scores,"features":{},"simulated":True,"error":str(ex)}

def analyze_audio_array(audio_array, sr=22050):
    """Analyze emotion from numpy audio array (microphone input)."""
    if not AUDIO_OK:
        vals   = np.random.dirichlet(np.ones(7))
        scores = {e: round(float(v)*100,2) for e,v in zip(EMOTION_KEYS,vals)}
        return {"dominant":max(scores,key=scores.get),"scores":scores,"features":{},"simulated":True}
    try:
        # Convert to float32 mono
        y = audio_array.astype(np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y / (np.max(np.abs(y)) + 1e-8)  # normalize
        return extract_audio_features(y, sr)
    except Exception as ex:
        scores = {e2: round(100/7,2) for e2 in EMOTION_KEYS}
        return {"dominant":"neutral","scores":scores,"features":{},"simulated":True,"error":str(ex)}

# ─────────────────────────────────────────────────────────────────
# MICROPHONE RECORDING  (sounddevice)
# ─────────────────────────────────────────────────────────────────
def record_microphone(duration=5, sr=22050):
    """Record audio from microphone for `duration` seconds using sounddevice."""
    if not SD_OK:
        return None, "sounddevice not installed. Run: pip install sounddevice"
    try:
        audio = sd.rec(int(duration * sr), samplerate=sr,
                       channels=1, dtype='float32')
        sd.wait()   # block until done
        return audio.flatten(), None
    except Exception as e:
        return None, str(e)

def numpy_to_wav_bytes(audio_array, sr=22050):
    """Convert numpy float32 array to WAV bytes for st.audio playback."""
    pcm = (audio_array * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────────
# MODALITY 3 — TEXT
# ─────────────────────────────────────────────────────────────────
def analyze_text(text):
    if not text or not text.strip():
        return {"dominant":"neutral","scores":{e:round(100/7,2) for e in EMOTION_KEYS},
                "sentiment":0,"method":"empty","keywords":[]}

    scores = {e: 5.0 for e in EMOTION_KEYS}
    words  = text.lower().split()
    kws    = []

    hf = get_hf_model()
    if hf:
        try:
            label_map = {"joy":"happy","sadness":"sad","anger":"angry","fear":"fear",
                         "surprise":"surprise","disgust":"disgust","neutral":"neutral"}
            results   = hf(text[:512])[0]
            scores    = {e: 0.0 for e in EMOTION_KEYS}
            for r in results:
                key = label_map.get(r["label"].lower(), r["label"].lower())
                if key in scores:
                    scores[key] = round(r["score"]*100,2)
            total  = sum(scores.values()) or 1
            scores = {k: round(v/total*100,2) for k,v in scores.items()}
            pol    = TextBlob(text).sentiment.polarity if TB_OK else 0.0
            return {"dominant":max(scores,key=scores.get),"scores":scores,
                    "sentiment":pol,"method":"HuggingFace Transformer","keywords":[]}
        except Exception:
            pass

    for w in words:
        c = ''.join(ch for ch in w if ch.isalpha())
        if c in TEXT_HAPPY:    scores["happy"]    += 20; kws.append(c)
        if c in TEXT_SAD:      scores["sad"]      += 20; kws.append(c)
        if c in TEXT_ANGRY:    scores["angry"]    += 20; kws.append(c)
        if c in TEXT_FEAR:     scores["fear"]     += 20; kws.append(c)
        if c in TEXT_SURPRISE: scores["surprise"] += 20; kws.append(c)
        if c in TEXT_DISGUST:  scores["disgust"]  += 20; kws.append(c)

    pol = 0.0
    if TB_OK:
        pol = TextBlob(text).sentiment.polarity
        if pol >  0.3: scores["happy"] += 30
        elif pol > 0:  scores["happy"] += 10
        elif pol < -0.5: scores["sad"] += 25; scores["angry"] += 15
        elif pol < 0:  scores["sad"]   += 15

    total  = sum(scores.values()) or 1
    scores = {k: round(v/total*100,2) for k,v in scores.items()}
    return {"dominant":max(scores,key=scores.get),"scores":scores,
            "sentiment":pol,"method":"Keyword + TextBlob","keywords":list(set(kws))[:8]}

# ─────────────────────────────────────────────────────────────────
# FUSION ENGINE
# ─────────────────────────────────────────────────────────────────
def fuse_emotions(results, weights):
    fused, total_w = {e: 0.0 for e in EMOTION_KEYS}, 0.0
    for mod, result in results.items():
        if result is None: continue
        w = weights.get(mod, 0.0)
        for e in EMOTION_KEYS:
            fused[e] += result["scores"].get(e,0.0) * w
        total_w += w
    if total_w > 0:
        fused = {k: round(v/total_w,2) for k,v in fused.items()}
    dominant = max(fused, key=fused.get)
    return {"dominant":dominant,"scores":fused,"confidence":fused[dominant]}

# ─────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────
def bars_html(scores):
    out = ""
    for e, val in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        info = EMOTIONS.get(e, EMOTIONS["neutral"])
        out += f"""
<div class="bar-row">
  <div class="bar-label">{info['emoji']} {e[:7].upper()}</div>
  <div class="bar-track"><div class="bar-fill" style="width:{val:.1f}%;background:{info['color']}"></div></div>
  <div class="bar-pct">{val:.0f}%</div>
</div>"""
    return out

def render_modality_card(result, title, badge_cls):
    if result is None:
        st.markdown(f"""
<div class="mcard">
  <div class="mcard-title"><span class="badge {badge_cls}">{title}</span></div>
  <div style="text-align:center;color:#3a5a6a;font-family:Share Tech Mono,monospace;font-size:12px;padding:24px 0">NO DATA YET</div>
</div>""", unsafe_allow_html=True); return
    dom  = result["dominant"]
    sc   = result["scores"]
    info = EMOTIONS.get(dom, EMOTIONS["neutral"])
    sim  = " · SIM" if result.get("simulated") else ""
    st.markdown(f"""
<div class="mcard">
  <div class="mcard-title"><span class="badge {badge_cls}">{title}{sim}</span></div>
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px">
    <div style="font-size:50px">{info['emoji']}</div>
    <div>
      <div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:{info['color']}">{dom.upper()}</div>
      <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#4a6070">CONFIDENCE: {sc.get(dom,0):.1f}%</div>
    </div>
  </div>{bars_html(sc)}
</div>""", unsafe_allow_html=True)

def render_fused_card(result):
    if result is None:
        st.markdown("""
<div class="emo-big">
  <span class="emo-emoji">🧬</span>
  <div class="emo-name" style="color:#3a5a6a">AWAITING INPUT</div>
  <div class="emo-conf">Run at least one modality to see fused result</div>
</div>""", unsafe_allow_html=True); return
    dom  = result["dominant"]
    sc   = result["scores"]
    conf = result["confidence"]
    info = EMOTIONS.get(dom, EMOTIONS["neutral"])
    st.markdown(f"""
<div class="emo-big" style="border-color:{info['color']}50">
  <span class="emo-emoji">{info['emoji']}</span>
  <div class="emo-name" style="color:{info['color']}">{dom.upper()}</div>
  <div class="emo-conf">{info['desc']}</div>
  <div style="margin-top:10px"><span class="badge badge-fused">⚡ FUSED · {conf:.1f}% CONFIDENCE</span></div>
</div>""", unsafe_allow_html=True)

def render_chart(results_dict):
    if not PD_OK:
        st.info("Install pandas for charts: `pip install pandas`"); return
    data = {}
    for name, result in results_dict.items():
        if result is None: continue
        data[name.upper()] = [result["scores"].get(e,0) for e in EMOTION_KEYS]
    if not data:
        st.info("Run analyses to see comparison chart."); return
    labels = [EMOTIONS[e]["emoji"]+" "+e.upper() for e in EMOTION_KEYS]
    df = pd.DataFrame(data, index=labels)
    st.bar_chart(df, width='stretch', height=300)

def render_history_chart(history):
    if not history: st.info("Run analyses to see history."); return
    if not PD_OK:
        counts = Counter(history)
        for e,c in counts.most_common():
            st.write(f"{EMOTIONS[e]['emoji']} {e.upper()}: {c}")
        return
    counts = Counter(history)
    labels = [EMOTIONS[e]["emoji"]+" "+e.upper() for e in EMOTION_KEYS if e in counts]
    vals   = [counts[e] for e in EMOTION_KEYS if e in counts]
    df     = pd.DataFrame({"Count": vals}, index=labels)
    st.bar_chart(df, width='stretch', height=240)

def render_log():
    entries = "<br>".join(st.session_state.log[:20]) if st.session_state.log \
              else "<span style='color:#3a5a6a'>System ready. Waiting for input...</span>"
    st.markdown(f"<div class='logbox'>{entries}</div>", unsafe_allow_html=True)

def do_fuse():
    """Run fusion and update session state."""
    fused = fuse_emotions({
        "face":   st.session_state.face_result,
        "speech": st.session_state.speech_result,
        "text":   st.session_state.text_result,
    }, st.session_state.get("_weights", {"face":0.45,"speech":0.35,"text":0.20}))
    st.session_state.fused_result = fused
    st.session_state.history.append(fused["dominant"])
    st.session_state.session_count += 1
    return fused

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    st.markdown("**Fusion Weights**")
    w_face   = st.slider("📷 Face",   0,100,45,5)
    w_speech = st.slider("🎤 Speech", 0,100,35,5)
    w_text   = st.slider("💬 Text",   0,100,20,5)
    total_w  = w_face + w_speech + w_text or 1
    weights  = {"face":w_face/total_w,"speech":w_speech/total_w,"text":w_text/total_w}
    st.session_state["_weights"] = weights
    st.caption(f"Face {w_face/total_w*100:.0f}% | Speech {w_speech/total_w*100:.0f}% | Text {w_text/total_w*100:.0f}%")

    st.divider()
    st.markdown("**Camera**")
    cam_idx       = st.selectbox("Camera index",[0,1,2],index=0)
    analyze_every = st.slider("Analyze every N frames",1,8,3)
    show_lm       = st.checkbox("Landmark dots",True)

    st.divider()
    st.markdown("**Microphone**")
    mic_duration = st.slider("Recording duration (sec)",2,30,5)
    st.session_state.mic_duration = mic_duration

    st.divider()
    st.markdown("**Library Status**")
    st.markdown(f"""
<div style='font-family:Share Tech Mono,monospace;font-size:11px;line-height:2.4'>
  <span style='color:#00e5ff'>OPENCV    </span> {"✅" if CV2_OK   else "⚠️ pip install opencv-python"}<br>
  <span style='color:#00e5ff'>DEEPFACE  </span> {"✅" if DF_OK    else "⚠️ pip install deepface tf-keras"}<br>
  <span style='color:#00ff88'>LIBROSA   </span> {"✅" if AUDIO_OK else "⚠️ pip install librosa soundfile"}<br>
  <span style='color:#ff3f80'>SOUNDDEV  </span> {"✅" if SD_OK    else "⚠️ pip install sounddevice"}<br>
  <span style='color:#ff3f80'>AUDIORCDR </span> {"✅" if MIC_OK   else "⚠️ pip install streamlit-audiorecorder"}<br>
  <span style='color:#9f6fff'>TEXTBLOB  </span> {"✅" if TB_OK    else "⚠️ pip install textblob"}<br>
  <span style='color:#ffb830'>HF TRANS  </span> {"✅" if HF_OK    else "⚠️ pip install transformers torch"}<br>
  <span style='color:#00e5ff'>PANDAS    </span> {"✅" if PD_OK    else "⚠️ pip install pandas"}
</div>""", unsafe_allow_html=True)

    st.divider()
    if st.button("🗑 Clear Session"):
        for k in ["log","face_result","speech_result","text_result","fused_result",
                  "history","mic_result","mic_audio_data"]:
            st.session_state[k] = [] if isinstance(st.session_state.get(k),list) else None
        st.session_state["session_count"] = 0
        st.session_state["face_frames"]   = 0
        st.rerun()

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="header-tag">MULTI-MODAL AI SYSTEM // v4.0 // PURE PYTHON</div>
  <div class="header-title">EMOTION <span>·</span> FUSION AI</div>
  <div style="margin-top:8px">
    <span class="badge badge-face">📷 FACE CNN</span>
    <span class="badge badge-speech">🎤 SPEECH DSP</span>
    <span class="badge badge-mic">🎙 LIVE MIC</span>
    <span class="badge badge-text">💬 NLP TEXT</span>
    <span class="badge badge-fused">⚡ FUSION ENGINE</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pipeline">
  <div class="pipe-node"><span class="pipe-icon">📷</span><span class="pipe-label">FACE</span></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-node"><span class="pipe-icon">🧠</span><span class="pipe-label">CNN</span></div>
  <div class="pipe-arrow">+</div>
  <div class="pipe-node"><span class="pipe-icon">🎙</span><span class="pipe-label">MIC</span></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-node"><span class="pipe-icon">📊</span><span class="pipe-label">LIBROSA</span></div>
  <div class="pipe-arrow">+</div>
  <div class="pipe-node"><span class="pipe-icon">🎤</span><span class="pipe-label">AUDIO</span></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-node"><span class="pipe-icon">📊</span><span class="pipe-label">FEATURES</span></div>
  <div class="pipe-arrow">+</div>
  <div class="pipe-node"><span class="pipe-icon">💬</span><span class="pipe-label">TEXT</span></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-node"><span class="pipe-icon">🔤</span><span class="pipe-label">NLP</span></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-node"><span class="pipe-icon">⚡</span><span class="pipe-label">FUSION</span></div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-node"><span class="pipe-icon">🎯</span><span class="pipe-label">OUTPUT</span></div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Live Face",
    "🎙 Live Microphone",
    "🎤 Audio Upload",
    "💬 Text Analysis",
    "📊 Fused Results",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — LIVE FACE
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📷 Real-Time Facial Emotion Detection")
    st.caption("OpenCV detects face → DeepFace CNN classifies 7 emotions in real time")

    col_cam, col_side = st.columns([3,2], gap="large")
    with col_cam:
        c1,c2 = st.columns(2)
        with c1: start_btn = st.button("▶ START CAMERA", disabled=st.session_state.cam_running)
        with c2: stop_btn  = st.button("■ STOP CAMERA",  disabled=not st.session_state.cam_running)
        frame_ph  = st.empty()
        status_ph = st.empty()
        if not st.session_state.cam_running:
            frame_ph.markdown("""
<div class="cam-ph"><div class="cam-ph-icon">📷</div>
<div class="cam-ph-text">CLICK ▶ START CAMERA</div></div>""", unsafe_allow_html=True)

    with col_side:
        st.markdown("#### Detected Emotion")
        face_card_ph = st.empty()
        face_card_ph.markdown("""
<div class="mcard"><div class="mcard-title"><span class="badge badge-face">📷 FACE MODULE</span></div>
<div style="text-align:center;color:#3a5a6a;font-family:Share Tech Mono,monospace;font-size:12px;padding:24px 0">
START CAMERA TO DETECT</div></div>""", unsafe_allow_html=True)
        st.markdown("#### Stats")
        stats_ph = st.empty()

    if start_btn:
        st.session_state.cam_running = True
        st.session_state.face_frames = 0
        add_log("Camera started","face")
        st.rerun()
    if stop_btn:
        st.session_state.cam_running = False
        add_log("Camera stopped","face")
        st.rerun()

    if st.session_state.cam_running:
        if not CV2_OK:
            st.error("Install OpenCV: `pip install opencv-python`")
            st.session_state.cam_running = False
        else:
            with st.spinner("Warming up DeepFace..."):
                warmup_deepface()
            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                st.error(f"Cannot open camera {cam_idx}")
                st.session_state.cam_running = False
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
                cascade = get_face_cascade()
                status_ph.success("🟢 Camera live — click ■ STOP to end")
                frame_n = 0
                while st.session_state.cam_running:
                    ret, frame = cap.read()
                    if not ret: break
                    frame  = cv2.flip(frame,1)
                    frame_n += 1
                    st.session_state.face_frames += 1
                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray  = cv2.equalizeHist(gray)
                    faces = cascade.detectMultiScale(gray,1.1,5,minSize=(60,60))
                    if len(faces) > 0:
                        if frame_n % analyze_every == 0:
                            x,y,w,h = faces[0]
                            r = analyze_face_image(frame[y:y+h,x:x+w])
                            st.session_state.face_result = r
                            fused = fuse_emotions({"face":r,
                                "speech":st.session_state.speech_result,
                                "text":st.session_state.text_result}, weights)
                            st.session_state.fused_result = fused
                            st.session_state.history.append(fused["dominant"])
                            st.session_state.session_count += 1
                            add_log(f"Face → {r['dominant'].upper()} ({r['scores'].get(r['dominant'],0):.0f}%)","face")
                        if show_lm and st.session_state.face_result:
                            frame = draw_face_overlay(frame,faces,
                                        st.session_state.face_result["dominant"],
                                        st.session_state.face_result["scores"])
                    else:
                        cv2.putText(frame,"No Face Detected",(150,240),
                                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(60,60,60),2,cv2.LINE_AA)
                    frame_ph.image(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),
                                   width='stretch')
                    if st.session_state.face_result:
                        r    = st.session_state.face_result
                        dom  = r["dominant"]; sc = r["scores"]
                        info = EMOTIONS.get(dom,EMOTIONS["neutral"])
                        face_card_ph.markdown(f"""
<div class="mcard">
  <div class="mcard-title"><span class="badge badge-face">📷 FACE MODULE</span></div>
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
    <div style="font-size:48px">{info['emoji']}</div>
    <div>
      <div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:{info['color']}">{dom.upper()}</div>
      <div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#4a6070">CONF: {sc.get(dom,0):.1f}%</div>
    </div>
  </div>{bars_html(sc)}</div>""", unsafe_allow_html=True)
                        stats_ph.markdown(f"""
<div class="mcard"><div class="mcard-title">STATS</div>
<div style="font-family:Share Tech Mono,monospace;font-size:11px;line-height:2.4">
  <span style="color:#4a6070">FACES   </span> {len(faces)}<br>
  <span style="color:#4a6070">EMOTION </span> <span style="color:{info['color']}">{dom.upper()}</span><br>
  <span style="color:#4a6070">FRAMES  </span> {st.session_state.face_frames}<br>
  <span style="color:#4a6070">BACKEND </span> {"DeepFace CNN" if DF_OK else "Simulated"}
</div></div>""", unsafe_allow_html=True)
                    time.sleep(0.04)
                cap.release()
                status_ph.info("⏹ Camera stopped.")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — 🎙 LIVE MICROPHONE  (NEW!)
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🎙 Live Microphone Emotion Detection")
    st.caption("Record voice from your microphone → extract acoustic features → classify emotion")

    col_m1, col_m2 = st.columns([2,2], gap="large")

    with col_m1:
        st.markdown("#### 🎙 Record Your Voice")

        # ── Method A: streamlit-audiorecorder (best UX) ──────────
        if MIC_OK:
            st.markdown("""
<div class="method-box">
  <div class="method-label">METHOD A — BROWSER MICROPHONE (RECOMMENDED)</div>
""", unsafe_allow_html=True)
            st.caption("Click the mic button below to record directly in your browser")

            audio_data = audiorecorder(
                start_prompt="🎙 Click to START recording",
                stop_prompt="⏹ Click to STOP recording",
                pause_prompt="",
                show_visualizer=True,
                key="mic_recorder"
            )

            if len(audio_data) > 0:
                # Export to WAV bytes then analyze
                wav_io = io.BytesIO()
                audio_data.export(wav_io, format="wav")
                wav_io.seek(0)
                st.audio(wav_io, format="audio/wav")
                wav_io.seek(0)

                if st.button("🔬 ANALYZE MIC RECORDING", key="analyze_mic_browser"):
                    with st.spinner("Analyzing voice emotion..."):
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(wav_io.read())
                            tmp_path = tmp.name
                        result = analyze_audio_file(tmp_path)
                        os.unlink(tmp_path)
                        st.session_state.speech_result = result
                        st.session_state.mic_result    = result
                        fused = fuse_emotions({
                            "face":   st.session_state.face_result,
                            "speech": result,
                            "text":   st.session_state.text_result,
                        }, weights)
                        st.session_state.fused_result = fused
                        st.session_state.history.append(fused["dominant"])
                        st.session_state.session_count += 1
                        add_log(f"Mic (browser) → {result['dominant'].upper()} ({result['scores'].get(result['dominant'],0):.0f}%)","mic")
                    dom  = result["dominant"]
                    info = EMOTIONS.get(dom,EMOTIONS["neutral"])
                    st.success(f"✅ Voice Emotion → **{info['emoji']} {dom.upper()}** ({result['scores'].get(dom,0):.0f}% confidence)")

            st.markdown("</div>", unsafe_allow_html=True)
            st.divider()

        # ── Method B: sounddevice (desktop) ──────────────────────
        st.markdown("""
<div class="method-box">
  <div class="method-label">METHOD B — DESKTOP MICROPHONE (sounddevice)</div>
""", unsafe_allow_html=True)
        if SD_OK:
            st.caption(f"Records {mic_duration} seconds from your system microphone")

            col_rb1, col_rb2 = st.columns(2)
            with col_rb1:
                rec_btn = st.button(
                    "🔴 RECORD NOW",
                    key="record_sd",
                    disabled=st.session_state.mic_recording
                )
            with col_rb2:
                # Show status
                if st.session_state.mic_recording:
                    st.markdown("""
<div style="display:flex;align-items:center;gap:8px;padding:10px;
 background:rgba(255,63,128,.1);border:1px solid #ff3f80;border-radius:8px">
  <span class="rec-dot"></span>
  <span style="font-family:Share Tech Mono,monospace;font-size:12px;color:#ff3f80">RECORDING...</span>
</div>""", unsafe_allow_html=True)

            if rec_btn and SD_OK:
                st.session_state.mic_recording = True
                progress_ph  = st.empty()
                waveform_ph  = st.empty()

                # Show animated waveform while recording
                waveform_ph.markdown("""
<div style="text-align:center;padding:20px;background:#0b1520;border:1px solid #1a2f45;border-radius:10px">
  <div class="wave-bars">
    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
    <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
  </div>
  <div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#ff3f80;margin-top:8px">
    <span class="rec-dot"></span>RECORDING VOICE...
  </div>
</div>""", unsafe_allow_html=True)

                # Show countdown
                for i in range(mic_duration, 0, -1):
                    progress_ph.markdown(f"""
<div style="text-align:center;font-family:Share Tech Mono,monospace;
 font-size:28px;color:#ff3f80;padding:8px">{i}s remaining</div>""",
                        unsafe_allow_html=True)
                    time.sleep(1)

                # Record audio
                audio_arr, err = record_microphone(mic_duration, sr=22050)
                st.session_state.mic_recording = False
                waveform_ph.empty()
                progress_ph.empty()

                if err:
                    st.error(f"Recording error: {err}")
                    add_log(f"Mic error: {err}", "mic")
                else:
                    st.session_state.mic_audio_data = audio_arr
                    add_log(f"Mic recorded {mic_duration}s of audio","mic")

                    # Analyze
                    with st.spinner("Analyzing voice emotion..."):
                        result = analyze_audio_array(audio_arr, sr=22050)
                        st.session_state.speech_result = result
                        st.session_state.mic_result    = result
                        fused = fuse_emotions({
                            "face":   st.session_state.face_result,
                            "speech": result,
                            "text":   st.session_state.text_result,
                        }, weights)
                        st.session_state.fused_result = fused
                        st.session_state.history.append(fused["dominant"])
                        st.session_state.session_count += 1
                        add_log(f"Mic → {result['dominant'].upper()} ({result['scores'].get(result['dominant'],0):.0f}%)","mic")

                    dom  = result["dominant"]
                    info = EMOTIONS.get(dom,EMOTIONS["neutral"])
                    st.success(f"✅ Voice Emotion → **{info['emoji']} {dom.upper()}** ({result['scores'].get(dom,0):.0f}% confidence)")

                    # Play back recorded audio
                    wav_buf = numpy_to_wav_bytes(audio_arr, sr=22050)
                    st.audio(wav_buf, format="audio/wav")
        else:
            st.warning("sounddevice not installed.")
            st.code("pip install sounddevice", language="bash")

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Method C: HTML5 browser mic via st.audio_input ───────
        st.markdown("""
<div class="method-box">
  <div class="method-label">METHOD C — STREAMLIT BUILT-IN MIC (v1.31+)</div>
""", unsafe_allow_html=True)
        try:
            # st.audio_input is available in Streamlit >= 1.31
            audio_input = st.audio_input("🎙 Click mic icon to record", key="builtin_mic")
            if audio_input is not None:
                st.audio(audio_input, format="audio/wav")
                if st.button("🔬 ANALYZE THIS RECORDING", key="analyze_builtin"):
                    with st.spinner("Analyzing..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_input.read())
                            tmp_path = tmp.name
                        result = analyze_audio_file(tmp_path)
                        os.unlink(tmp_path)
                        st.session_state.speech_result = result
                        st.session_state.mic_result    = result
                        fused = fuse_emotions({
                            "face":   st.session_state.face_result,
                            "speech": result,
                            "text":   st.session_state.text_result,
                        }, weights)
                        st.session_state.fused_result = fused
                        st.session_state.history.append(fused["dominant"])
                        st.session_state.session_count += 1
                        add_log(f"Mic (builtin) → {result['dominant'].upper()}","mic")
                    dom = result["dominant"]
                    info = EMOTIONS.get(dom,EMOTIONS["neutral"])
                    st.success(f"✅ Voice → **{info['emoji']} {dom.upper()}**")
        except AttributeError:
            st.info("Upgrade Streamlit for built-in mic: `pip install --upgrade streamlit`")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_m2:
        st.markdown("#### 🎙 Microphone Result")

        r = st.session_state.mic_result or st.session_state.speech_result
        if r:
            dom  = r["dominant"]
            sc   = r["scores"]
            info = EMOTIONS.get(dom, EMOTIONS["neutral"])
            st.markdown(f"""
<div class="mcard" style="border-color:{info['color']}40">
  <div class="mcard-title"><span class="badge badge-mic">🎙 MICROPHONE MODULE</span></div>
  <div style="text-align:center;padding:16px 0">
    <div style="font-size:64px">{info['emoji']}</div>
    <div style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;
                color:{info['color']};margin:8px 0">{dom.upper()}</div>
    <div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#4a6070">
      CONFIDENCE: {sc.get(dom,0):.1f}%</div>
  </div>
  {bars_html(sc)}
</div>""", unsafe_allow_html=True)

            feats = r.get("features",{})
            if feats:
                rows = "".join(f"""
<div style="display:flex;justify-content:space-between;padding:5px 0;
 border-bottom:1px solid #1a2f45">
  <span style="color:#4a6070;font-size:11px;font-family:Share Tech Mono,monospace">{k}</span>
  <span style="color:#00e5ff;font-family:Share Tech Mono,monospace;font-size:11px">{v}</span>
</div>""" for k,v in feats.items())
                st.markdown(f"""
<div class="mcard">
  <div class="mcard-title">VOICE ACOUSTIC FEATURES</div>
  {rows}
</div>""", unsafe_allow_html=True)

            # Show waveform info if available
            if st.session_state.mic_audio_data is not None:
                arr = st.session_state.mic_audio_data
                st.markdown(f"""
<div class="mcard">
  <div class="mcard-title">RECORDING INFO</div>
  <div style="font-family:Share Tech Mono,monospace;font-size:11px;line-height:2.4">
    <span style="color:#4a6070">SAMPLES   </span> <span style="color:#c5d8e8">{len(arr):,}</span><br>
    <span style="color:#4a6070">DURATION  </span> <span style="color:#c5d8e8">{len(arr)/22050:.1f}s</span><br>
    <span style="color:#4a6070">SAMPLE RATE</span> <span style="color:#c5d8e8">22,050 Hz</span><br>
    <span style="color:#4a6070">PEAK AMP  </span> <span style="color:#c5d8e8">{float(np.max(np.abs(arr))):.4f}</span><br>
    <span style="color:#4a6070">RMS ENERGY</span> <span style="color:#c5d8e8">{float(np.sqrt(np.mean(arr**2))):.4f}</span>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="mic-card">
  <span class="mic-icon">🎙</span>
  <div class="mic-status" style="color:#4a6070">WAITING FOR VOICE INPUT</div>
  <div style="font-family:Share Tech Mono,monospace;font-size:10px;color:#2a3f50;margin-top:8px">
    USE ANY METHOD ON THE LEFT<br>TO RECORD YOUR VOICE
  </div>
</div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("""
<div class="mcard">
  <div class="mcard-title">HOW VOICE EMOTION WORKS</div>
  <div style="font-size:12px;color:#4a6070;line-height:1.9">
    <b style="color:#c5d8e8">1. Pitch</b> — High pitch = excitement/anger<br>
    <b style="color:#c5d8e8">2. Energy</b> — High energy = intense emotion<br>
    <b style="color:#c5d8e8">3. Tempo</b> — Fast speech = happiness/anger<br>
    <b style="color:#c5d8e8">4. ZCR</b> — High ZCR = fear/anxiety<br>
    <b style="color:#c5d8e8">5. MFCC</b> — Vocal quality fingerprint<br>
    <b style="color:#c5d8e8">6. Spectral</b> — Voice brightness/tone
  </div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — AUDIO FILE UPLOAD
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🎤 Audio File Upload & Analysis")
    st.caption("Upload any audio file → Librosa extracts acoustic features → emotion classification")

    col_a1,col_a2 = st.columns([2,2], gap="large")
    with col_a1:
        st.markdown("#### Upload Audio File")
        audio_file = st.file_uploader("Drop audio", type=["wav","mp3","ogg","flac","m4a"],
                                      label_visibility="collapsed")
        if audio_file:
            st.audio(audio_file)
            st.caption(f"**{audio_file.name}** · {audio_file.size/1024:.1f} KB")
            if st.button("🔬 ANALYZE AUDIO FILE"):
                with st.spinner("Extracting features..."):
                    sfx = "." + audio_file.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(delete=False,suffix=sfx) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                    result = analyze_audio_file(tmp_path)
                    os.unlink(tmp_path)
                    st.session_state.speech_result = result
                    fused = fuse_emotions({"face":st.session_state.face_result,"speech":result,
                                           "text":st.session_state.text_result},weights)
                    st.session_state.fused_result = fused
                    st.session_state.history.append(fused["dominant"])
                    st.session_state.session_count += 1
                    add_log(f"Audio file → {result['dominant'].upper()}","speech")
                st.success(f"✅ Audio → **{result['dominant'].upper()}**")
        else:
            st.markdown("""
<div style="background:#0b1520;border:1px dashed #1a2f45;border-radius:10px;padding:32px;text-align:center">
  <div style="font-size:40px;margin-bottom:10px">🎵</div>
  <div style="font-family:Share Tech Mono,monospace;color:#3a5a6a;font-size:12px;letter-spacing:2px">
    UPLOAD AUDIO FILE<br><span style="font-size:10px;color:#2a3f50">WAV · MP3 · OGG · FLAC · M4A</span>
  </div>
</div>""", unsafe_allow_html=True)

    with col_a2:
        st.markdown("#### Result")
        render_modality_card(st.session_state.speech_result, "🎤 SPEECH MODULE", "badge-speech")
        r = st.session_state.speech_result
        if r and r.get("features"):
            feats = r["features"]
            rows  = "".join(f"""
<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1a2f45">
  <span style="color:#4a6070;font-size:11px">{k}</span>
  <span style="color:#00e5ff;font-family:Share Tech Mono,monospace;font-size:11px">{v}</span>
</div>""" for k,v in feats.items())
            st.markdown(f"""
<div class="mcard"><div class="mcard-title">ACOUSTIC FEATURES</div>{rows}</div>""",
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — TEXT
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 💬 Text Emotion Analysis")
    st.caption("HuggingFace Transformer OR TextBlob + keyword matching → emotion prediction")

    col_t1,col_t2 = st.columns([2,2], gap="large")
    with col_t1:
        st.markdown("#### Enter Text")
        samples = {
            "Select a sample...":"",
            "😄 Happy":  "I'm so excited and happy today! Everything is going wonderfully well!",
            "😢 Sad":    "I feel so lonely and hopeless. Nothing seems to be working out for me.",
            "😠 Angry":  "I'm absolutely furious! This is completely unacceptable and infuriating.",
            "😨 Fear":   "I'm terrified and anxious. The uncertainty is making me panic badly.",
            "😮 Surprise":"I'm completely shocked and astonished! I never expected this at all.",
            "🤢 Disgust":"This is disgusting and revolting. I find it completely horrible.",
        }
        sel  = st.selectbox("Sample:",list(samples.keys()))
        text = st.text_area("Your text:",value=samples.get(sel,""),height=160,
                            placeholder="Type anything to detect its emotion...")
        st.caption(f"Words: {len(text.split())} · Chars: {len(text)}")
        if st.button("🔬 ANALYZE TEXT EMOTION"):
            if not text.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Running NLP..."):
                    result = analyze_text(text)
                    st.session_state.text_result = result
                    fused = fuse_emotions({"face":st.session_state.face_result,
                                           "speech":st.session_state.speech_result,
                                           "text":result},weights)
                    st.session_state.fused_result = fused
                    st.session_state.history.append(fused["dominant"])
                    st.session_state.session_count += 1
                    add_log(f"Text → {result['dominant'].upper()} via {result['method']}","text")
                st.success(f"✅ Text → **{result['dominant'].upper()}** | {result['method']}")

    with col_t2:
        st.markdown("#### Result")
        r = st.session_state.text_result
        if r:
            render_modality_card(r,f"💬 TEXT · {r.get('method','')[:20].upper()}","badge-text")
            pol   = r.get("sentiment",0)
            p_lbl = "POSITIVE" if pol>0.1 else ("NEGATIVE" if pol<-0.1 else "NEUTRAL")
            p_col = "#00ff88" if pol>0.1 else ("#ff3f80" if pol<-0.1 else "#8ba3b0")
            kws   = r.get("keywords",[])
            kw_h  = " ".join(f"<span class='badge badge-text'>{k}</span>" for k in kws) if kws else "—"
            st.markdown(f"""
<div class="mcard">
  <div class="mcard-title">SENTIMENT & KEYWORDS</div>
  <div style="font-family:Share Tech Mono,monospace;font-size:12px;margin-bottom:10px">
    <span style="color:#4a6070">SENTIMENT: </span>
    <span style="color:{p_col}">{p_lbl} ({pol:+.2f})</span>
  </div>
  <div style="line-height:2.4">{kw_h}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="mcard"><div class="mcard-title"><span class="badge badge-text">💬 TEXT MODULE</span></div>
<div style="text-align:center;color:#3a5a6a;font-family:Share Tech Mono,monospace;font-size:12px;padding:28px 0">
ENTER TEXT TO ANALYZE</div></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5 — FUSED RESULTS
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### ⚡ Fused Emotion Result & Analytics")

    col_r1,col_r2 = st.columns([2,3], gap="large")
    with col_r1:
        render_fused_card(st.session_state.fused_result)
        if st.session_state.fused_result:
            dom_em = st.session_state.fused_result["dominant"]
            st.markdown(f"""
<div class="insight-card">
  <div class="insight-title">🔍 CLINICAL INSIGHT</div>
  <div class="insight-body">{INSIGHTS.get(dom_em,"")}</div>
</div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Session Stats")
        m1,m2 = st.columns(2)
        m3,m4 = st.columns(2)
        m1.metric("ANALYSES",   st.session_state.session_count)
        m2.metric("CAM FRAMES", st.session_state.face_frames)
        top_em = Counter(st.session_state.history).most_common(1)[0][0].upper() \
                 if st.session_state.history else "—"
        m3.metric("TOP EMOTION", top_em)
        active = sum(1 for x in [st.session_state.face_result,
                                   st.session_state.speech_result,
                                   st.session_state.text_result] if x)
        m4.metric("ACTIVE MODES", f"{active}/3")

    with col_r2:
        st.markdown("#### Per-Modality Results")
        render_modality_card(st.session_state.face_result,   "📷 FACE",         "badge-face")
        render_modality_card(st.session_state.mic_result or st.session_state.speech_result,
                             "🎙 MIC / SPEECH",  "badge-mic")
        render_modality_card(st.session_state.text_result,   "💬 TEXT",         "badge-text")

    st.divider()
    col_c1,col_c2 = st.columns([1,1], gap="large")
    with col_c1:
        st.markdown("#### 📊 Modality Score Comparison")
        render_chart({"face":st.session_state.face_result,
                      "speech":st.session_state.speech_result,
                      "text":st.session_state.text_result,
                      "fused":st.session_state.fused_result})
    with col_c2:
        st.markdown("#### 📈 Session Emotion History")
        render_history_chart(st.session_state.history)

    st.divider()
    if st.button("⚡ RE-FUSE WITH CURRENT WEIGHTS"):
        fused = fuse_emotions({"face":st.session_state.face_result,
                                "speech":st.session_state.speech_result,
                                "text":st.session_state.text_result}, weights)
        st.session_state.fused_result = fused
        if any(x for x in [st.session_state.face_result,
                             st.session_state.speech_result,
                             st.session_state.text_result]):
            st.session_state.history.append(fused["dominant"])
            add_log(f"Re-fused → {fused['dominant'].upper()} ({fused['confidence']:.1f}%)","fused")
            st.success(f"✅ Fused: **{fused['dominant'].upper()}** · {fused['confidence']:.1f}% confidence")
            st.rerun()
        else:
            st.warning("Run at least one modality first.")

    st.markdown("#### 🖥 System Log")
    render_log()
