---
title: MultiModal Emotion AI
sdk: streamlit
sdk_version: 1.39.0
python_version: "3.10"
app_file: multimodal_emotion.py
pinned: false
---


🧬 MultiModal Emotion AI

A real-time, multi-modal emotion recognition app built with Streamlit. It fuses signals from face, voice/microphone, uploaded audio, and text into a single confidence-weighted emotion prediction — with a fully custom dark, cyberpunk-styled UI.

✨ Features


📷 Live Face Emotion Detection — OpenCV face detection + DeepFace CNN classifies 7 emotions (happy, sad, angry, fear, surprise, disgust, neutral) from a live webcam feed, with on-frame emotion overlays and landmark markers.
🎙 Live Microphone Analysis — Record audio directly in-browser (via sounddevice or streamlit-audiorecorder) and extract acoustic features (pitch, energy, tempo, zero-crossing rate, MFCC variance, spectral centroid) using librosa, converted into an emotion prediction through rule-based DSP scoring.
🎤 Audio File Upload — Upload .wav, .mp3, .ogg, .flac, or .m4a files for the same acoustic-feature-based emotion analysis.
💬 Text Emotion Analysis — Uses a HuggingFace transformer (j-hartmann/emotion-english-distilroberta-base) when available, falling back to TextBlob sentiment + keyword matching. Includes sample text presets for each emotion.
⚡ Fusion Engine — Combines face, speech, and text predictions using user-adjustable weights (sidebar sliders) into a single fused emotion + confidence score.
📊 Analytics Dashboard — Per-modality result cards, modality score comparison chart, session emotion history chart, and a live system log.
🩹 Graceful Degradation — Every optional dependency (OpenCV, DeepFace, librosa, TextBlob, transformers, sounddevice, audiorecorder, pandas) is optional. If a library isn't installed, the app falls back to a simulated result and shows an install hint in the sidebar instead of crashing.



🖼️ Pipeline Overview

📷 Face  → 🧠 CNN (DeepFace)        ─┐
🎙 Mic   → 📊 Acoustic Features      ├─→ ⚡ Fusion Engine → 🎯 Final Emotion
💬 Text  → 🔤 NLP (Transformer/TB)  ─┘






 Run the app

streamlit run multimodal_emotion.py

Then open the URL Streamlit prints (typically http://localhost:8501) in your browser.


📦 Dependency Reference

LibraryPurposeRequired forstreamlitApp framework / UICore appnumpy, pillowArray & image handlingCore appopencv-pythonWebcam capture & face detectionLive Face tabdeepface, tf-kerasCNN-based facial emotion classificationLive Face tablibrosa, soundfileAudio loading & acoustic feature extractionMic & Audio Upload tabssounddeviceMicrophone recordingLive Microphone tabstreamlit-audiorecorderBrowser-based mic recorder widgetLive Microphone tabtextblobSentiment polarity scoringText tab (fallback method)transformers, torchTransformer-based emotion classificationText tab (primary method)pandasData tables / chartsFused Results charts

The sidebar's Library Status panel shows real-time install status for each of these.


🎛️ Using the App


Sidebar Settings

Adjust Fusion Weights for Face / Speech / Text (auto-normalized to 100%).
Configure camera index, frame-analysis rate, and landmark overlay toggle.
Set microphone recording duration.
Check Library Status for missing optional dependencies.
Clear Session to reset all results, history, and logs.



Tab 1 — Live Face: Start/stop the webcam to detect facial emotion in real time.
Tab 2 — Live Microphone: Record a short clip and get an instant voice-based emotion reading.
Tab 3 — Audio Upload: Upload a pre-recorded audio file for the same analysis.
Tab 4 — Text Analysis: Type or select sample text to get an NLP-based emotion + sentiment + keyword breakdown.
Tab 5 — Fused Results: View the combined emotion, clinical-style insight, per-modality breakdown, comparison charts, session stats, and the system log. Use Re-fuse with Current Weights to recompute after adjusting sliders.



🧠 How Fusion Works

Each modality (face, speech, text) produces a 7-emotion probability distribution. The fusion engine computes a weighted average across all active modalities:

fused_score[emotion] = Σ (modality_score[emotion] × modality_weight) / Σ modality_weight

The emotion with the highest fused score becomes the dominant prediction, alongside a confidence percentage.


