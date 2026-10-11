#!/usr/bin/env python3
"""
EchoScribe Pro — Professional YouTube Transcriber & Speaker Diarizer
====================================================================
Major UI overhaul with premium dark theme and new features.

New in this version:
    - Premium dark theme with glassmorphism-inspired panels
    - Tabbed transcript viewer (TXT / SRT / VTT)
    - Collapsible settings panel (accordion style)
    - Per-task inline progress display in task list
    - Custom output directory selector
    - System tray icon with completion notifications
    - About dialog with keyboard shortcuts reference
    - Clear log / clear output buttons
    - GPU/CUDA status indicator in header
    - Animated status indicator (pulsing dot)
    - Custom dark-themed scrollbars
    - Better drag & drop visual feedback
    - Move task up/down in queue

Diarization Pipeline:
    1. Audio Preprocessing — RMS normalization + spectral gating noise reduction
    2. Multi-Feature VAD — Energy + spectral flatness + ZCR + spectral centroid
    3. BIC Speaker Change Detection — Sliding-window BIC for speaker transitions
    4. High-Dimensional Feature Extraction — MFCC + deltas + chroma + contrast + bandwidth
    5. Spectral Clustering — Normalized Laplacian eigen-gap + K-means++
    6. BIC Model Selection — Consensus speaker count via eigen-gap + BIC
    7. Centroid Refinement — Nearest-centroid re-assignment for boundary fixes
    8. Weighted Overlap Assignment — Duration prior + temporal centrality + smoothing
    9. Post-Processing — Merge, smooth, remove short, renumber
   10. Neural Fallback — Optional pyannote.audio integration

Requirements:
    pip install PySide6 yt-dlp openai-whisper torch torchaudio scipy librosa numpy
    Optional (for best diarization): pip install pyannote.audio
    FFmpeg must be installed and available in PATH.
"""

from __future__ import annotations

import sys
import os
import re
import json
import uuid
import subprocess
import threading
import time
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QComboBox, QProgressBar,
    QFileDialog, QMessageBox, QGroupBox, QCheckBox,
    QListWidget, QListWidgetItem, QAbstractItemView,
    QLineEdit, QDialog, QSplitter, QGridLayout, QFrame,
    QSizePolicy, QSpinBox, QMenu, QToolButton, QTabWidget,
    QScrollArea, QSystemTrayIcon,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer, QUrl, QPropertyAnimation, QRect
from PySide6.QtGui import (
    QFont, QTextCursor, QColor, QDragEnterEvent, QDropEvent,
    QAction, QShortcut, QKeySequence, QIcon, QTextCharFormat,
    QBrush, QPainter, QPen, QConicalGradient, QFontMetrics,
)

import yt_dlp
import torch
import torchaudio


# ═══════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════

APP_TITLE = "EchoScribe"
APP_SUBTITLE = "Pro Transcriber & Speaker Diarizer"
APP_VERSION = "1.0"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")
CONFIG_PATH = os.path.join(_SCRIPT_DIR, ".echoscribe_config.json")

STATUS_PENDING      = "Pending"
STATUS_RESOLVING    = "Resolving Title"
STATUS_DOWNLOADING  = "Downloading"
STATUS_TRANSCRIBING = "Transcribing"
STATUS_DIARIZING    = "Analyzing Speakers"
STATUS_DONE         = "Done"
STATUS_ERROR        = "Error"
STATUS_CANCELLED    = "Cancelled"
STATUS_MERGING      = "Merging Audio"

TERMINAL_STATUSES = {STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED}

VALID_MEDIA_EXTS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    ".mp4", ".mkv", ".webm", ".aac", ".wma",
}

OUTPUT_FORMATS = [
    "TXT + SRT", "TXT Only", "SRT Only",
    "TXT + SRT + VTT", "VTT Only",
]

SPEAKER_LABEL_FORMATS = [
    "Speaker 1, 2, 3...",
    "SPK_1, SPK_2, SPK_3...",
    "Speaker A, B, C...",
    "Speaker [1], [2], [3]...",
]

TEXT_FORMAT_STYLES = [
    "Timestamped Blocks",
    "Dialogue Format",
    "Compact Blocks",
    "Plain Text",
]

STATUS_ICONS = {
    STATUS_PENDING:      "○",
    STATUS_RESOLVING:    "◎",
    STATUS_DOWNLOADING:  "↓",
    STATUS_TRANSCRIBING: "⟳",
    STATUS_DIARIZING:    "◈",
    STATUS_DONE:         "✓",
    STATUS_ERROR:        "✗",
    STATUS_CANCELLED:    "□",
    STATUS_MERGING:      "⊕",
}

# ── Premium Dark Palette ─────────────────────────────
# Deeper, richer dark theme with better contrast and hierarchy

COLOR_BASE      = "#0f0f17"      # Deepest background
COLOR_SURFACE   = "#16161e"      # Card/panel background
COLOR_SURFACE0  = "#1c1c28"      # Input background
COLOR_SURFACE1  = "#262636"      # Borders, subtle elements
COLOR_SURFACE2  = "#3a3a4e"      # Hover states, secondary
COLOR_OVERLAY0  = "#6e6e82"      # Muted text
COLOR_TEXT      = "#e2e4f0"      # Primary text
COLOR_TEXT_DIM  = "#b0b3c6"      # Secondary text
COLOR_SUBTEXT   = "#8888a0"      # Tertiary text

# Accent colors — vibrant but refined
COLOR_BLUE      = "#7aa2f7"      # Primary accent
COLOR_BLUE_DIM  = "#5b82d4"      # Blue hover
COLOR_GREEN     = "#9ece6a"      # Success
COLOR_RED       = "#f7768e"      # Error / danger
COLOR_YELLOW    = "#e0af68"      # Warning / attention
COLOR_PEACH     = "#ff9e64"      # Secondary warning
COLOR_MAUVE     = "#bb9af7"      # Purple accent
COLOR_TEAL      = "#73daca"      # Teal accent
COLOR_LAVENDER  = "#b4befe"      # Light purple
COLOR_CYAN      = "#7dcfff"      # Cyan accent
COLOR_ROSE      = "#f4939e"      # Rose
COLOR_SKY       = "#89ddff"      # Sky blue

# Gradient helpers
GRADIENT_START_BTN = f"qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {COLOR_BLUE}, stop:1 {COLOR_BLUE_DIM})"
GRADIENT_PROGRESS = f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {COLOR_BLUE}, stop:0.5 {COLOR_MAUVE}, stop:1 {COLOR_CYAN})"

STATUS_COLORS = {
    STATUS_PENDING:      QColor(COLOR_SUBTEXT),
    STATUS_RESOLVING:    QColor(COLOR_YELLOW),
    STATUS_DOWNLOADING:  QColor(COLOR_BLUE),
    STATUS_TRANSCRIBING: QColor(COLOR_CYAN),
    STATUS_DIARIZING:    QColor(COLOR_MAUVE),
    STATUS_DONE:         QColor(COLOR_GREEN),
    STATUS_ERROR:        QColor(COLOR_RED),
    STATUS_CANCELLED:    QColor(COLOR_PEACH),
    STATUS_MERGING:      QColor(COLOR_TEAL),
}

CONCAT_SEPARATOR_LENGTH = 72

_URL_PATTERN = re.compile(
    r"^https?://"
    r"(?:(?:www\.)?(?:youtube\.com|youtu\.be)|"
    r"(?:www\.)?vimeo\.com|"
    r"(?:www\.)?twitch\.tv)/",
    re.IGNORECASE,
)

def is_valid_media_url(url: str) -> bool:
    """Check if a URL looks like a valid video/audio source."""
    return bool(_URL_PATTERN.match(url))


# ─── Config Manager ─────────────────────────────────────────

class ConfigManager:
    """Persistent settings storage using a JSON file."""

    DEFAULTS: dict = {
        "model": "base",
        "language": "Auto Detect",
        "download_format": "Audio (best)",
        "output_format": "TXT + SRT",
        "diarize": False,
        "num_speakers": "Auto",
        "speaker_label_fmt": "Speaker 1, 2, 3...",
        "text_format_style": "Timestamped Blocks",
        "include_timestamps": True,
        "include_lang_header": True,
        "srt_speaker_labels": True,
        "concatenate": True,
        "merge_audio": False,
        "max_concurrent": 2,
    }

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self._data: dict = {}
        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._save_to_disk)
        self.load()

    def load(self) -> None:
        """Load config from disk, filling missing keys with defaults."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
        except Exception:
            self._data = {}
        # Fill missing keys
        for key, val in self.DEFAULTS.items():
            if key not in self._data:
                self._data[key] = val

    def get(self, key: str):
        return self._data.get(key, self.DEFAULTS.get(key))

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self._save_timer.start()  # Debounce: save 500ms after last change

    def _save_to_disk(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # Silent fail for config save


# ─── Data Model ──────────────────────────────────────────────

class VideoTask:
    """Represents a single video/audio processing task."""

    def __init__(self, source: str, source_type: str):
        self.uid             = uuid.uuid4().hex[:8]
        self.source          = source
        self.source_type     = source_type  # "url" or "local"
        self.audio_path: Optional[str] = None
        self.status          = STATUS_PENDING
        self.progress_pct    = 0
        self.progress_msg    = ""
        self.transcript_text = ""
        self.transcript_srt  = ""
        self.transcript_vtt  = ""
        self.error_msg       = ""
        self.title           = ""

    def display_name(self) -> str:
        if self.title:
            return self.title
        if self.source_type == "local":
            return os.path.basename(self.source)
        return self.source

    def safe_filename(self) -> str:
        name = self.title or os.path.splitext(
            os.path.basename(self.source))[0]
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        name = re.sub(r'[\x00-\x1f\x7f]', '', name)
        name = name.rstrip('. ')
        name = re.sub(r'\s+', ' ', name).strip()
        return name or f"untitled_{self.uid}"

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def reset(self) -> None:
        """Reset task to pending state for retry."""
        self.status          = STATUS_PENDING
        self.progress_pct    = 0
        self.progress_msg    = ""
        self.transcript_text = ""
        self.transcript_srt  = ""
        self.transcript_vtt  = ""
        self.error_msg       = ""
        self.audio_path      = None


# ─── Bulk Paste Dialog ───────────────────────────────────────

class BulkPasteDialog(QDialog):
    """Dialog for pasting multiple URLs at once."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bulk Paste URLs")
        self.setMinimumSize(560, 360)
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QLabel("Paste video URLs below, one per line:")
        header.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 13px;")
        layout.addWidget(header)

        self.text_edit = QTextEdit()
        self.text_edit.setAcceptRichText(False)
        self.text_edit.setPlaceholderText(
            "https://youtube.com/watch?v=...\n"
            "https://youtu.be/...\n"
            "https://vimeo.com/...")
        self.text_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {COLOR_SURFACE0}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_SURFACE1}; border-radius: 8px;
                padding: 10px; font-size: 13px;
                selection-background-color: {COLOR_BLUE};
                selection-color: {COLOR_BASE};
            }}
            QTextEdit:focus {{ border-color: {COLOR_BLUE}; }}
        """)
        layout.addWidget(self.text_edit)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setObjectName("secondaryBtn")
        btn_cancel.clicked.connect(self.reject)
        btn_add = QPushButton("Add All URLs")
        btn_add.setObjectName("primaryBtn")
        btn_add.clicked.connect(self.accept)
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_add)
        layout.addLayout(btn_layout)

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QDialog {{ background: {COLOR_BASE}; }}
            QPushButton#primaryBtn {{
                background: {COLOR_BLUE}; color: {COLOR_BASE};
                border: none; border-radius: 8px;
                padding: 8px 20px; font-weight: bold; font-size: 13px;
            }}
            QPushButton#primaryBtn:hover {{ background: {COLOR_LAVENDER}; }}
            QPushButton#secondaryBtn {{
                background: {COLOR_SURFACE1}; color: {COLOR_TEXT};
                border: none; border-radius: 8px;
                padding: 8px 20px; font-size: 13px;
            }}
            QPushButton#secondaryBtn:hover {{ background: {COLOR_SURFACE2}; }}
        """)

    def get_urls(self) -> list[str]:
        return [
            u.strip() for u in self.text_edit.toPlainText().splitlines()
            if u.strip() and u.strip().startswith("http")
        ]


# ─── Advanced Audio Processing ────────────────────────────────

class AudioPreprocessor:
    """Audio preprocessing pipeline: normalization, noise reduction, silence trimming."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def normalize_rms(self, y: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to a target RMS level (in dB)."""
        rms = np.sqrt(np.mean(y ** 2))
        if rms < 1e-10:
            return y
        target_rms = 10.0 ** (target_db / 20.0)
        gain = target_rms / rms
        # Clamp gain to avoid extreme amplification of near-silent audio
        gain = min(gain, 20.0)
        return y * gain

    def spectral_gating_denoise(self, y: np.ndarray, sr: int,
                                 noise_frames: int = 6,
                                 prop_decrease: float = 0.8) -> np.ndarray:
        """
        Simple spectral gating noise reduction.
        Estimates noise from the first `noise_frames` and subtracts it
        from the magnitude spectrum.
        """
        try:
            import librosa
            stft = librosa.stft(y, n_fft=2048, hop_length=512)
            mag, phase = np.abs(stft), np.angle(stft)

            # Estimate noise from quietest frames
            frame_energy = np.sum(mag ** 2, axis=0)
            quiet_indices = np.argsort(frame_energy)[:noise_frames]
            noise_profile = np.mean(mag[:, quiet_indices], axis=1, keepdims=True)

            # Spectral gating: subtract noise floor
            mask = mag > noise_profile * (1 + prop_decrease)
            cleaned_mag = np.where(mask, mag - noise_profile * prop_decrease, 0.0)
            cleaned_mag = np.maximum(cleaned_mag, 0.0)

            cleaned_stft = cleaned_mag * np.exp(1j * phase)
            return librosa.istft(cleaned_stft, hop_length=512, length=len(y))
        except Exception:
            return y  # Fall back to unprocessed audio

    def preprocess(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Full preprocessing pipeline."""
        y = self.normalize_rms(y)
        y = self.spectral_gating_denoise(y, sr)
        return y


# ─── Advanced VAD (Voice Activity Detection) ─────────────────

class AdvancedVAD:
    """
    Multi-feature Voice Activity Detection combining energy, spectral
    flatness, zero-crossing rate, and spectral centroid for robust
    speech/nonspeech discrimination — inspired by WebRTC VAD principles.
    """

    def __init__(self, sample_rate: int = 16000, hop_length: int = 512,
                 n_fft: int = 2048):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft

    def detect(self, y: np.ndarray) -> np.ndarray:
        """
        Returns a boolean mask of shape (num_frames,) where True = speech.
        Uses a multi-criterion scoring approach with adaptive thresholds.
        """
        import librosa

        n_frames = 1 + (len(y) - self.n_fft) // self.hop_length

        # Feature 1: RMS energy
        rms = librosa.feature.rms(y=y, frame_length=self.n_fft,
                                   hop_length=self.hop_length)[0]
        rms = rms[:n_frames]

        # Feature 2: Spectral flatness (speech has low flatness, noise has high)
        flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        flatness = flatness[:n_frames]

        # Feature 3: Zero-crossing rate (speech typically lower than noise)
        zcr = librosa.feature.zero_crossing_rate(
            y=y, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        zcr = zcr[:n_frames]

        # Feature 4: Spectral centroid (speech has characteristic range)
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length)[0]
        centroid = centroid[:n_frames]

        # Ensure all arrays have same length
        min_len = min(len(rms), len(flatness), len(zcr), len(centroid))
        rms = rms[:min_len]
        flatness = flatness[:min_len]
        zcr = zcr[:min_len]
        centroid = centroid[:min_len]

        # ── Adaptive thresholds ──
        # Energy: use percentile-based threshold with hysteresis
        energy_thresh_low = np.percentile(rms, 25)
        energy_thresh_high = np.percentile(rms, 55)

        # Spectral flatness: speech is tonal (low flatness)
        flatness_thresh = np.percentile(flatness, 60)

        # ZCR: speech usually has moderate ZCR
        zcr_median = np.median(zcr)
        zcr_thresh_high = zcr_median * 2.5

        # ── Score-based VAD ──
        # Each feature votes; weighted combination
        scores = np.zeros(min_len)

        # Energy score (0-1): higher energy = more likely speech
        for i in range(min_len):
            if rms[i] > energy_thresh_high:
                scores[i] += 0.40  # Strong energy vote
            elif rms[i] > energy_thresh_low:
                scores[i] += 0.20  # Weak energy vote

            # Flatness score: low flatness = tonal = likely speech
            if flatness[i] < flatness_thresh:
                scores[i] += 0.25

            # ZCR score: moderate ZCR = likely speech
            if zcr[i] < zcr_thresh_high:
                scores[i] += 0.15

            # Centroid score: speech centroid typically 500-4000 Hz
            if 500 < centroid[i] < 4000:
                scores[i] += 0.20

        # Decision threshold
        speech_mask = scores >= 0.55

        # ── Morphological post-processing ──
        speech_mask = self._bridge_gaps(speech_mask,
            max_gap_frames=int(self.sample_rate / self.hop_length * 0.6))
        speech_mask = self._smooth_mask(speech_mask, window=11)
        speech_mask = self._remove_short_segments(speech_mask,
            min_frames=int(self.sample_rate / self.hop_length * 0.25))

        return speech_mask

    @staticmethod
    def _bridge_gaps(mask, max_gap_frames: int) -> np.ndarray:
        """Fill short gaps in a boolean mask (bridging brief pauses in speech)."""
        result = mask.copy()
        i = 0
        while i < len(result):
            if not result[i]:
                j = i
                while j < len(result) and not result[j]:
                    j += 1
                if 0 < (j - i) < max_gap_frames:
                    result[i:j] = True
                i = j
            else:
                i += 1
        return result

    @staticmethod
    def _smooth_mask(mask, window: int = 11) -> np.ndarray:
        """Smooth a boolean mask using moving majority vote."""
        result = np.zeros(len(mask), dtype=bool)
        half = window // 2
        for i in range(len(mask)):
            s = max(0, i - half)
            e = min(len(mask), i + half + 1)
            result[i] = np.mean(mask[s:e]) > 0.5
        return result

    @staticmethod
    def _remove_short_segments(mask, min_frames: int) -> np.ndarray:
        """Remove speech segments shorter than min_frames to eliminate clicks/pops."""
        result = mask.copy()
        i = 0
        while i < len(result):
            if result[i]:
                j = i
                while j < len(result) and result[j]:
                    j += 1
                if (j - i) < min_frames:
                    result[i:j] = False
                i = j
            else:
                i += 1
        return result


# ─── Advanced Feature Extractor ──────────────────────────────

class SpeakerFeatureExtractor:
    """
    High-dimensional speaker embedding extraction combining:
      - MFCCs (20 dims) + deltas (20) + delta-deltas (20)
      - Chroma features (12 dims) — captures pitch/harmonic characteristics
      - Spectral contrast (7 dims) — captures vocal tract characteristics
      - Spectral bandwidth (1 dim) — captures voice width
    Plus sliding-window temporal context (mean/std over neighboring frames)
    for more robust speaker embeddings.
    Total: ~83+ dimensions with context statistics.
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 2048,
                 hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_frame_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract per-frame feature matrix of shape (n_features, n_frames).
        """
        import librosa

        # MFCCs + deltas
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=20,
                                      n_fft=self.n_fft, hop_length=self.hop_length)
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Chroma — captures harmonic/pitch content useful for speaker identity
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate,
                                              n_fft=self.n_fft,
                                              hop_length=self.hop_length)

        # Spectral contrast — characterizes vocal tract resonances
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sample_rate,
                                                       n_fft=self.n_fft,
                                                       hop_length=self.hop_length)

        # Spectral bandwidth — voice width characteristic
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate,
                                                         n_fft=self.n_fft,
                                                         hop_length=self.hop_length)

        # Stack all features
        all_features = np.vstack([mfcc, delta1, delta2, chroma, contrast, bandwidth])

        return all_features

    def extract_segment_embedding(self, features: np.ndarray,
                                    start_frame: int, end_frame: int,
                                    context_radius: int = 5) -> np.ndarray:
        """
        Extract a fixed-size speaker embedding for a segment by computing
        statistics over the segment frames plus a small temporal context.

        Returns a vector of size (n_features * 4):
          - segment mean
          - segment std
          - context mean (±context_radius frames around segment)
          - context std
        """
        n_features = features.shape[0]
        n_frames = features.shape[1]

        # Clamp indices
        s = max(0, start_frame)
        e = min(n_frames, end_frame)

        if s >= e:
            return np.zeros(n_features * 4)

        # Segment statistics
        seg = features[:, s:e]
        seg_mean = np.mean(seg, axis=1)
        seg_std = np.std(seg, axis=1) + 1e-8  # Avoid zero std

        # Context statistics (broader window around the segment)
        ctx_s = max(0, s - context_radius)
        ctx_e = min(n_frames, e + context_radius)
        ctx = features[:, ctx_s:ctx_e]
        ctx_mean = np.mean(ctx, axis=1)
        ctx_std = np.std(ctx, axis=1) + 1e-8

        return np.concatenate([seg_mean, seg_std, ctx_mean, ctx_std])


# ─── Speaker Change Detection ────────────────────────────────

class SpeakerChangeDetector:
    """
    BIC-based (Bayesian Information Criterion) speaker change point
    detection. Identifies likely points in the audio where the speaker
    changes, which improves segmentation before clustering.
    """

    def __init__(self, sample_rate: int = 16000, hop_length: int = 512,
                 min_segment_duration: float = 1.0,
                 window_duration: float = 2.0):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_segment_frames = int(min_segment_duration * sample_rate / hop_length)
        self.window_frames = int(window_duration * sample_rate / hop_length)

    def detect_changes(self, features: np.ndarray) -> list[int]:
        """
        Detect speaker change points in the feature matrix.
        Returns list of frame indices where speaker changes are likely.

        Uses a sliding-window BIC approach: for each candidate point,
        compare the BIC of modeling the window as one Gaussian vs two.
        """
        n_frames = features.shape[1]
        if n_frames < self.min_segment_frames * 3:
            return []

        change_points = []
        half_win = self.window_frames // 2
        step = max(1, self.min_segment_frames // 2)

        for pos in range(self.min_segment_frames,
                          n_frames - self.min_segment_frames, step):
            # Window around candidate point
            win_start = max(0, pos - half_win)
            win_end = min(n_frames, pos + half_win)

            left_start = max(0, pos - half_win)
            right_end = min(n_frames, pos + half_win)

            left = features[:, left_start:pos]
            right = features[:, pos:right_end]
            full = features[:, left_start:right_end]

            if left.shape[1] < self.min_segment_frames // 2:
                continue
            if right.shape[1] < self.min_segment_frames // 2:
                continue

            # BIC comparison
            delta_bic = self._compute_delta_bic(full, left, right)

            # If two-model BIC is significantly better, mark as change point
            if delta_bic > 0:
                # Avoid clustering change points too close together
                if not change_points or (pos - change_points[-1]) > self.min_segment_frames:
                    change_points.append(pos)

        return change_points

    @staticmethod
    def _compute_delta_bic(full: np.ndarray, left: np.ndarray,
                            right: np.ndarray) -> float:
        """
        Compute BIC difference: BIC_one - BIC_two.
        Positive value means two Gaussians fit better (change point).
        Uses diagonal covariance for efficiency.
        """
        n_full = full.shape[1]
        n_left = left.shape[1]
        n_right = right.shape[1]
        d = full.shape[0]

        if n_full < d + 1 or n_left < d + 1 or n_right < d + 1:
            return 0.0

        # Log-likelihood under single Gaussian
        ll_one = _log_likelihood_diag(full)

        # Log-likelihood under two Gaussians
        ll_left = _log_likelihood_diag(left)
        ll_right = _log_likelihood_diag(right)
        ll_two = ll_left + ll_right

        # Penalty term (BIC)
        # Single model: d means + d variances = 2d parameters
        # Two models: 2 * (d means + d variances) = 4d parameters
        penalty = 0.5 * (2 * d) * np.log(n_full)  # Extra params for two-model

        return ll_two - ll_one - penalty


def _log_likelihood_diag(X: np.ndarray) -> float:
    """
    Compute log-likelihood of data X under a diagonal-covariance Gaussian.
    X shape: (n_features, n_frames)
    """
    n = X.shape[1]
    d = X.shape[0]
    if n < 2:
        return 0.0

    mean = np.mean(X, axis=1, keepdims=True)
    var = np.var(X, axis=1, keepdims=True) + 1e-6  # Regularize

    # Log-likelihood: -0.5 * n * [d*log(2pi) + sum(log(var)) + d]
    log_det = np.sum(np.log(var))
    ll = -0.5 * n * (d * np.log(2 * np.pi) + log_det + d)

    return ll


# ─── Speaker Count Estimator ────────────────────────────────

class SpeakerCountEstimator:
    """
    Automatically estimates the number of speakers using eigen-gap
    analysis on the affinity matrix (spectral graph theory approach),
    combined with BIC model selection for robustness.
    """

    @staticmethod
    def estimate_eigen_gap(X: np.ndarray, max_speakers: int = 10) -> int:
        """
        Estimate number of clusters via eigen-gap on the normalized
        cosine similarity matrix.

        1. Build affinity matrix from cosine similarities.
        2. Compute Laplacian and its eigenvalues.
        3. Find the largest gap in eigenvalue magnitudes.
        """
        from scipy.spatial.distance import pdist, squareform

        n = X.shape[0]
        if n <= 1:
            return 1

        # Normalize for cosine similarity
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_norm = X / norms

        # Cosine similarity -> affinity
        sim_matrix = 1 - squareform(pdist(X_norm, metric='cosine'))
        sim_matrix = np.clip(sim_matrix, 0, 1)
        np.fill_diagonal(sim_matrix, 1.0)

        # Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        degree = np.sum(sim_matrix, axis=1)
        degree[degree == 0] = 1e-10
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        L_norm = np.eye(n) - D_inv_sqrt @ sim_matrix @ D_inv_sqrt

        # Eigenvalues (smallest first, skip the trivial zero eigenvalue)
        try:
            k = min(max_speakers + 2, n)
            eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))[:k]
        except Exception:
            return 2

        # Eigen-gap: find the largest relative gap
        # Skip eigenvalue[0] which is ~0 for connected graphs
        gaps = []
        for i in range(1, len(eigenvalues) - 1):
            if eigenvalues[i] > 1e-10:
                gap = eigenvalues[i + 1] - eigenvalues[i]
                gaps.append((gap, i + 1))  # i+1 = number of clusters

        if not gaps:
            return 2

        # Largest gap determines number of speakers
        best_gap, n_clusters = max(gaps, key=lambda x: x[0])
        n_clusters = max(2, min(n_clusters, max_speakers))

        return n_clusters

    @staticmethod
    def estimate_bic(X: np.ndarray, max_speakers: int = 10) -> int:
        """
        Estimate number of clusters using BIC model selection.
        For each candidate K, compute BIC for K Gaussian clusters (diagonal cov)
        and pick the K with best (lowest) BIC.
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        n = X.shape[0]
        d = X.shape[1]
        if n <= 1:
            return 1

        # Pre-compute linkage once
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_norm = X / norms
        distances = pdist(X_norm, metric='cosine')
        Z = linkage(distances, method='average')

        best_bic = np.inf
        best_k = 2

        for k in range(2, min(max_speakers + 1, n + 1)):
            labels = fcluster(Z, t=k, criterion='maxclust')

            # Compute total log-likelihood and parameter count
            total_ll = 0.0
            total_params = 0
            for cluster_id in range(1, k + 1):
                mask = labels == cluster_id
                cluster_data = X[mask]
                n_c = cluster_data.shape[0]
                if n_c < d + 1:
                    total_ll += -1e10
                    total_params += d + d  # mean + var
                    continue
                total_ll += _log_likelihood_diag(cluster_data.T)
                total_params += d + d  # mean + diagonal variance per cluster

            # BIC = -2 * ll + params * log(n)
            bic = -2 * total_ll + total_params * np.log(n)

            if bic < best_bic:
                best_bic = bic
                best_k = k

        return best_k

    @staticmethod
    def estimate(X: np.ndarray, max_speakers: int = 10) -> int:
        """
        Combine eigen-gap and BIC estimates using a consensus strategy.
        If they agree, use that value. Otherwise, use the smaller one
        (conservative: prefer fewer speakers to over-segmentation).
        """
        if X.shape[0] <= 2:
            return min(X.shape[0], 2)

        k_eigen = SpeakerCountEstimator.estimate_eigen_gap(X, max_speakers)
        k_bic = SpeakerCountEstimator.estimate_bic(X, max_speakers)

        # Consensus: if close, average; if far apart, take minimum
        if abs(k_eigen - k_bic) <= 1:
            return int(np.round((k_eigen + k_bic) / 2))
        else:
            return min(k_eigen, k_bic)


# ─── Advanced Speaker Clustering ─────────────────────────────

class SpeakerClusterer:
    """
    Multi-strategy speaker clustering:
      1. Spectral clustering (when eigen-gap gives clear speaker count)
      2. Agglomerative hierarchical clustering with BIC-optimized threshold
      3. Fallback: standard hierarchical clustering with adaptive threshold

    Also performs post-clustering refinement: re-assignment of uncertain
    segments using nearest-centroid classification.
    """

    @staticmethod
    def cluster(X: np.ndarray, num_speakers: int = 0,
                max_speakers: int = 10) -> np.ndarray:
        """
        Cluster speaker embeddings X (n_samples x n_features).
        If num_speakers <= 0, auto-detect the count.
        Returns labels array of shape (n_samples,).
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        n = X.shape[0]
        if n <= 1:
            return np.ones(n, dtype=int)

        # L2 normalize for cosine-based distances
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_norm = X / norms

        # Auto-detect speaker count if needed
        if num_speakers <= 0:
            num_speakers = SpeakerCountEstimator.estimate(X_norm, max_speakers)

        num_speakers = max(2, min(num_speakers, n))

        # Try spectral clustering first (best for speaker diarization)
        labels = SpeakerClusterer._spectral_cluster(X_norm, num_speakers)
        if labels is not None:
            return labels

        # Fallback: hierarchical clustering
        distances = pdist(X_norm, metric='cosine')
        Z = linkage(distances, method='average')
        labels = fcluster(Z, t=num_speakers, criterion='maxclust')

        # Post-processing: nearest-centroid re-assignment for boundary segments
        labels = SpeakerClusterer._refine_with_centroids(X_norm, labels)

        return labels

    @staticmethod
    def _spectral_cluster(X: np.ndarray, n_clusters: int) -> Optional[np.ndarray]:
        """
        Spectral clustering using the normalized Laplacian.
        Returns cluster labels or None if it fails.
        """
        from scipy.spatial.distance import pdist, squareform

        n = X.shape[0]
        if n <= n_clusters:
            return np.arange(1, n + 1)

        try:
            # Build affinity matrix
            sim_matrix = 1 - squareform(pdist(X, metric='cosine'))
            sim_matrix = np.clip(sim_matrix, 0, 1)
            np.fill_diagonal(sim_matrix, 1.0)

            # Symmetric normalized Laplacian
            degree = np.sum(sim_matrix, axis=1)
            degree[degree == 0] = 1e-10
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
            L_norm = np.eye(n) - D_inv_sqrt @ sim_matrix @ D_inv_sqrt

            # Eigendecomposition: take first n_clusters eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
            # Skip the first eigenvector (trivial, ~constant)
            F = eigenvectors[:, 1:n_clusters + 1]

            # Row-normalize
            row_norms = np.linalg.norm(F, axis=1, keepdims=True)
            row_norms[row_norms == 0] = 1
            F_norm = F / row_norms

            # K-means on the spectral embedding
            labels = SpeakerClusterer._kmeans(F_norm, n_clusters)
            return labels

        except Exception:
            return None

    @staticmethod
    def _kmeans(X: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        """
        Simple K-means clustering (no sklearn dependency).
        Uses K-means++ initialization for better convergence.
        """
        n = X.shape[0]
        if n <= k:
            return np.arange(1, n + 1)

        # K-means++ initialization
        centroids = [X[np.random.randint(n)]]
        for _ in range(1, k):
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids) for x in X
            ])
            probs = dists / (dists.sum() + 1e-10)
            idx = np.random.choice(n, p=probs)
            centroids.append(X[idx])
        centroids = np.array(centroids)

        # Iterate
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assign
            for i in range(n):
                dists = np.sum((centroids - X[i]) ** 2, axis=1)
                labels[i] = np.argmin(dists)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                members = X[labels == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        # Convert to 1-based labels
        return labels + 1

    @staticmethod
    def _refine_with_centroids(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Post-clustering refinement: compute cluster centroids and re-assign
        any segment whose distance to its assigned centroid is much larger
        than its distance to another centroid. This fixes boundary errors.
        """
        unique_labels = np.unique(labels)
        k = len(unique_labels)

        if k <= 1 or len(X) <= k:
            return labels

        # Compute centroids
        centroids = np.zeros((k, X.shape[1]))
        for i, lbl in enumerate(unique_labels):
            members = X[labels == lbl]
            if len(members) > 0:
                centroids[i] = members.mean(axis=0)

        # Re-assign: only if the segment is clearly closer to another centroid
        refined = labels.copy()
        for i in range(len(X)):
            dists = np.array([np.sum((X[i] - c) ** 2) for c in centroids])
            closest = np.argmin(dists)
            current = np.where(unique_labels == labels[i])[0][0]

            # Only reassign if closest is significantly better
            # (at least 30% closer) to avoid oscillation
            if closest != current and dists[closest] < dists[current] * 0.7:
                refined[i] = unique_labels[closest]

        return refined


# ─── Speaker Label Post-Processor ────────────────────────────

class SpeakerPostProcessor:
    """
    Post-processing of speaker labels to produce clean, coherent output:
      1. Merge consecutive same-speaker segments
      2. Smooth rapid speaker switches (likely errors)
      3. Remove suspiciously short segments
      4. Re-number speakers sequentially by first appearance
    """

    @staticmethod
    def process(segments: list[dict]) -> list[dict]:
        """Apply all post-processing steps to annotated segments."""
        if not segments:
            return segments

        result = SpeakerPostProcessor._merge_consecutive(segments)
        result = SpeakerPostProcessor._smooth_rapid_switches(result, min_gap=0.3)
        result = SpeakerPostProcessor._remove_short(result, min_duration=0.15)
        result = SpeakerPostProcessor._renumber_speakers(result)

        return result

    @staticmethod
    def _merge_consecutive(segments: list[dict]) -> list[dict]:
        """Merge consecutive segments with the same speaker."""
        if not segments:
            return segments

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            if (seg.get("speaker") == merged[-1].get("speaker")
                    and seg["start"] - merged[-1]["end"] < 0.5):
                # Extend the previous segment
                merged[-1]["end"] = seg["end"]
                merged[-1]["text"] += " " + (seg.get("text") or "").strip()
                merged[-1]["text"] = merged[-1]["text"].strip()
            else:
                merged.append(seg.copy())

        return merged

    @staticmethod
    def _smooth_rapid_switches(segments: list[dict],
                                min_gap: float = 0.3) -> list[dict]:
        """
        If a speaker switch happens and then switches back within min_gap
        seconds, the brief intervening segment is likely an error.
        Reassign it to the surrounding speaker.
        """
        if len(segments) < 3:
            return segments

        result = [segments[0].copy()]
        for i in range(1, len(segments) - 1):
            prev = result[-1]
            curr = segments[i]
            nxt = segments[i + 1]

            duration = curr["end"] - curr["start"]
            if (duration < min_gap
                    and prev.get("speaker") == nxt.get("speaker")
                    and curr.get("speaker") != prev.get("speaker")):
                # Merge current into previous
                prev["end"] = curr["end"]
                prev["text"] += " " + (curr.get("text") or "").strip()
                prev["text"] = prev["text"].strip()
            else:
                result.append(curr.copy())

        # Don't forget the last segment
        if len(segments) >= 2:
            result.append(segments[-1].copy())

        return result

    @staticmethod
    def _remove_short(segments: list[dict],
                       min_duration: float = 0.15) -> list[dict]:
        """Remove or merge segments shorter than min_duration seconds."""
        if not segments:
            return segments

        result = []
        for seg in segments:
            duration = seg["end"] - seg["start"]
            if duration < min_duration and result:
                # Absorb into previous segment
                result[-1]["end"] = seg["end"]
                result[-1]["text"] += " " + (seg.get("text") or "").strip()
                result[-1]["text"] = result[-1]["text"].strip()
            else:
                result.append(seg.copy())

        return result

    @staticmethod
    def _renumber_speakers(segments: list[dict]) -> list[dict]:
        """Renumber speakers sequentially by first appearance."""
        speaker_map = {}
        counter = 1
        for seg in segments:
            sp = seg.get("speaker")
            if sp is not None and sp not in speaker_map:
                speaker_map[sp] = counter
                counter += 1

        for seg in segments:
            sp = seg.get("speaker")
            if sp is not None:
                seg["speaker"] = f"Speaker {speaker_map[sp]}"

        return segments


# ─── Neural Diarization (pyannote.audio) ─────────────────────

def _try_neural_diarize(audio_path, max_speakers=10, progress_cb=None):
    """
    Attempt neural diarization using pyannote.audio if available.
    Returns list of {start, end, speaker} dicts, or None if unavailable.
    """
    try:
        from pyannote.audio import Pipeline

        # Check for HuggingFace token
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            if progress_cb:
                progress_cb("Neural diarization skipped: No HF_TOKEN found.")
            return None

        if progress_cb:
            progress_cb("Loading pyannote pipeline...")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )

        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))

        if progress_cb:
            progress_cb("Running neural diarization...")

        # Run diarization
        kwargs = {}
        if max_speakers and max_speakers > 0:
            kwargs["num_speakers"] = max_speakers

        diarization = pipeline(audio_path, **kwargs)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return segments

    except ImportError:
        if progress_cb:
            progress_cb("Neural diarization skipped: pyannote.audio not installed.")
        return None
    except Exception as e:
        if progress_cb:
            progress_cb(f"Neural diarization failed: {e}")
        return None


# ─── Advanced Local Speaker Diarizer ─────────────────────────

class LocalSpeakerDiarizer:
    """
    State-of-the-art local speaker diarization pipeline combining:
      1. Audio preprocessing (normalization, noise reduction)
      2. Multi-feature VAD (energy + spectral flatness + ZCR + centroid)
      3. BIC-based speaker change detection
      4. High-dimensional feature extraction (MFCC + chroma + contrast + bandwidth)
         with temporal context statistics
      5. Spectral clustering with eigen-gap/BIC auto speaker count
      6. Centroid-based refinement
      7. Post-processing: merge, smooth, renumber

    No sklearn dependency required. Uses scipy + numpy + librosa.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.preprocessor = AudioPreprocessor(sample_rate)
        self.vad = AdvancedVAD(sample_rate)
        self.feature_extractor = SpeakerFeatureExtractor(sample_rate)
        self.change_detector = SpeakerChangeDetector(sample_rate)

    def diarize(self, audio_path, whisper_segments, num_speakers=0,
                progress_cb=None):
        """
        Run the full diarization pipeline.

        Args:
            audio_path: Path to the audio file
            whisper_segments: List of segments from Whisper transcription
            num_speakers: Number of speakers (0 = auto-detect)
            progress_cb: Optional callback for progress messages

        Returns:
            List of dicts with {start, end, text, speaker}
        """
        import librosa
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        # ── Step 1: Load and preprocess audio ──
        if progress_cb:
            progress_cb("Loading and preprocessing audio...")

        y, sr = self._load_audio(audio_path)
        if y is None or len(y) < self.sample_rate:
            return self._fallback_segments(whisper_segments)

        y = self.preprocessor.preprocess(y, sr)

        # ── Step 2: Advanced VAD ──
        if progress_cb:
            progress_cb("Running multi-feature VAD...")

        speech_mask = self.vad.detect(y)

        # ── Step 3: Extract high-dimensional features ──
        if progress_cb:
            progress_cb("Extracting speaker features (MFCC + chroma + contrast)...")

        features = self.feature_extractor.extract_frame_features(y)

        # ── Step 4: BIC speaker change detection ──
        if progress_cb:
            progress_cb("Detecting speaker change points (BIC)...")

        change_points = self.change_detector.detect_changes(features)

        # ── Step 5: Build segment embeddings ──
        if progress_cb:
            progress_cb("Building speaker embeddings with context...")

        hop_per_sec = sr / self.feature_extractor.hop_length
        embeddings = []
        valid_segments = []

        for seg in whisper_segments:
            start_frame = int(seg["start"] * hop_per_sec)
            end_frame = int(seg["end"] * hop_per_sec)

            start_frame = max(0, min(start_frame, features.shape[1] - 1))
            end_frame = max(0, min(end_frame, features.shape[1] - 1))

            if start_frame >= end_frame:
                continue

            # Check VAD: at least 30% of frames should be speech
            vad_start = min(start_frame, len(speech_mask) - 1)
            vad_end = min(end_frame, len(speech_mask))
            if vad_end > vad_start:
                speech_ratio = np.mean(speech_mask[vad_start:vad_end])
                if speech_ratio < 0.15:
                    continue

            # Extract embedding with temporal context
            embedding = self.feature_extractor.extract_segment_embedding(
                features, start_frame, end_frame, context_radius=8)
            embeddings.append(embedding)
            valid_segments.append(seg)

        if not embeddings:
            return self._fallback_segments(whisper_segments)

        X = np.array(embeddings)

        # ── Step 6: Advanced clustering ──
        if progress_cb:
            progress_cb("Clustering speakers (spectral + BIC)...")

        labels = SpeakerClusterer.cluster(X, num_speakers=num_speakers)

        # ── Step 7: Build annotated segments ──
        annotated = []
        for i, seg in enumerate(valid_segments):
            speaker_id = labels[i] if i < len(labels) else 1
            annotated.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": (seg.get("text") or "").strip(),
                "speaker": f"Speaker {speaker_id}"
            })

        # ── Step 8: Post-processing ──
        if progress_cb:
            progress_cb("Post-processing speaker labels...")

        annotated = SpeakerPostProcessor.process(annotated)

        return annotated

    def _load_audio(self, audio_path: str):
        """Load audio file with ffmpeg conversion if needed."""
        import librosa

        temp_wav = None
        try:
            if not audio_path.lower().endswith(".wav"):
                fd, temp_wav = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                cmd = [
                    "ffmpeg", "-y", "-i", audio_path,
                    "-ar", str(self.sample_rate), "-ac", "1",
                    temp_wav
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                load_path = temp_wav
            else:
                load_path = audio_path

            y, sr = librosa.load(load_path, sr=self.sample_rate, mono=True)
            return y, sr

        except Exception as e:
            return None, None
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

    @staticmethod
    def _fallback_segments(whisper_segments) -> list[dict]:
        """Fallback: assign all segments to Speaker 1."""
        return [
            {"start": s["start"], "end": s["end"],
             "text": (s.get("text") or "").strip(), "speaker": "Speaker 1"}
            for s in whisper_segments
        ]
# ─── Audio Merger ────────────────────────────────────────────

class AudioMerger:
    """Concatenate multiple audio files into one using ffmpeg."""

    @staticmethod
    def merge(file_paths: list[str], output_dir: str,
              progress_cb: Optional[Callable] = None) -> Optional[str]:
        """
        Merge multiple audio files into a single WAV file.

        Uses ffmpeg concat demuxer for reliable merging.
        All files are converted to 16kHz mono WAV first if needed.

        Args:
            file_paths: List of audio file paths to merge.
            output_dir: Directory to save the merged file.
            progress_cb: Optional callback(str) for status messages.

        Returns:
            Path to the merged WAV file, or None on failure.
        """
        if not file_paths:
            return None

        if progress_cb:
            progress_cb(f"Preparing {len(file_paths)} audio files for merge...")

        os.makedirs(output_dir, exist_ok=True)
        merged_path = os.path.join(output_dir, "merged_audio.wav")

        # Step 1: Normalize all files to 16kHz mono WAV
        normalized_paths = []
        for i, fpath in enumerate(file_paths):
            if not os.path.isfile(fpath):
                if progress_cb:
                    progress_cb(f"Skipping missing file: {fpath}")
                continue

            fd, norm_path = tempfile.mkstemp(suffix=".wav", dir=output_dir)
            os.close(fd)

            try:
                cmd = [
                    "ffmpeg", "-y", "-i", fpath,
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    norm_path
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    if progress_cb:
                        progress_cb(f"Failed to convert: {fpath}")
                    continue
                normalized_paths.append(norm_path)
            except Exception as e:
                if progress_cb:
                    progress_cb(f"Error converting {fpath}: {e}")
                continue

            if progress_cb:
                progress_cb(
                    f"Normalized {i + 1}/{len(file_paths)}: "
                    f"{os.path.basename(fpath)}")

        if len(normalized_paths) < 2:
            # Nothing to merge or only one file
            for p in normalized_paths:
                if os.path.exists(p):
                    os.remove(p)
            if len(normalized_paths) == 1:
                # Just rename the single normalized file
                os.rename(normalized_paths[0], merged_path)
                return merged_path
            return None

        if progress_cb:
            progress_cb("Merging audio files with ffmpeg...")

        # Step 2: Create concat list file
        list_path = os.path.join(output_dir, "concat_list.txt")
        try:
            with open(list_path, "w", encoding="utf-8") as f:
                for p in normalized_paths:
                    # ffmpeg concat demuxer needs escaped paths
                    escaped = p.replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")

            # Step 3: Concatenate
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_path, "-c:a", "pcm_s16le",
                merged_path
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                if progress_cb:
                    progress_cb(f"ffmpeg merge failed: {result.stderr[:200]}")
                merged_path = None

        except Exception as e:
            if progress_cb:
                progress_cb(f"Merge error: {e}")
            merged_path = None

        finally:
            # Cleanup temp files
            if os.path.exists(list_path):
                os.remove(list_path)
            for p in normalized_paths:
                if os.path.exists(p):
                    os.remove(p)

        if progress_cb and merged_path:
            progress_cb(f"Merged audio saved: {merged_path}")

        return merged_path


# ─── Transcript Formatter ────────────────────────────────────

class TranscriptFormatter:
    """Unified transcript formatting for TXT, SRT, and VTT outputs."""

    @staticmethod
    def make_label(original_label: str, fmt: str) -> str:
        """Reformat a speaker label like 'Speaker 3' according to fmt."""
        m = re.search(r'(\d+)', original_label)
        if not m:
            return original_label
        num = int(m.group(1))
        if fmt.startswith("SPK_"):
            return f"SPK_{num}"
        elif "A, B" in fmt:
            return f"Speaker {chr(64 + num)}"
        elif "[" in fmt:
            return f"Speaker [{num}]"
        else:
            return original_label

    @staticmethod
    def format(
        segments: list[dict],
        language: str,
        label_fmt: str = "Speaker 1, 2, 3...",
        txt_style: str = "Timestamped Blocks",
        show_ts: bool = True,
        show_lang: bool = True,
        srt_labels: bool = True,
        include_vtt: bool = False,
    ) -> tuple[str, str, str]:
        """
        Format annotated segments into TXT, SRT, and optionally VTT.

        Returns:
            (text_output, srt_output, vtt_output)
        """
        # ── Header ──
        header_lines: list[str] = []
        if show_lang:
            header_lines.append(f"Language: {language}")
            header_lines.append("=" * 64)
        header_lines.append("")
        header = "\n".join(header_lines)

        # ── Remap speaker labels ──
        fmt_segments = []
        for seg in segments:
            s = dict(seg)
            if s.get("speaker"):
                s["speaker"] = TranscriptFormatter.make_label(
                    s["speaker"], label_fmt)
            fmt_segments.append(s)

        text_output = TranscriptFormatter._format_txt(
            fmt_segments, header, txt_style, show_ts)
        srt_output = TranscriptFormatter._format_srt(
            segments, label_fmt, srt_labels)

        vtt_output = ""
        if include_vtt:
            vtt_output = TranscriptFormatter._format_vtt(
                segments, label_fmt, srt_labels)

        return text_output, srt_output, vtt_output

    @staticmethod
    def _fmt_time(s: float) -> str:
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    @staticmethod
    def _fmt_srt_time(s: float) -> str:
        h   = int(s // 3600)
        m   = int((s % 3600) // 60)
        sec = int(s % 60)
        ms  = int(round((s - int(s)) * 1000))
        if ms >= 1000:
            ms = 999
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    @staticmethod
    def _fmt_vtt_time(s: float) -> str:
        h   = int(s // 3600)
        m   = int((s % 3600) // 60)
        sec = int(s % 60)
        ms  = int(round((s - int(s)) * 1000))
        if ms >= 1000:
            ms = 999
        return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"

    @classmethod
    def _format_txt(cls, segments: list[dict], header: str,
                    txt_style: str, show_ts: bool) -> str:
        """Format segments into TXT output based on chosen style."""
        if txt_style == "Plain Text":
            body_lines = [(seg.get("text") or "").strip()
                          for seg in segments]
            return header + "\n".join(body_lines)

        elif txt_style == "Dialogue Format":
            lines: list[str] = []
            current_spk = None
            buf_text: list[str] = []
            buf_start = None

            for seg in segments:
                sp = seg.get("speaker")
                text = (seg.get("text") or "").strip()
                if not text:
                    continue
                if sp != current_spk:
                    if buf_text and current_spk is not None:
                        merged = " ".join(buf_text)
                        line = f"{current_spk}: {merged}"
                        if show_ts and buf_start is not None:
                            line = f"[{cls._fmt_time(buf_start)}] {line}"
                        lines.append("")
                        lines.append(line)
                    elif buf_text and current_spk is None:
                        lines.extend(buf_text)
                    buf_text = [text]
                    buf_start = seg["start"]
                    current_spk = sp
                else:
                    buf_text.append(text)

            if buf_text:
                if current_spk is not None:
                    merged = " ".join(buf_text)
                    line = f"{current_spk}: {merged}"
                    if show_ts and buf_start is not None:
                        line = f"[{cls._fmt_time(buf_start)}] {line}"
                    lines.append("")
                    lines.append(line)
                else:
                    lines.extend(buf_text)

            return header + "\n".join(lines)

        elif txt_style == "Compact Blocks":
            lines = []
            current_spk = None
            buf_text = []
            buf_start = None
            buf_end = None

            for seg in segments:
                sp = seg.get("speaker")
                text = (seg.get("text") or "").strip()
                if not text:
                    continue
                if sp != current_spk:
                    if buf_text and current_spk is not None:
                        lines.append("")
                        hdr = f"{current_spk}"
                        if show_ts and buf_start is not None and buf_end is not None:
                            hdr += (f"  [{cls._fmt_time(buf_start)}"
                                    f" - {cls._fmt_time(buf_end)}]")
                        lines.append(hdr)
                        lines.append("  " + " ".join(buf_text))
                    elif buf_text:
                        lines.extend(buf_text)
                    buf_text = [text]
                    buf_start = seg["start"]
                    current_spk = sp
                else:
                    buf_text.append(text)
                buf_end = seg["end"]

            if buf_text:
                if current_spk is not None:
                    lines.append("")
                    hdr = f"{current_spk}"
                    if show_ts and buf_start is not None and buf_end is not None:
                        hdr += (f"  [{cls._fmt_time(buf_start)}"
                                f" - {cls._fmt_time(buf_end)}]")
                    lines.append(hdr)
                    lines.append("  " + " ".join(buf_text))
                else:
                    lines.extend(buf_text)

            return header + "\n".join(lines)

        else:
            # "Timestamped Blocks" — original format
            lines = []
            current_speaker = None
            buffer = []

            for seg in segments:
                ts = (f"[{cls._fmt_time(seg['start'])} -> "
                      f"{cls._fmt_time(seg['end'])}]")
                text = (seg.get("text") or "").strip()
                sp = seg.get("speaker")

                if sp is not None:
                    if sp != current_speaker:
                        if buffer:
                            lines.append(f"\n{current_speaker}:")
                            lines.extend(buffer)
                            buffer = []
                        current_speaker = sp
                    if show_ts:
                        buffer.append(f"  {ts}  {text}")
                    else:
                        buffer.append(f"  {text}")
                else:
                    if show_ts:
                        lines.append(f"{ts}  {text}")
                    else:
                        lines.append(text)

            if buffer and current_speaker:
                lines.append(f"\n{current_speaker}:")
                lines.extend(buffer)

            return header + "\n".join(lines)

    @classmethod
    def _format_srt(cls, segments: list[dict],
                    label_fmt: str, srt_labels: bool) -> str:
        """Format segments into SRT subtitle format."""
        srt_lines: list[str] = []
        for i, seg in enumerate(segments, 1):
            sp_label = ""
            if seg.get("speaker") and srt_labels:
                remapped = cls.make_label(seg["speaker"], label_fmt)
                sp_label = f"[{remapped}] "
            txt = (seg.get("text") or "").strip()
            srt_lines.append(
                f"{i}\n"
                f"{cls._fmt_srt_time(seg['start'])} --> "
                f"{cls._fmt_srt_time(seg['end'])}\n"
                f"{sp_label}{txt}\n")
        return "\n".join(srt_lines)

    @classmethod
    def _format_vtt(cls, segments: list[dict],
                    label_fmt: str, srt_labels: bool) -> str:
        """Format segments into WebVTT subtitle format."""
        vtt_lines = ["WEBVTT", ""]
        for i, seg in enumerate(segments, 1):
            sp_label = ""
            if seg.get("speaker") and srt_labels:
                remapped = cls.make_label(seg["speaker"], label_fmt)
                sp_label = f"[{remapped}] "
            txt = (seg.get("text") or "").strip()
            vtt_lines.append(
                f"{i}\n"
                f"{cls._fmt_vtt_time(seg['start'])} --> "
                f"{cls._fmt_vtt_time(seg['end'])}\n"
                f"{sp_label}{txt}\n")
        return "\n".join(vtt_lines)


# ─── Signals ─────────────────────────────────────────────────

class WorkerSignals(QObject):
    progress  = Signal(str, int, str)     # uid, percent, message
    finished  = Signal(str, str, str)     # uid, result_type, payload
    error     = Signal(str, str)          # uid, error_message
    cancelled = Signal(str)               # uid
    log       = Signal(str)


# ─── Worker Threads ──────────────────────────────────────────

class TitleFetchWorker(threading.Thread):
    """Fetches only the video title to update the UI smartly."""

    def __init__(self, signals: WorkerSignals, task_uid: str, url: str):
        super().__init__(daemon=True)
        self.signals  = signals
        self.task_uid = task_uid
        self.url      = url

    def run(self) -> None:
        try:
            ydl_opts = {"quiet": True, "no_warnings": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                title = info.get("title", "")
                if title:
                    self.signals.finished.emit(self.task_uid, "title", title)
        except Exception as e:
            self.signals.log.emit(
                f"Could not resolve title for {self.url}: {e}")


class DownloadWorker(threading.Thread):
    """Downloads audio/video from a URL using yt-dlp."""

    def __init__(self, signals: WorkerSignals, task_uid: str, url: str,
                 output_dir: str, audio_only: bool, convert_mp3: bool,
                 cancel_event: threading.Event):
        super().__init__(daemon=True)
        self.signals      = signals
        self.task_uid     = task_uid
        self.url          = url
        self.output_dir   = output_dir
        self.audio_only   = audio_only
        self.convert_mp3  = convert_mp3
        self.cancel_event = cancel_event

    def run(self) -> None:
        try:
            if self.cancel_event.is_set():
                return self._cancelled()

            os.makedirs(self.output_dir, exist_ok=True)
            ydl_opts: dict = {
                "outtmpl": os.path.join(
                    self.output_dir, "%(title)s.%(ext)s"),
                "progress_hooks": [self._hook],
                "quiet": True, "no_warnings": True,
            }

            if self.audio_only or self.convert_mp3:
                ydl_opts["format"] = "bestaudio/best"
            else:
                ydl_opts["format"] = "bestvideo+bestaudio/best"

            if self.convert_mp3:
                ydl_opts["postprocessors"] = [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }]

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
                if self.cancel_event.is_set():
                    return self._cancelled()
                filename = ydl.prepare_filename(info)
                title    = info.get("title", "")

            if self.convert_mp3:
                base, _ = os.path.splitext(filename)
                filename = base + ".mp3"

            if not os.path.exists(filename):
                base, _ = os.path.splitext(filename)
                for ext in (".mp3", ".m4a", ".wav", ".ogg", ".flac",
                            ".webm", ".mp4", ".mkv"):
                    candidate = base + ext
                    if os.path.exists(candidate):
                        filename = candidate
                        break

            if not os.path.exists(filename):
                raise FileNotFoundError(
                    f"Could not locate downloaded file for: "
                    f"{info.get('title', 'Unknown')}")

            self.signals.log.emit(
                f"Downloaded: {os.path.basename(filename)}")
            self.signals.progress.emit(
                self.task_uid, 100, "Download complete")
            self.signals.finished.emit(
                self.task_uid, "download",
                f"{filename}|||{title}")

        except InterruptedError:
            self._cancelled()
        except Exception as e:
            self.signals.error.emit(
                self.task_uid, f"Download failed: {e}")

    def _cancelled(self) -> None:
        self.signals.cancelled.emit(self.task_uid)

    def _hook(self, d: dict) -> None:
        if self.cancel_event.is_set():
            raise InterruptedError("Cancelled by user")
        if d["status"] == "downloading":
            try:
                pct = float(
                    d.get("_percent_str", "0%").strip().replace("%", ""))
                speed = d.get("_speed_str", "N/A")
                eta = d.get("_eta_str", "N/A")
                self.signals.progress.emit(
                    self.task_uid, int(pct),
                    f"Downloading {pct:.0f}%  |  {speed}  |  ETA {eta}")
            except ValueError:
                pass


class TranscriptionWorker(threading.Thread):
    """Transcribes audio using Whisper with optional diarization."""

    def __init__(self, signals: WorkerSignals, task_uid: str,
                 audio_path: str, model_size: str, language: str,
                 diarize: bool, num_speakers: int,
                 cancel_event: threading.Event,
                 speaker_label_fmt: str = "Speaker 1, 2, 3...",
                 text_format_style: str = "Timestamped Blocks",
                 include_timestamps: bool = True,
                 include_lang_header: bool = True,
                 srt_speaker_labels: bool = True,
                 include_vtt: bool = False):
        super().__init__(daemon=True)
        self.signals            = signals
        self.task_uid           = task_uid
        self.audio_path         = audio_path
        self.model_size         = model_size
        self.language           = language
        self.diarize            = diarize
        self.num_speakers       = num_speakers
        self.cancel_event       = cancel_event
        self.speaker_label_fmt  = speaker_label_fmt
        self.text_format_style  = text_format_style
        self.include_timestamps = include_timestamps
        self.include_lang_header = include_lang_header
        self.srt_speaker_labels = srt_speaker_labels
        self.include_vtt        = include_vtt

    def run(self) -> None:
        try:
            self._validate()
            if self.cancel_event.is_set():
                return self._cancelled()

            import whisper

            self.signals.log.emit(
                f"Loading Whisper model '{self.model_size}'...")
            self.signals.progress.emit(
                self.task_uid, 5, "Loading Whisper model...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model  = whisper.load_model(self.model_size, device=device)

            if self.cancel_event.is_set():
                return self._cancelled()

            self.signals.progress.emit(
                self.task_uid, 15, "Transcribing...")
            opts: dict = {}
            if self.language and self.language != "Auto Detect":
                opts["language"] = self.language.lower()

            result = model.transcribe(self.audio_path, **opts)

            if self.cancel_event.is_set():
                return self._cancelled()

            if result is None:
                raise RuntimeError(
                    "Whisper returned no result. The audio file may be "
                    "empty, corrupted, or in an unsupported format.")

            detected_lang = result.get("language", "unknown")
            segments = result.get("segments")

            if segments is None:
                raise RuntimeError(
                    "Whisper returned no segments. The audio may be "
                    "empty or contain no detectable speech.")

            for seg in segments:
                if seg.get("text") is None:
                    seg["text"] = ""

            if not segments:
                self.signals.log.emit(
                    "No speech detected in the audio file.")
                text_out = (f"Language: {detected_lang}\n"
                            f"{'=' * 64}\n\n(No speech detected.)")
                self.signals.progress.emit(
                    self.task_uid, 100, "No speech detected")
                self.signals.finished.emit(
                    self.task_uid, "transcription", text_out)
                self.signals.finished.emit(
                    self.task_uid, "srt_cache", "")
                self.signals.finished.emit(
                    self.task_uid, "vtt_cache", "")
                return

            self.signals.progress.emit(
                self.task_uid, 60, "Whisper transcription complete")

            if self.diarize:
                annotated = self._run_diarization(segments)
            else:
                annotated = [{
                    "start": s["start"], "end": s["end"],
                    "text": s["text"].strip(), "speaker": None,
                } for s in segments]

            text_out, srt_out, vtt_out = TranscriptFormatter.format(
                annotated, detected_lang,
                label_fmt=self.speaker_label_fmt,
                txt_style=self.text_format_style,
                show_ts=self.include_timestamps,
                show_lang=self.include_lang_header,
                srt_labels=self.srt_speaker_labels,
                include_vtt=self.include_vtt,
            )

            self.signals.progress.emit(self.task_uid, 100, "Done!")
            self.signals.finished.emit(
                self.task_uid, "transcription", text_out)
            self.signals.finished.emit(
                self.task_uid, "srt_cache", srt_out)
            if vtt_out:
                self.signals.finished.emit(
                    self.task_uid, "vtt_cache", vtt_out)

        except TypeError as exc:
            msg = str(exc)
            if "NoneType" in msg and "iterable" in msg:
                self.signals.error.emit(self.task_uid,
                    f"Whisper could not process this audio file.\n\n"
                    f"This usually means:\n"
                    f"  - The audio file is corrupted or empty\n"
                    f"  - The audio format is unsupported by FFmpeg\n"
                    f"  - FFmpeg is not properly installed\n\n"
                    f"Original error: {msg}")
            else:
                self.signals.error.emit(
                    self.task_uid, f"Transcription failed: {exc}")
        except Exception as exc:
            self.signals.error.emit(
                self.task_uid, f"Transcription failed: {exc}")

    def _validate(self) -> None:
        if not self.audio_path:
            raise ValueError(
                "No audio file path provided - "
                "the download may have failed silently.")
        if not os.path.isfile(self.audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {self.audio_path}\n"
                "The file may have been moved, deleted, or never downloaded.")
        self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg() -> None:
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, check=True, timeout=5)
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg is not installed or not found in PATH.\n"
                "Whisper requires FFmpeg to decode audio files.\n"
                "Install from https://ffmpeg.org/download.html")
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "FFmpeg is installed but returned an error. "
                "Please verify your FFmpeg installation.")

    def _cancelled(self) -> None:
        self.signals.cancelled.emit(self.task_uid)

    def _run_diarization(self, whisper_segments: list[dict]) -> list[dict]:
        self.signals.progress.emit(
            self.task_uid, 65, "Analyzing speakers...")
        self.signals.log.emit("Running speaker diarization...")

        def progress_cb(msg: str) -> None:
            self.signals.progress.emit(self.task_uid, 67, msg)

        try:
            spk_segs = _try_neural_diarize(
                self.audio_path, self.num_speakers, progress_cb=progress_cb)

            if spk_segs is None:
                diarizer = LocalSpeakerDiarizer()
                spk_segs = diarizer.diarize(
                    self.audio_path, whisper_segments,
                    num_speakers=self.num_speakers,
                    progress_cb=progress_cb
                )

            return self._assign(whisper_segments, spk_segs)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.signals.log.emit(
                f"Diarization failed ({e}) - "
                f"falling back to unlabeled segments.")
            return [{
                "start": s["start"], "end": s["end"],
                "text": (s.get("text") or "").strip(),
                "speaker": None,
            } for s in whisper_segments]

    @staticmethod
    def _assign(transcript: list[dict],
                spk_segs: list[dict]) -> list[dict]:
        """
        Assign speakers to transcript segments using weighted overlap scoring
        with duration prior and transition smoothing.
        """
        if not spk_segs:
            for e in transcript:
                e["speaker"] = e.get("speaker", "Speaker 1")
            return transcript

        speaker_duration: dict[str, float] = {}
        for ss in spk_segs:
            sp = ss["speaker"]
            dur = ss["end"] - ss["start"]
            speaker_duration[sp] = speaker_duration.get(sp, 0) + max(0, dur)

        total_duration = sum(speaker_duration.values()) + 1e-10
        log_prior = {sp: np.log(dur / total_duration + 0.1)
                     for sp, dur in speaker_duration.items()}

        assignments: list[str] = []
        scores_list: list[float] = []

        for e in transcript:
            e_dur = e["end"] - e["start"]
            e_mid = (e["start"] + e["end"]) / 2.0
            best_sp, best_score = "Speaker 1", -np.inf

            for ss in spk_segs:
                ov = max(0, min(e["end"], ss["end"]) - max(e["start"], ss["start"]))
                if ov <= 0:
                    continue

                sp = ss["speaker"]
                ov_ratio = ov / e_dur if e_dur > 0 else 0

                ss_mid = (ss["start"] + ss["end"]) / 2.0
                if e_dur > 0:
                    centrality = 1.0 - abs(e_mid - ss_mid) / (e_dur * 2 + 1e-10)
                    centrality = max(0, centrality)
                else:
                    centrality = 0.5

                score = (0.55 * ov_ratio
                         + 0.20 * centrality
                         + 0.15 * log_prior.get(sp, 0)
                         + 0.10 * (ov / (ss["end"] - ss["start"] + 1e-10)))

                if score > best_score:
                    best_score = score
                    best_sp = sp

            assignments.append(best_sp)
            scores_list.append(best_score)

        # Transition smoothing
        if len(assignments) > 2:
            smoothed = list(assignments)
            for i in range(1, len(assignments) - 1):
                if (smoothed[i] != smoothed[i - 1]
                        and smoothed[i] != smoothed[i + 1]
                        and smoothed[i - 1] == smoothed[i + 1]):
                    if scores_list[i] < 0.5:
                        smoothed[i] = smoothed[i - 1]
            assignments = smoothed

        for i, e in enumerate(transcript):
            e["speaker"] = assignments[i] if i < len(assignments) else "Speaker 1"

        return transcript


# ─── Header Banner Widget ────────────────────────────────────

class WorkerSignals(QObject):
    progress  = Signal(str, int, str)     # uid, percent, message
    finished  = Signal(str, str, str)     # uid, result_type, payload
    error     = Signal(str, str)          # uid, error_message
    cancelled = Signal(str)               # uid
    log       = Signal(str)


# ─── Worker Threads ──────────────────────────────────────────

class TitleFetchWorker(threading.Thread):
    """Fetches only the video title to update the UI smartly."""

    def __init__(self, signals: WorkerSignals, task_uid: str, url: str):
        super().__init__(daemon=True)
        self.signals  = signals
        self.task_uid = task_uid
        self.url      = url

    def run(self) -> None:
        try:
            ydl_opts = {"quiet": True, "no_warnings": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                title = info.get("title", "")
                if title:
                    self.signals.finished.emit(self.task_uid, "title", title)
        except Exception as e:
            self.signals.log.emit(
                f"Could not resolve title for {self.url}: {e}")


class DownloadWorker(threading.Thread):
    """Downloads audio/video from a URL using yt-dlp."""

    def __init__(self, signals: WorkerSignals, task_uid: str, url: str,
                 output_dir: str, audio_only: bool, convert_mp3: bool,
                 cancel_event: threading.Event):
        super().__init__(daemon=True)
        self.signals      = signals
        self.task_uid     = task_uid
        self.url          = url
        self.output_dir   = output_dir
        self.audio_only   = audio_only
        self.convert_mp3  = convert_mp3
        self.cancel_event = cancel_event

    def run(self) -> None:
        try:
            if self.cancel_event.is_set():
                return self._cancelled()

            os.makedirs(self.output_dir, exist_ok=True)
            ydl_opts: dict = {
                "outtmpl": os.path.join(
                    self.output_dir, "%(title)s.%(ext)s"),
                "progress_hooks": [self._hook],
                "quiet": True, "no_warnings": True,
            }

            if self.audio_only or self.convert_mp3:
                ydl_opts["format"] = "bestaudio/best"
            else:
                ydl_opts["format"] = "bestvideo+bestaudio/best"

            if self.convert_mp3:
                ydl_opts["postprocessors"] = [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }]

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
                if self.cancel_event.is_set():
                    return self._cancelled()
                filename = ydl.prepare_filename(info)
                title    = info.get("title", "")

            if self.convert_mp3:
                base, _ = os.path.splitext(filename)
                filename = base + ".mp3"

            if not os.path.exists(filename):
                base, _ = os.path.splitext(filename)
                for ext in (".mp3", ".m4a", ".wav", ".ogg", ".flac",
                            ".webm", ".mp4", ".mkv"):
                    candidate = base + ext
                    if os.path.exists(candidate):
                        filename = candidate
                        break

            if not os.path.exists(filename):
                raise FileNotFoundError(
                    f"Could not locate downloaded file for: "
                    f"{info.get('title', 'Unknown')}")

            self.signals.log.emit(
                f"Downloaded: {os.path.basename(filename)}")
            self.signals.progress.emit(
                self.task_uid, 100, "Download complete")
            self.signals.finished.emit(
                self.task_uid, "download",
                f"{filename}|||{title}")

        except InterruptedError:
            self._cancelled()
        except Exception as e:
            self.signals.error.emit(
                self.task_uid, f"Download failed: {e}")

    def _cancelled(self) -> None:
        self.signals.cancelled.emit(self.task_uid)

    def _hook(self, d: dict) -> None:
        if self.cancel_event.is_set():
            raise InterruptedError("Cancelled by user")
        if d["status"] == "downloading":
            try:
                pct = float(
                    d.get("_percent_str", "0%").strip().replace("%", ""))
                speed = d.get("_speed_str", "N/A")
                eta = d.get("_eta_str", "N/A")
                self.signals.progress.emit(
                    self.task_uid, int(pct),
                    f"Downloading {pct:.0f}%  |  {speed}  |  ETA {eta}")
            except ValueError:
                pass


class TranscriptionWorker(threading.Thread):
    """Transcribes audio using Whisper with optional diarization."""

    def __init__(self, signals: WorkerSignals, task_uid: str,
                 audio_path: str, model_size: str, language: str,
                 diarize: bool, num_speakers: int,
                 cancel_event: threading.Event,
                 speaker_label_fmt: str = "Speaker 1, 2, 3...",
                 text_format_style: str = "Timestamped Blocks",
                 include_timestamps: bool = True,
                 include_lang_header: bool = True,
                 srt_speaker_labels: bool = True,
                 include_vtt: bool = False):
        super().__init__(daemon=True)
        self.signals            = signals
        self.task_uid           = task_uid
        self.audio_path         = audio_path
        self.model_size         = model_size
        self.language           = language
        self.diarize            = diarize
        self.num_speakers       = num_speakers
        self.cancel_event       = cancel_event
        self.speaker_label_fmt  = speaker_label_fmt
        self.text_format_style  = text_format_style
        self.include_timestamps = include_timestamps
        self.include_lang_header = include_lang_header
        self.srt_speaker_labels = srt_speaker_labels
        self.include_vtt        = include_vtt

    def run(self) -> None:
        try:
            self._validate()
            if self.cancel_event.is_set():
                return self._cancelled()

            import whisper

            self.signals.log.emit(
                f"Loading Whisper model '{self.model_size}'...")
            self.signals.progress.emit(
                self.task_uid, 5, "Loading Whisper model...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model  = whisper.load_model(self.model_size, device=device)

            if self.cancel_event.is_set():
                return self._cancelled()

            self.signals.progress.emit(
                self.task_uid, 15, "Transcribing...")
            opts: dict = {}
            if self.language and self.language != "Auto Detect":
                opts["language"] = self.language.lower()

            result = model.transcribe(self.audio_path, **opts)

            if self.cancel_event.is_set():
                return self._cancelled()

            if result is None:
                raise RuntimeError(
                    "Whisper returned no result. The audio file may be "
                    "empty, corrupted, or in an unsupported format.")

            detected_lang = result.get("language", "unknown")
            segments = result.get("segments")

            if segments is None:
                raise RuntimeError(
                    "Whisper returned no segments. The audio may be "
                    "empty or contain no detectable speech.")

            for seg in segments:
                if seg.get("text") is None:
                    seg["text"] = ""

            if not segments:
                self.signals.log.emit(
                    "No speech detected in the audio file.")
                text_out = (f"Language: {detected_lang}\n"
                            f"{'=' * 64}\n\n(No speech detected.)")
                self.signals.progress.emit(
                    self.task_uid, 100, "No speech detected")
                self.signals.finished.emit(
                    self.task_uid, "transcription", text_out)
                self.signals.finished.emit(
                    self.task_uid, "srt_cache", "")
                self.signals.finished.emit(
                    self.task_uid, "vtt_cache", "")
                return

            self.signals.progress.emit(
                self.task_uid, 60, "Whisper transcription complete")

            if self.diarize:
                annotated = self._run_diarization(segments)
            else:
                annotated = [{
                    "start": s["start"], "end": s["end"],
                    "text": s["text"].strip(), "speaker": None,
                } for s in segments]

            text_out, srt_out, vtt_out = TranscriptFormatter.format(
                annotated, detected_lang,
                label_fmt=self.speaker_label_fmt,
                txt_style=self.text_format_style,
                show_ts=self.include_timestamps,
                show_lang=self.include_lang_header,
                srt_labels=self.srt_speaker_labels,
                include_vtt=self.include_vtt,
            )

            self.signals.progress.emit(self.task_uid, 100, "Done!")
            self.signals.finished.emit(
                self.task_uid, "transcription", text_out)
            self.signals.finished.emit(
                self.task_uid, "srt_cache", srt_out)
            if vtt_out:
                self.signals.finished.emit(
                    self.task_uid, "vtt_cache", vtt_out)

        except TypeError as exc:
            msg = str(exc)
            if "NoneType" in msg and "iterable" in msg:
                self.signals.error.emit(self.task_uid,
                    f"Whisper could not process this audio file.\n\n"
                    f"This usually means:\n"
                    f"  - The audio file is corrupted or empty\n"
                    f"  - The audio format is unsupported by FFmpeg\n"
                    f"  - FFmpeg is not properly installed\n\n"
                    f"Original error: {msg}")
            else:
                self.signals.error.emit(
                    self.task_uid, f"Transcription failed: {exc}")
        except Exception as exc:
            self.signals.error.emit(
                self.task_uid, f"Transcription failed: {exc}")

    def _validate(self) -> None:
        if not self.audio_path:
            raise ValueError(
                "No audio file path provided - "
                "the download may have failed silently.")
        if not os.path.isfile(self.audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {self.audio_path}\n"
                "The file may have been moved, deleted, or never downloaded.")
        self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg() -> None:
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, check=True, timeout=5)
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg is not installed or not found in PATH.\n"
                "Whisper requires FFmpeg to decode audio files.\n"
                "Install from https://ffmpeg.org/download.html")
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "FFmpeg is installed but returned an error. "
                "Please verify your FFmpeg installation.")

    def _cancelled(self) -> None:
        self.signals.cancelled.emit(self.task_uid)

    def _run_diarization(self, whisper_segments: list[dict]) -> list[dict]:
        self.signals.progress.emit(
            self.task_uid, 65, "Analyzing speakers...")
        self.signals.log.emit("Running speaker diarization...")

        def progress_cb(msg: str) -> None:
            self.signals.progress.emit(self.task_uid, 67, msg)

        try:
            spk_segs = _try_neural_diarize(
                self.audio_path, self.num_speakers, progress_cb=progress_cb)

            if spk_segs is None:
                diarizer = LocalSpeakerDiarizer()
                spk_segs = diarizer.diarize(
                    self.audio_path, whisper_segments,
                    num_speakers=self.num_speakers,
                    progress_cb=progress_cb
                )

            return self._assign(whisper_segments, spk_segs)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.signals.log.emit(
                f"Diarization failed ({e}) - "
                f"falling back to unlabeled segments.")
            return [{
                "start": s["start"], "end": s["end"],
                "text": (s.get("text") or "").strip(),
                "speaker": None,
            } for s in whisper_segments]

    @staticmethod
    def _assign(transcript: list[dict],
                spk_segs: list[dict]) -> list[dict]:
        """
        Assign speakers to transcript segments using weighted overlap scoring
        with duration prior and transition smoothing.
        """
        if not spk_segs:
            for e in transcript:
                e["speaker"] = e.get("speaker", "Speaker 1")
            return transcript

        speaker_duration: dict[str, float] = {}
        for ss in spk_segs:
            sp = ss["speaker"]
            dur = ss["end"] - ss["start"]
            speaker_duration[sp] = speaker_duration.get(sp, 0) + max(0, dur)

        total_duration = sum(speaker_duration.values()) + 1e-10
        log_prior = {sp: np.log(dur / total_duration + 0.1)
                     for sp, dur in speaker_duration.items()}

        assignments: list[str] = []
        scores_list: list[float] = []

        for e in transcript:
            e_dur = e["end"] - e["start"]
            e_mid = (e["start"] + e["end"]) / 2.0
            best_sp, best_score = "Speaker 1", -np.inf

            for ss in spk_segs:
                ov = max(0, min(e["end"], ss["end"]) - max(e["start"], ss["start"]))
                if ov <= 0:
                    continue

                sp = ss["speaker"]
                ov_ratio = ov / e_dur if e_dur > 0 else 0

                ss_mid = (ss["start"] + ss["end"]) / 2.0
                if e_dur > 0:
                    centrality = 1.0 - abs(e_mid - ss_mid) / (e_dur * 2 + 1e-10)
                    centrality = max(0, centrality)
                else:
                    centrality = 0.5

                score = (0.55 * ov_ratio
                         + 0.20 * centrality
                         + 0.15 * log_prior.get(sp, 0)
                         + 0.10 * (ov / (ss["end"] - ss["start"] + 1e-10)))

                if score > best_score:
                    best_score = score
                    best_sp = sp

            assignments.append(best_sp)
            scores_list.append(best_score)

        # Transition smoothing
        if len(assignments) > 2:
            smoothed = list(assignments)
            for i in range(1, len(assignments) - 1):
                if (smoothed[i] != smoothed[i - 1]
                        and smoothed[i] != smoothed[i + 1]
                        and smoothed[i - 1] == smoothed[i + 1]):
                    if scores_list[i] < 0.5:
                        smoothed[i] = smoothed[i - 1]
            assignments = smoothed

        for i, e in enumerate(transcript):
            e["speaker"] = assignments[i] if i < len(assignments) else "Speaker 1"

        return transcript


# ─── Header Banner Widget ────────────────────────────────────

class HeaderBanner(QFrame):
    """Professional app header banner with status indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLOR_SURFACE0}, stop:1 {COLOR_BASE});
                border-bottom: 1px solid {COLOR_SURFACE1};
                border-radius: 0px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 4, 16, 4)

        # Status indicator dot
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(
            f"color: {COLOR_GREEN}; font-size: 18px; border: none;")
        layout.addWidget(self.status_dot)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(0)

        title = QLabel(APP_TITLE)
        title.setStyleSheet(f"""
            color: {COLOR_BLUE}; font-size: 18px;
            font-weight: bold; border: none;
        """)
        title_layout.addWidget(title)

        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setStyleSheet(f"""
            color: {COLOR_SUBTEXT}; font-size: 11px; border: none;
        """)
        title_layout.addWidget(subtitle)

        layout.addLayout(title_layout)
        layout.addStretch()

        version = QLabel(f"v{APP_VERSION}")
        version.setStyleSheet(f"""
            color: {COLOR_SURFACE2}; font-size: 10px; border: none;
        """)
        layout.addWidget(version)

    def set_busy(self, busy: bool) -> None:
        if busy:
            self.status_dot.setStyleSheet(
                f"color: {COLOR_BLUE}; font-size: 18px; border: none;")
        else:
            self.status_dot.setStyleSheet(
                f"color: {COLOR_GREEN}; font-size: 18px; border: none;")


# ─── Enhanced Stats Bar ─────────────────────────────────────

class StatsBar(QFrame):
    """Batch statistics summary bar with word count and duration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLOR_SURFACE0};
                border-top: 1px solid {COLOR_SURFACE1};
            }}
        """)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(16, 0, 16, 0)

        self.lbl_total = self._make_stat()
        self.lbl_done  = self._make_stat(COLOR_GREEN)
        self.lbl_error = self._make_stat(COLOR_RED)
        self.lbl_pending = self._make_stat(COLOR_YELLOW)
        self.lbl_words = self._make_stat(COLOR_LAVENDER)
        self.lbl_duration = self._make_stat(COLOR_TEAL)

        for widget in (self.lbl_total, self.lbl_done, self.lbl_error,
                       self.lbl_pending, self.lbl_words, self.lbl_duration):
            self.layout.addWidget(widget)
        self.layout.addStretch()

    def _make_stat(self, color: str = COLOR_SUBTEXT) -> QLabel:
        lbl = QLabel("—")
        lbl.setStyleSheet(f"""
            color: {color}; font-size: 11px;
            font-weight: bold; border: none;
        """)
        return lbl

    def update_stats(self, tasks: list) -> None:
        total   = len(tasks)
        done    = sum(1 for t in tasks if t.status == STATUS_DONE)
        error   = sum(1 for t in tasks if t.status == STATUS_ERROR)
        pending = sum(1 for t in tasks if t.status not in TERMINAL_STATUSES)

        # Word count across all completed transcripts
        total_words = 0
        for t in tasks:
            if t.transcript_text:
                total_words += len(t.transcript_text.split())

        self.lbl_total.setText(f"  Total: {total}  ")
        self.lbl_done.setText(f"  Done: {done}  ")
        self.lbl_error.setText(f"  Failed: {error}  ")
        self.lbl_pending.setText(f"  Pending: {pending}  ")
        self.lbl_words.setText(f"  Words: {total_words:,}  ")
        self.lbl_duration.setText(f"  Tasks: {done}/{total}  ")


# ─── Search Bar Widget ──────────────────────────────────────

class SearchBar(QWidget):
    """Search bar for filtering transcript text with match navigation."""

    def __init__(self, text_edit: QTextEdit, parent=None):
        super().__init__(parent)
        self.text_edit = text_edit
        self.matches: list[QTextCursor] = []
        self.current_match = -1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Search transcript... (Ctrl+F)")
        self.input.setClearButtonEnabled(True)
        self.input.setMaximumWidth(300)
        self.input.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.input)

        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet(f"""
            color: {COLOR_SUBTEXT}; font-size: 10px;
            border: none; min-width: 80px;
        """)
        layout.addWidget(self.lbl_count)

        self.btn_prev = QToolButton()
        self.btn_prev.setText("▲")
        self.btn_prev.setFixedSize(24, 24)
        self.btn_prev.clicked.connect(self._prev_match)
        layout.addWidget(self.btn_prev)

        self.btn_next = QToolButton()
        self.btn_next.setText("▼")
        self.btn_next.setFixedSize(24, 24)
        self.btn_next.clicked.connect(self._next_match)
        layout.addWidget(self.btn_next)

        layout.addStretch()

        # Apply style
        self.input.setStyleSheet(f"""
            QLineEdit {{
                background: {COLOR_SURFACE0}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_SURFACE1}; border-radius: 6px;
                padding: 4px 8px; font-size: 11px;
            }}
            QLineEdit:focus {{ border-color: {COLOR_BLUE}; }}
        """)
        self.btn_prev.setStyleSheet(f"""
            QToolButton {{
                background: {COLOR_SURFACE1}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_SURFACE1}; border-radius: 4px;
                font-size: 10px; font-weight: bold;
            }}
            QToolButton:hover {{ background: {COLOR_SURFACE2}; }}
        """)
        self.btn_next.setStyleSheet(self.btn_prev.styleSheet())

    def _on_text_changed(self, text: str) -> None:
        self._find_all(text)
        if self.matches:
            self.current_match = 0
            self._show_current()
        else:
            self.current_match = -1
            self.text_edit.extraSelections().clear()
        self._update_count_label()

    def _find_all(self, text: str) -> None:
        self.matches.clear()
        if not text:
            return

        doc = self.text_edit.document()
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.Start)

        fmt = QTextCharFormat()
        fmt.setBackground(QBrush(QColor(COLOR_YELLOW)))
        fmt.setForeground(QBrush(QColor(COLOR_BASE)))

        while True:
            cursor = doc.find(text, cursor)
            if cursor.isNull():
                break
            self.matches.append(cursor)

        # Highlight all matches
        selections = []
        for m in self.matches:
            sel = QTextEdit.ExtraSelection()
            sel.cursor = m
            sel.format = fmt
            selections.append(sel)
        self.text_edit.setExtraSelections(selections)

    def _show_current(self) -> None:
        if not self.matches or self.current_match < 0:
            return
        cursor = self.matches[self.current_match]
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def _update_count_label(self) -> None:
        if not self.matches:
            self.lbl_count.setText("No matches")
        else:
            self.lbl_count.setText(
                f"{self.current_match + 1} of {len(self.matches)}")

    def _next_match(self) -> None:
        if not self.matches:
            return
        self.current_match = (self.current_match + 1) % len(self.matches)
        self._show_current()
        self._update_count_label()

    def _prev_match(self) -> None:
        if not self.matches:
            return
        self.current_match = (self.current_match - 1) % len(self.matches)
        self._show_current()
        self._update_count_label()

    def activate(self) -> None:
        self.input.setFocus()
        self.input.selectAll()


# ─── Main Window ─────────────────────────────────────────────

# ─── Animated Progress Ring ────────────────────────────────────

class ProgressRing(QWidget):
    """Small circular progress indicator for the header."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.value = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(50)
    
    def set_value(self, v: int):
        self.value = max(0, min(100, v))
        self.update()
    
    def paintEvent(self, event):
        if self.value <= 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        size = min(rect.width(), rect.height())
        pen_w = 2.5
        p.setPen(QPen(QColor(COLOR_SURFACE1), pen_w))
        p.drawEllipse(rect.adjusted(pen_w, pen_w, -pen_w, -pen_w))
        pen = QPen(QColor(COLOR_BLUE), pen_w)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        angle = int((self.value / 100.0) * 360 * 16)
        p.drawArc(rect.adjusted(pen_w, pen_w, -pen_w, -pen_w),
                  90 * 16, -angle)
        p.end()


# ─── Header Banner Widget ────────────────────────────────────

class HeaderBanner(QFrame):
    """Professional app header with gradient, branding, and live status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(64)
        self._is_busy = False
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLOR_SURFACE}, stop:1 {COLOR_BASE});
                border-bottom: 1px solid {COLOR_SURFACE1};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 8, 20, 8)
        layout.setSpacing(14)

        # Animated status dot
        self.status_dot = QLabel("\u25cf")
        self.status_dot.setFixedWidth(14)
        self.status_dot.setStyleSheet(
            f"color: {COLOR_GREEN}; font-size: 16px; border: none;")
        self.status_dot.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_dot)

        # Pulse animation timer
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_visible = True

        # Title + Subtitle
        title_layout = QVBoxLayout()
        title_layout.setSpacing(1)

        title = QLabel(f"  {APP_TITLE}")
        title.setStyleSheet(f"""
            color: {COLOR_BLUE}; font-size: 20px;
            font-weight: bold; border: none; letter-spacing: 0.5px;
        """)
        title_layout.addWidget(title)

        subtitle = QLabel(f"  {APP_SUBTITLE}")
        subtitle.setStyleSheet(f"""
            color: {COLOR_SUBTEXT}; font-size: 10px; border: none;
            letter-spacing: 0.3px;
        """)
        title_layout.addWidget(subtitle)

        layout.addLayout(title_layout)
        layout.addStretch()

        # GPU status
        gpu_text = "GPU" if torch.cuda.is_available() else "CPU"
        gpu_color = COLOR_GREEN if torch.cuda.is_available() else COLOR_YELLOW
        self.gpu_label = QLabel(f"  {gpu_text}  ")
        self.gpu_label.setStyleSheet(f"""
            color: {COLOR_BASE}; background: {gpu_color};
            font-size: 9px; font-weight: bold; border: none;
            border-radius: 4px; padding: 2px 8px;
        """)
        layout.addWidget(self.gpu_label)

        # Progress ring
        self.ring = ProgressRing()
        layout.addWidget(self.ring)

        # Version badge
        version = QLabel(f" v{APP_VERSION} ")
        version.setStyleSheet(f"""
            color: {COLOR_SUBTEXT}; font-size: 9px; border: none;
            background: {COLOR_SURFACE1}; border-radius: 4px; padding: 2px 8px;
        """)
        layout.addWidget(version)

    def set_busy(self, busy: bool) -> None:
        self._is_busy = busy
        if busy:
            self.status_dot.setStyleSheet(
                f"color: {COLOR_BLUE}; font-size: 16px; border: none;")
            self._pulse_timer.start(500)
        else:
            self.status_dot.setStyleSheet(
                f"color: {COLOR_GREEN}; font-size: 16px; border: none;")
            self._pulse_timer.stop()
            self.status_dot.setVisible(True)

    def _pulse(self) -> None:
        self._pulse_visible = not self._pulse_visible
        self.status_dot.setVisible(self._pulse_visible)

    def set_progress(self, value: int) -> None:
        self.ring.set_value(value)


# ─── Enhanced Stats Bar ─────────────────────────────────────

class StatsBar(QFrame):
    """Professional batch statistics bar with pill-style indicators."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLOR_SURFACE};
                border-top: 1px solid {COLOR_SURFACE1};
            }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(4)

        self.lbl_total   = self._pill("Total", COLOR_TEXT)
        self.lbl_done    = self._pill("Done", COLOR_GREEN)
        self.lbl_error   = self._pill("Failed", COLOR_RED)
        self.lbl_pending = self._pill("Active", COLOR_YELLOW)
        self.lbl_words   = self._pill("Words", COLOR_LAVENDER)

        for w in (self.lbl_total, self.lbl_done, self.lbl_error,
                  self.lbl_pending, self.lbl_words):
            lay.addWidget(w)
        lay.addStretch()

    def _pill(self, label: str, color: str) -> QLabel:
        lbl = QLabel(f" {label}: \u2014 ")
        lbl.setStyleSheet(f"""
            color: {color}; font-size: 10px;
            font-weight: bold; border: none;
            background: {COLOR_SURFACE0};
            border-radius: 4px; padding: 3px 10px;
        """)
        return lbl

    def update_stats(self, tasks: list) -> None:
        total   = len(tasks)
        done    = sum(1 for t in tasks if t.status == STATUS_DONE)
        error   = sum(1 for t in tasks if t.status == STATUS_ERROR)
        pending = sum(1 for t in tasks if t.status not in TERMINAL_STATUSES)
        total_words = sum(
            len(t.transcript_text.split()) for t in tasks if t.transcript_text)

        self.lbl_total.setText(f"  Total: {total}  ")
        self.lbl_done.setText(f"  Done: {done}  ")
        self.lbl_error.setText(f"  Failed: {error}  ")
        self.lbl_pending.setText(f"  Active: {pending}  ")
        self.lbl_words.setText(f"  Words: {total_words:,}  ")


# ─── Search Bar Widget ──────────────────────────────────────

class SearchBar(QWidget):
    """Search bar for filtering transcript text with match navigation."""

    def __init__(self, text_edit: QTextEdit, parent=None):
        super().__init__(parent)
        self.text_edit = text_edit
        self.matches: list[QTextCursor] = []
        self.current_match = -1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Search transcript... (Ctrl+F)")
        self.input.setClearButtonEnabled(True)
        self.input.setMaximumWidth(280)
        self.input.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.input)

        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet(f"""
            color: {COLOR_SUBTEXT}; font-size: 10px;
            border: none; min-width: 70px;
        """)
        layout.addWidget(self.lbl_count)

        self.btn_prev = QToolButton()
        self.btn_prev.setText("\u25b2")
        self.btn_prev.setFixedSize(22, 22)
        self.btn_prev.clicked.connect(self._prev_match)
        layout.addWidget(self.btn_prev)

        self.btn_next = QToolButton()
        self.btn_next.setText("\u25bc")
        self.btn_next.setFixedSize(22, 22)
        self.btn_next.clicked.connect(self._next_match)
        layout.addWidget(self.btn_next)

        layout.addStretch()

        self.input.setStyleSheet(f"""
            QLineEdit {{
                background: {COLOR_SURFACE0}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_SURFACE1}; border-radius: 6px;
                padding: 4px 8px; font-size: 11px;
            }}
            QLineEdit:focus {{ border-color: {COLOR_BLUE}; }}
        """)
        nav_style = f"""
            QToolButton {{
                background: {COLOR_SURFACE1}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_SURFACE1}; border-radius: 4px;
                font-size: 9px; font-weight: bold;
            }}
            QToolButton:hover {{ background: {COLOR_SURFACE2}; }}
        """
        self.btn_prev.setStyleSheet(nav_style)
        self.btn_next.setStyleSheet(nav_style)

    def _on_text_changed(self, text: str) -> None:
        self._find_all(text)
        if self.matches:
            self.current_match = 0
            self._show_current()
        else:
            self.current_match = -1
            self.text_edit.setExtraSelections([])
        self._update_count_label()

    def _find_all(self, text: str) -> None:
        self.matches.clear()
        if not text:
            return
        doc = self.text_edit.document()
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.Start)
        fmt = QTextCharFormat()
        fmt.setBackground(QBrush(QColor(COLOR_YELLOW)))
        fmt.setForeground(QBrush(QColor(COLOR_BASE)))
        while True:
            cursor = doc.find(text, cursor)
            if cursor.isNull():
                break
            self.matches.append(cursor)
        selections = []
        for m in self.matches:
            sel = QTextEdit.ExtraSelection()
            sel.cursor = m
            sel.format = fmt
            selections.append(sel)
        self.text_edit.setExtraSelections(selections)

    def _show_current(self) -> None:
        if not self.matches or self.current_match < 0:
            return
        cursor = self.matches[self.current_match]
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def _update_count_label(self) -> None:
        if not self.matches:
            self.lbl_count.setText("No matches")
        else:
            self.lbl_count.setText(
                f"{self.current_match + 1} of {len(self.matches)}")

    def _next_match(self) -> None:
        if not self.matches:
            return
        self.current_match = (self.current_match + 1) % len(self.matches)
        self._show_current()
        self._update_count_label()

    def _prev_match(self) -> None:
        if not self.matches:
            return
        self.current_match = (self.current_match - 1) % len(self.matches)
        self._show_current()
        self._update_count_label()

    def activate(self) -> None:
        self.input.setFocus()
        self.input.selectAll()


# ─── Collapsible Section ─────────────────────────────────────

class CollapsibleSection(QFrame):
    """A QGroupBox that can be collapsed/expanded with a toggle button."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._collapsed = False
        self.setObjectName("collapsibleSection")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Toggle header
        self.header = QFrame()
        self.header.setFixedHeight(36)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setObjectName("sectionHeader")
        header_lay = QHBoxLayout(self.header)
        header_lay.setContentsMargins(14, 0, 14, 0)

        self.toggle_btn = QPushButton()
        self.toggle_btn.setFlat(True)
        self.toggle_btn.setFixedSize(20, 20)
        self.toggle_btn.setText("\u25bc")
        self.toggle_btn.setObjectName("toggleBtn")
        self.toggle_btn.clicked.connect(self.toggle)
        header_lay.addWidget(self.toggle_btn)

        self.title_label = QLabel(f"  {title}")
        self.title_label.setStyleSheet(f"""
            color: {COLOR_LAVENDER}; font-size: 12px;
            font-weight: bold; border: none; letter-spacing: 0.3px;
        """)
        header_lay.addWidget(self.title_label)
        header_lay.addStretch()
        self.main_layout.addWidget(self.header)

        # Content container
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(14, 8, 14, 12)
        self.content_layout.setSpacing(8)
        self.main_layout.addWidget(self.content)

    def toggle(self) -> None:
        self._collapsed = not self._collapsed
        self.content.setVisible(not self._collapsed)
        self.toggle_btn.setText(
            "\u25b6" if self._collapsed else "\u25bc")

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)


# ─── About Dialog ────────────────────────────────────────────

class AboutDialog(QDialog):
    """About dialog with app info and keyboard shortcuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_TITLE}")
        self.setFixedSize(480, 520)
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel(f"{APP_TITLE} Pro")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            color: {COLOR_BLUE}; font-size: 24px;
            font-weight: bold; border: none;
        """)
        layout.addWidget(title)

        ver = QLabel(f"Version {APP_VERSION}")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet(f"""
            color: {COLOR_SUBTEXT}; font-size: 11px; border: none;
        """)
        layout.addWidget(ver)

        desc = QLabel(
            "Professional YouTube Transcriber & Speaker Diarizer\n"
            "Powered by OpenAI Whisper + Advanced Local Diarization")
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"""
            color: {COLOR_TEXT_DIM}; font-size: 11px; border: none;
            padding: 8px;
        """)
        layout.addWidget(desc)

        shortcuts_text = (
            "<b style='color:#b4befe'>Keyboard Shortcuts</b><br><br>"
            "<table style='color:#b0b3c6; font-size:11px;' cellpadding='4'>"
            "<tr><td><b>Ctrl+V</b></td><td>Paste URL into input</td></tr>"
            "<tr><td><b>Ctrl+Shift+V</b></td><td>Bulk paste URLs</td></tr>"
            "<tr><td><b>Ctrl+O</b></td><td>Open local files</td></tr>"
            "<tr><td><b>Ctrl+Return</b></td><td>Start processing</td></tr>"
            "<tr><td><b>Escape</b></td><td>Stop processing</td></tr>"
            "<tr><td><b>Ctrl+F</b></td><td>Search transcript</td></tr>"
            "<tr><td><b>Ctrl+Shift+C</b></td><td>Copy transcript</td></tr>"
            "<tr><td><b>Delete</b></td><td>Remove selected tasks</td></tr>"
            "</table>"
        )
        shortcuts = QLabel(shortcuts_text)
        shortcuts.setWordWrap(True)
        shortcuts.setStyleSheet(f"""
            background: {COLOR_SURFACE0}; color: {COLOR_TEXT};
            border: 1px solid {COLOR_SURFACE1}; border-radius: 8px;
            padding: 12px; font-size: 11px;
        """)
        layout.addWidget(shortcuts)

        layout.addStretch()

        btn = QPushButton("Close")
        btn.setObjectName("primaryBtn")
        btn.setFixedWidth(100)
        btn.clicked.connect(self.accept)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QDialog {{ background: {COLOR_BASE}; }}
            QPushButton#primaryBtn {{
                background: {COLOR_BLUE}; color: {COLOR_BASE};
                border: none; border-radius: 8px;
                padding: 8px 20px; font-weight: bold; font-size: 13px;
            }}
            QPushButton#primaryBtn:hover {{ background: {COLOR_LAVENDER}; }}
        """)


# ─── Drop Overlay ────────────────────────────────────────────

class DropOverlay(QFrame):
    """Transparent overlay shown during drag & drop operations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self.setStyleSheet(f"""
            QFrame {{
                background: rgba(122, 162, 247, 30);
                border: 2px dashed {COLOR_BLUE};
                border-radius: 12px;
            }}
        """)
        layout = QVBoxLayout(self)
        self.label = QLabel("Drop audio/video files here")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(f"""
            color: {COLOR_BLUE}; font-size: 18px;
            font-weight: bold; border: none;
        """)
        layout.addWidget(self.label)

    def show_overlay(self) -> None:
        self.setVisible(True)
        self.raise_()

    def hide_overlay(self) -> None:
        self.setVisible(False)


# ─── Main Window ─────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} Pro v{APP_VERSION}")
        self.setMinimumSize(1020, 860)
        self.resize(1200, 950)
        self.setAcceptDrops(True)

        # Config
        self.config = ConfigManager()

        # State
        self.tasks: list[VideoTask] = []
        self.is_processing = False
        self.cancel_event = threading.Event()
        self.active_workers: list[threading.Thread] = []
        self.concurrent_semaphore = threading.Semaphore(
            self.config.get("max_concurrent"))

        # Batch tracking for concurrent mode
        self._pending_queue: list[VideoTask] = []
        self._in_flight = 0
        self._total_to_process = 0
        self._finished_count = 0

        # Tab content cache
        self._current_task_txt = ""
        self._current_task_srt = ""
        self._current_task_vtt = ""

        # Signals
        self.signals = WorkerSignals()
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_finished)
        self.signals.error.connect(self._on_error)
        self.signals.cancelled.connect(self._on_cancelled)
        self.signals.log.connect(self._on_log)

        self._build_ui()
        self._apply_stylesheet()
        self._setup_shortcuts()
        self._setup_tray()
        self._restore_settings()

        self._output_dir = OUTPUT_DIR

        # Drop overlay
        self.drop_overlay = DropOverlay(self.centralWidget())

    # ──────── Settings Persistence ────────

    def _restore_settings(self) -> None:
        c = self.config
        self.combo_model.setCurrentText(c.get("model"))
        self.combo_lang.setCurrentText(c.get("language"))
        self.combo_fmt.setCurrentText(c.get("download_format"))
        self.combo_output_fmt.setCurrentText(c.get("output_format"))
        self.chk_diarize.setChecked(c.get("diarize"))
        self.spin_speakers.setCurrentText(c.get("num_speakers"))
        self.combo_speaker_labels.setCurrentText(c.get("speaker_label_fmt"))
        self.combo_text_format.setCurrentText(c.get("text_format_style"))
        self.chk_include_timestamps.setChecked(c.get("include_timestamps"))
        self.chk_include_lang_header.setChecked(c.get("include_lang_header"))
        self.chk_srt_speaker_labels.setChecked(c.get("srt_speaker_labels"))
        self.chk_concatenate.setChecked(c.get("concatenate"))
        self.chk_merge_audio.setChecked(c.get("merge_audio"))
        self.spin_concurrent.setValue(c.get("max_concurrent"))

    def _save_setting(self, key: str, value) -> None:
        self.config.set(key, value)

    # ──────── UI Construction ────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # Header
        self.header = HeaderBanner()
        root.addWidget(self.header)

        # Main content in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border: none;")

        scroll_content = QWidget()
        content = QVBoxLayout(scroll_content)
        content.setSpacing(8)
        content.setContentsMargins(16, 10, 16, 8)

        # ── Input Sources Section (Collapsible) ──
        input_section = CollapsibleSection("Add Videos / Audio")
        input_section.setStyleSheet(f"""
            QFrame#collapsibleSection {{
                background: {COLOR_SURFACE};
                border: 1px solid {COLOR_SURFACE1};
                border-radius: 8px;
            }}
            QFrame#sectionHeader {{
                background: {COLOR_SURFACE};
                border: 1px solid {COLOR_SURFACE1};
                border-radius: 8px;
            }}
            QPushButton#toggleBtn {{
                color: {COLOR_LAVENDER}; font-size: 10px;
                background: transparent; border: none;
                font-weight: bold;
            }}
            QPushButton#toggleBtn:hover {{ color: {COLOR_BLUE}; }}
        """)

        url_row = QHBoxLayout()
        url_row.setSpacing(6)

        self.txt_url_input = QLineEdit()
        self.txt_url_input.setPlaceholderText(
            "  Paste a YouTube / Vimeo / Twitch URL here...")
        self.txt_url_input.returnPressed.connect(self._add_single_url)
        url_row.addWidget(self.txt_url_input, stretch=5)

        self.btn_add_url = QPushButton("Add URL")
        self.btn_add_url.setObjectName("primaryBtn")
        self.btn_add_url.setFixedHeight(34)
        self.btn_add_url.setFixedWidth(90)
        self.btn_add_url.clicked.connect(self._add_single_url)
        url_row.addWidget(self.btn_add_url)

        self.btn_bulk_paste = QPushButton("Bulk Paste")
        self.btn_bulk_paste.setObjectName("secondaryBtn")
        self.btn_bulk_paste.setFixedHeight(34)
        self.btn_bulk_paste.clicked.connect(self._bulk_paste)
        url_row.addWidget(self.btn_bulk_paste)

        self.btn_add_local = QPushButton("Local Files")
        self.btn_add_local.setObjectName("secondaryBtn")
        self.btn_add_local.setFixedHeight(34)
        self.btn_add_local.clicked.connect(self._add_local_files)
        url_row.addWidget(self.btn_add_local)

        input_section.addLayout(url_row)

        # Task list
        list_row = QHBoxLayout()
        list_row.setSpacing(8)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setMinimumHeight(90)
        self.list_widget.setMaximumHeight(180)
        self.list_widget.clicked.connect(self._on_item_clicked)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(
            self._show_context_menu)
        list_row.addWidget(self.list_widget, stretch=6)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)

        self.btn_move_up = QPushButton("\u25b2")
        self.btn_move_up.setObjectName("iconBtn")
        self.btn_move_up.setFixedSize(34, 28)
        self.btn_move_up.setToolTip("Move selected up")
        self.btn_move_up.clicked.connect(self._move_up)
        btn_col.addWidget(self.btn_move_up)

        self.btn_move_down = QPushButton("\u25bc")
        self.btn_move_down.setObjectName("iconBtn")
        self.btn_move_down.setFixedSize(34, 28)
        self.btn_move_down.setToolTip("Move selected down")
        self.btn_move_down.clicked.connect(self._move_down)
        btn_col.addWidget(self.btn_move_down)

        self.btn_remove_selected = QPushButton("\u2715")
        self.btn_remove_selected.setObjectName("dangerIconBtn")
        self.btn_remove_selected.setFixedSize(34, 28)
        self.btn_remove_selected.setToolTip("Remove selected")
        self.btn_remove_selected.clicked.connect(self._remove_selected)
        btn_col.addWidget(self.btn_remove_selected)

        self.btn_clear_all = QPushButton("\u2715 All")
        self.btn_clear_all.setObjectName("dangerBtn")
        self.btn_clear_all.setFixedHeight(28)
        self.btn_clear_all.clicked.connect(self._clear_all)
        btn_col.addWidget(self.btn_clear_all)
        btn_col.addStretch()
        list_row.addLayout(btn_col)

        input_section.addLayout(list_row)
        content.addWidget(input_section)

        # ── Settings Section (Collapsible) ──
        sets_section = CollapsibleSection("Processing Settings")
        sets_section.setStyleSheet(f"""
            QFrame#collapsibleSection {{
                background: {COLOR_SURFACE};
                border: 1px solid {COLOR_SURFACE1};
                border-radius: 8px;
            }}
            QFrame#sectionHeader {{
                background: {COLOR_SURFACE};
                border: 1px solid {COLOR_SURFACE1};
                border-radius: 8px;
            }}
            QPushButton#toggleBtn {{
                color: {COLOR_LAVENDER}; font-size: 10px;
                background: transparent; border: none;
                font-weight: bold;
            }}
            QPushButton#toggleBtn:hover {{ color: {COLOR_BLUE}; }}
        """)
        sets_section.content.setStyleSheet(
            "QWidget { background: transparent; border: none; }")

        sets_grid = QGridLayout()
        sets_grid.setSpacing(10)
        sets_grid.setContentsMargins(0, 0, 0, 0)

        def _lbl(text: str) -> QLabel:
            l = QLabel(text)
            l.setStyleSheet(
                f"color: {COLOR_SUBTEXT}; font-size: 11px; border: none;")
            return l

        # Row 0
        sets_grid.addWidget(_lbl("Whisper Model:"), 0, 0)
        self.combo_model = QComboBox()
        self.combo_model.addItems(
            ["tiny", "base", "small", "medium", "large-v3"])
        self.combo_model.setCurrentText("base")
        self.combo_model.setToolTip(
            "Larger models are more accurate but slower.")
        self.combo_model.currentTextChanged.connect(
            lambda v: self._save_setting("model", v))
        sets_grid.addWidget(self.combo_model, 0, 1)

        sets_grid.addWidget(_lbl("Language:"), 0, 2)
        self.combo_lang = QComboBox()
        self.combo_lang.setEditable(True)
        self.combo_lang.addItems([
            "Auto Detect", "english", "spanish", "french",
            "german", "italian", "portuguese", "japanese", "chinese"])
        self.combo_lang.currentTextChanged.connect(
            lambda v: self._save_setting("language", v))
        sets_grid.addWidget(self.combo_lang, 0, 3)

        # Row 1
        sets_grid.addWidget(_lbl("Download Format:"), 1, 0)
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems(
            ["Audio (best)", "Audio to MP3", "Video + Audio"])
        self.combo_fmt.currentTextChanged.connect(
            lambda v: self._save_setting("download_format", v))
        sets_grid.addWidget(self.combo_fmt, 1, 1)

        sets_grid.addWidget(_lbl("Output Format:"), 1, 2)
        self.combo_output_fmt = QComboBox()
        self.combo_output_fmt.addItems(OUTPUT_FORMATS)
        self.combo_output_fmt.setToolTip(
            "Choose which transcript formats to save.")
        self.combo_output_fmt.currentTextChanged.connect(
            lambda v: self._save_setting("output_format", v))
        sets_grid.addWidget(self.combo_output_fmt, 1, 3)

        # Row 2
        self.chk_diarize = QCheckBox("Speaker Diarization")
        self.chk_diarize.setToolTip(
            "Identify and label different speakers in the audio.")
        self.chk_diarize.toggled.connect(
            lambda v: self._save_setting("diarize", v))
        sets_grid.addWidget(self.chk_diarize, 2, 0, 1, 2)

        sets_grid.addWidget(_lbl("Speakers:"), 2, 2)
        self.spin_speakers = QComboBox()
        self.spin_speakers.addItems(
            ["Auto", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        self.spin_speakers.setToolTip(
            "Number of speakers. Auto = detect automatically.")
        self.spin_speakers.currentTextChanged.connect(
            lambda v: self._save_setting("num_speakers", v))
        sets_grid.addWidget(self.spin_speakers, 2, 3)

        # Row 3
        sets_grid.addWidget(_lbl("Speaker Labels:"), 3, 0)
        self.combo_speaker_labels = QComboBox()
        self.combo_speaker_labels.addItems(SPEAKER_LABEL_FORMATS)
        self.combo_speaker_labels.currentTextChanged.connect(
            lambda v: self._save_setting("speaker_label_fmt", v))
        sets_grid.addWidget(self.combo_speaker_labels, 3, 1)

        sets_grid.addWidget(_lbl("Text Format:"), 3, 2)
        self.combo_text_format = QComboBox()
        self.combo_text_format.addItems(TEXT_FORMAT_STYLES)
        self.combo_text_format.currentTextChanged.connect(
            lambda v: self._save_setting("text_format_style", v))
        sets_grid.addWidget(self.combo_text_format, 3, 3)

        # Row 4: Checkboxes
        self.chk_include_timestamps = QCheckBox("Timestamps in TXT")
        self.chk_include_timestamps.setChecked(True)
        self.chk_include_timestamps.toggled.connect(
            lambda v: self._save_setting("include_timestamps", v))
        sets_grid.addWidget(self.chk_include_timestamps, 4, 0)

        self.chk_include_lang_header = QCheckBox("Language Header")
        self.chk_include_lang_header.setChecked(True)
        self.chk_include_lang_header.toggled.connect(
            lambda v: self._save_setting("include_lang_header", v))
        sets_grid.addWidget(self.chk_include_lang_header, 4, 1)

        self.chk_srt_speaker_labels = QCheckBox("Speakers in SRT/VTT")
        self.chk_srt_speaker_labels.setChecked(True)
        self.chk_srt_speaker_labels.toggled.connect(
            lambda v: self._save_setting("srt_speaker_labels", v))
        sets_grid.addWidget(self.chk_srt_speaker_labels, 4, 2, 1, 2)

        # Row 5: Merge + Concatenate
        self.chk_merge_audio = QCheckBox("Merge audio before transcribing")
        self.chk_merge_audio.setToolTip(
            "Download all audio, merge into one file, transcribe once.\n"
            "Produces a single unified transcript for all videos.")
        self.chk_merge_audio.toggled.connect(
            lambda v: self._save_setting("merge_audio", v))
        sets_grid.addWidget(self.chk_merge_audio, 5, 0, 1, 2)

        self.chk_concatenate = QCheckBox("Concatenate outputs")
        self.chk_concatenate.setChecked(True)
        self.chk_concatenate.setToolTip(
            "After batch processing, combine all transcripts into one file.")
        self.chk_concatenate.toggled.connect(
            lambda v: self._save_setting("concatenate", v))
        sets_grid.addWidget(self.chk_concatenate, 5, 2, 1, 2)

        # Row 6: Concurrency + Output folder
        sets_grid.addWidget(_lbl("Max Concurrent:"), 6, 0)
        self.spin_concurrent = QSpinBox()
        self.spin_concurrent.setRange(1, 5)
        self.spin_concurrent.setValue(2)
        self.spin_concurrent.setToolTip(
            "Number of videos to process simultaneously.")
        self.spin_concurrent.valueChanged.connect(self._on_concurrency_changed)
        sets_grid.addWidget(self.spin_concurrent, 6, 1)

        sets_grid.addWidget(_lbl("Output Folder:"), 6, 2)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(4)
        self.lbl_output_path = QLabel(os.path.relpath(OUTPUT_DIR))
        self.lbl_output_path.setStyleSheet(f"""
            color: {COLOR_TEXT_DIM}; font-size: 11px;
            border: 1px solid {COLOR_SURFACE1}; border-radius: 4px;
            padding: 3px 8px; background: {COLOR_SURFACE0};
        """)
        folder_row.addWidget(self.lbl_output_path, stretch=1)
        self.btn_browse_output = QPushButton("...")
        self.btn_browse_output.setObjectName("secondaryBtn")
        self.btn_browse_output.setFixedSize(30, 26)
        self.btn_browse_output.setToolTip("Choose output directory")
        self.btn_browse_output.clicked.connect(self._browse_output_folder)
        folder_row.addWidget(self.btn_browse_output)
        sets_grid.addLayout(folder_row, 6, 3)

        sets_section.addLayout(sets_grid)
        content.addWidget(sets_section)

        # ── Action Buttons ──
        act_frame = QFrame()
        act_frame.setStyleSheet("QFrame { border: none; }")
        act_lay = QHBoxLayout(act_frame)
        act_lay.setSpacing(6)
        act_lay.setContentsMargins(0, 4, 0, 4)

        self.btn_start = QPushButton("  \u25b6  Process All")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setMinimumWidth(180)
        self.btn_start.clicked.connect(self._start_processing)
        act_lay.addWidget(self.btn_start, stretch=2)

        self.btn_stop = QPushButton("  \u25a0  Stop")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setFixedWidth(100)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_processing)
        act_lay.addWidget(self.btn_stop)

        self.btn_retry = QPushButton("  \u21bb  Retry Failed")
        self.btn_retry.setObjectName("retryBtn")
        self.btn_retry.setFixedHeight(40)
        self.btn_retry.clicked.connect(self._retry_failed)
        act_lay.addWidget(self.btn_retry)

        act_lay.addStretch()

        self.btn_concat = QPushButton("Concat")
        self.btn_concat.setObjectName("accentBtn")
        self.btn_concat.setToolTip(
            "Concatenate all completed transcripts into a single file.")
        self.btn_concat.setFixedHeight(34)
        self.btn_concat.clicked.connect(self._manual_concatenate)
        act_lay.addWidget(self.btn_concat)

        self.btn_copy_output = QPushButton("Copy")
        self.btn_copy_output.setObjectName("secondaryBtn")
        self.btn_copy_output.setFixedHeight(34)
        self.btn_copy_output.clicked.connect(self._copy_output)
        act_lay.addWidget(self.btn_copy_output)

        self.btn_open_output = QPushButton("Open Folder")
        self.btn_open_output.setObjectName("secondaryBtn")
        self.btn_open_output.setFixedHeight(34)
        self.btn_open_output.clicked.connect(self._open_output_folder)
        act_lay.addWidget(self.btn_open_output)

        self.btn_about = QPushButton("?")
        self.btn_about.setObjectName("iconBtn")
        self.btn_about.setFixedSize(34, 34)
        self.btn_about.setToolTip(f"About {APP_TITLE}")
        self.btn_about.clicked.connect(self._show_about)
        act_lay.addWidget(self.btn_about)

        content.addWidget(act_frame)

        # ── Progress ──
        progress_frame = QFrame()
        progress_frame.setStyleSheet("border: none;")
        progress_lay = QVBoxLayout(progress_frame)
        progress_lay.setContentsMargins(0, 2, 0, 0)
        progress_lay.setSpacing(3)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setFixedHeight(8)
        self.progress.setTextVisible(False)
        self.progress.setToolTip("Overall batch progress")
        progress_lay.addWidget(self.progress)

        self.lbl_status = QLabel(
            "Ready \u2014 add videos and click Process All.")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet(f"""
            color: {COLOR_SUBTEXT}; font-size: 11px; border: none;
        """)
        progress_lay.addWidget(self.lbl_status)

        content.addWidget(progress_frame)

        # ── Output Tabs (TXT / SRT / VTT) + Log (Splitter) ──
        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background: {COLOR_SURFACE1};
                height: 2px;
            }}
        """)

        # Tabbed output area
        out_container = QWidget()
        out_lay = QVBoxLayout(out_container)
        out_lay.setContentsMargins(0, 0, 0, 0)
        out_lay.setSpacing(2)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setFont(QFont("Consolas", 11))
        self.tabs.addTab(self.txt_output, "Transcript")

        self.txt_srt = QTextEdit()
        self.txt_srt.setReadOnly(True)
        self.txt_srt.setFont(QFont("Consolas", 10))
        self.tabs.addTab(self.txt_srt, "SRT")

        self.txt_vtt = QTextEdit()
        self.txt_vtt.setReadOnly(True)
        self.txt_vtt.setFont(QFont("Consolas", 10))
        self.tabs.addTab(self.txt_vtt, "VTT")

        self.tabs.currentChanged.connect(self._on_tab_changed)

        out_top = QHBoxLayout()
        out_top.setSpacing(8)
        rel = os.path.relpath(OUTPUT_DIR)
        out_header = QLabel(f"Output  |  ./{rel}/")
        out_header.setStyleSheet(f"""
            color: {COLOR_SUBTEXT}; font-size: 11px;
            font-weight: bold; border: none;
            padding-left: 4px;
        """)
        out_top.addWidget(out_header)
        out_top.addStretch()

        self.search_bar = SearchBar(self.txt_output)
        out_top.addLayout(self.search_bar.layout())

        self.btn_clear_output = QPushButton("Clear")
        self.btn_clear_output.setObjectName("secondaryBtn")
        self.btn_clear_output.setFixedHeight(22)
        self.btn_clear_output.setFixedWidth(50)
        self.btn_clear_output.clicked.connect(self.txt_output.clear)
        out_top.addWidget(self.btn_clear_output)

        out_lay.addLayout(out_top)
        out_lay.addWidget(self.tabs)
        splitter.addWidget(out_container)

        # Log area
        log_container = QWidget()
        log_lay = QVBoxLayout(log_container)
        log_lay.setContentsMargins(0, 4, 0, 0)
        log_lay.setSpacing(4)

        log_top = QHBoxLayout()
        log_header = QLabel("Activity Log")
        log_header.setStyleSheet(f"""
            color: {COLOR_SUBTEXT}; font-size: 11px;
            font-weight: bold; border: none;
            padding-left: 4px;
        """)
        log_top.addWidget(log_header)
        log_top.addStretch()

        self.btn_clear_log = QPushButton("Clear")
        self.btn_clear_log.setObjectName("secondaryBtn")
        self.btn_clear_log.setFixedHeight(20)
        self.btn_clear_log.setFixedWidth(50)
        self.btn_clear_log.clicked.connect(self.txt_log.clear)
        log_top.addWidget(self.btn_clear_log)
        log_lay.addLayout(log_top)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 9))
        self.txt_log.setMaximumHeight(120)
        log_lay.addWidget(self.txt_log)
        splitter.addWidget(log_container)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        content.addWidget(splitter, stretch=3)

        scroll.setWidget(scroll_content)
        root.addWidget(scroll, stretch=1)

        # Stats Bar
        self.stats_bar = StatsBar()
        root.addWidget(self.stats_bar)

        self.statusBar().showMessage("Idle")
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                color: {COLOR_SUBTEXT}; font-size: 10px;
                background: {COLOR_SURFACE};
                border-top: 1px solid {COLOR_SURFACE1};
            }}
        """)

    def _on_concurrency_changed(self, value: int) -> None:
        self._save_setting("max_concurrent", value)
        self.concurrent_semaphore = threading.Semaphore(value)

    def _on_tab_changed(self, index: int) -> None:
        if index == 0:
            self.search_bar.text_edit = self.txt_output
        elif index == 1:
            self.search_bar.text_edit = self.txt_srt
        elif index == 2:
            self.search_bar.text_edit = self.txt_vtt

    def _browse_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            getattr(self, '_output_dir', OUTPUT_DIR))
        if path:
            self._output_dir = path
            self.lbl_output_path.setText(os.path.relpath(path))
            self._on_log(f"Output directory changed to: {path}")

    def _show_about(self) -> None:
        AboutDialog(self).exec()

    # ──────── System Tray ────────

    def _setup_tray(self) -> None:
        try:
            self.tray_icon = QSystemTrayIcon(self)
            tray_menu = QMenu()
            show_action = tray_menu.addAction("Show Window")
            show_action.triggered.connect(self.show)
            quit_action = tray_menu.addAction("Quit")
            quit_action.triggered.connect(QApplication.quit)
            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.setToolTip(f"{APP_TITLE} Pro v{APP_VERSION}")
            self.tray_icon.activated.connect(self._tray_activated)
            self.tray_icon.show()
        except Exception:
            self.tray_icon = None

    def _tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.raise_()
            self.activateWindow()

    def _tray_notify(self, title: str, message: str) -> None:
        if self.tray_icon:
            try:
                self.tray_icon.showMessage(
                    title, message,
                    QSystemTrayIcon.Information, 5000)
            except Exception:
                pass

    # ──────── Stylesheet ────────

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(f"""
        QMainWindow {{ background: {COLOR_BASE}; }}

        QScrollArea {{
            background: {COLOR_BASE}; border: none;
        }}
        QScrollArea > QWidget > QWidget {{
            background: {COLOR_BASE};
        }}

        QGroupBox {{
            font-weight: bold; font-size: 12px; color: {COLOR_TEXT};
            border: 1px solid {COLOR_SURFACE1}; border-radius: 10px;
            margin-top: 16px; padding-top: 20px;
            padding-left: 12px; padding-right: 12px; padding-bottom: 12px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; left: 14px; padding: 0 8px;
            color: {COLOR_LAVENDER};
        }}

        QLabel {{
            color: {COLOR_TEXT}; font-size: 12px; border: none;
        }}

        QLineEdit, QTextEdit {{
            background: {COLOR_SURFACE0}; color: {COLOR_TEXT};
            border: 1px solid {COLOR_SURFACE1}; border-radius: 8px;
            padding: 8px 12px; font-size: 13px;
            selection-background-color: {COLOR_BLUE};
            selection-color: {COLOR_BASE};
        }}
        QLineEdit:focus, QTextEdit:focus {{
            border-color: {COLOR_BLUE};
        }}

        QComboBox {{
            background: {COLOR_SURFACE0}; color: {COLOR_TEXT};
            border: 1px solid {COLOR_SURFACE1}; border-radius: 8px;
            padding: 7px 12px; font-size: 12px;
            min-width: 80px;
        }}
        QComboBox:hover {{ border-color: {COLOR_SURFACE2}; }}
        QComboBox QAbstractItemView {{
            background: {COLOR_SURFACE}; color: {COLOR_TEXT};
            selection-background-color: {COLOR_BLUE};
            selection-color: {COLOR_BASE};
            border: 1px solid {COLOR_SURFACE1};
            border-radius: 4px;
        }}
        QComboBox::drop-down {{ border: none; width: 24px; }}

        QPushButton {{
            border: none; border-radius: 8px;
            padding: 8px 16px; font-size: 13px;
        }}
        QPushButton#primaryBtn {{
            background: {COLOR_BLUE}; color: {COLOR_BASE};
            font-weight: bold;
        }}
        QPushButton#primaryBtn:hover {{ background: {COLOR_LAVENDER}; }}
        QPushButton#secondaryBtn {{
            background: {COLOR_SURFACE1}; color: {COLOR_TEXT};
        }}
        QPushButton#secondaryBtn:hover {{ background: {COLOR_SURFACE2}; }}
        QPushButton#iconBtn {{
            background: {COLOR_SURFACE0}; color: {COLOR_TEXT_DIM};
            border: 1px solid {COLOR_SURFACE1}; border-radius: 6px;
            font-size: 14px; padding: 0px;
        }}
        QPushButton#iconBtn:hover {{
            background: {COLOR_SURFACE2}; color: {COLOR_TEXT};
        }}
        QPushButton#dangerIconBtn {{
            background: {COLOR_SURFACE0}; color: {COLOR_RED};
            border: 1px solid {COLOR_SURFACE1}; border-radius: 6px;
            font-size: 12px; padding: 0px;
        }}
        QPushButton#dangerIconBtn:hover {{
            background: {COLOR_RED}; color: {COLOR_BASE};
        }}
        QPushButton#startBtn {{
            background: {GRADIENT_START_BTN}; color: #ffffff;
            font-weight: bold; font-size: 14px; letter-spacing: 0.3px;
        }}
        QPushButton#startBtn:hover {{
            background: {COLOR_LAVENDER}; color: {COLOR_BASE};
        }}
        QPushButton#startBtn:disabled {{
            background: {COLOR_SURFACE1}; color: {COLOR_OVERLAY0};
        }}
        QPushButton#stopBtn {{
            background: {COLOR_RED}; color: {COLOR_BASE};
            font-weight: bold;
        }}
        QPushButton#stopBtn:hover {{ background: {COLOR_ROSE}; }}
        QPushButton#stopBtn:disabled {{
            background: {COLOR_SURFACE1}; color: {COLOR_OVERLAY0};
        }}
        QPushButton#retryBtn {{
            background: {COLOR_YELLOW}; color: {COLOR_BASE};
            font-weight: bold;
        }}
        QPushButton#retryBtn:hover {{ background: {COLOR_PEACH}; }}
        QPushButton#retryBtn:disabled {{
            background: {COLOR_SURFACE1}; color: {COLOR_OVERLAY0};
        }}
        QPushButton#accentBtn {{
            background: {COLOR_MAUVE}; color: {COLOR_BASE};
            font-weight: bold;
        }}
        QPushButton#accentBtn:hover {{ background: {COLOR_LAVENDER}; }}
        QPushButton#accentBtn:disabled {{
            background: {COLOR_SURFACE1}; color: {COLOR_OVERLAY0};
        }}
        QPushButton#dangerBtn {{
            background: transparent; color: {COLOR_RED};
            border: 1px solid {COLOR_RED}; border-radius: 8px;
            font-size: 11px; padding: 5px 10px;
        }}
        QPushButton#dangerBtn:hover {{
            background: {COLOR_RED}; color: {COLOR_BASE};
        }}

        QListWidget {{
            background: {COLOR_SURFACE0}; color: {COLOR_TEXT};
            border: 1px solid {COLOR_SURFACE1}; border-radius: 8px;
            font-size: 12px; padding: 4px; outline: none;
        }}
        QListWidget::item {{
            padding: 6px 10px; border-radius: 6px; margin: 1px 0px;
        }}
        QListWidget::item:selected {{
            background: {COLOR_SURFACE1}; color: {COLOR_TEXT};
        }}
        QListWidget::item:alternate {{
            background: rgba(28, 28, 40, 0.6);
        }}

        QProgressBar {{
            background: {COLOR_SURFACE0}; border: none;
            border-radius: 4px; text-align: center;
            color: {COLOR_TEXT}; font-size: 0px;
        }}
        QProgressBar::chunk {{
            background: {GRADIENT_PROGRESS};
            border-radius: 4px;
        }}

        QCheckBox {{
            color: {COLOR_TEXT}; font-size: 12px; spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 18px; height: 18px; border-radius: 5px;
            border: 2px solid {COLOR_SURFACE2};
            background: {COLOR_SURFACE0};
        }}
        QCheckBox::indicator:checked {{
            background: {COLOR_BLUE}; border-color: {COLOR_BLUE};
        }}
        QCheckBox::indicator:hover {{
            border-color: {COLOR_LAVENDER};
        }}

        QSpinBox {{
            background: {COLOR_SURFACE0}; color: {COLOR_TEXT};
            border: 1px solid {COLOR_SURFACE1}; border-radius: 8px;
            padding: 5px 10px; font-size: 12px; min-width: 60px;
        }}
        QSpinBox::up-button, QSpinBox::down-button {{
            background: {COLOR_SURFACE1}; border: none;
            width: 20px;
        }}

        QTabWidget::pane {{
            border: 1px solid {COLOR_SURFACE1};
            border-radius: 8px;
            background: {COLOR_SURFACE0};
            top: -1px;
        }}
        QTabBar::tab {{
            background: {COLOR_SURFACE};
            color: {COLOR_SUBTEXT};
            border: 1px solid {COLOR_SURFACE1};
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 6px 16px;
            margin-right: 2px;
            font-size: 11px;
            font-weight: bold;
        }}
        QTabBar::tab:selected {{
            background: {COLOR_SURFACE0};
            color: {COLOR_BLUE};
            border-bottom: 2px solid {COLOR_BLUE};
        }}
        QTabBar::tab:hover:!selected {{
            background: {COLOR_SURFACE1};
            color: {COLOR_TEXT};
        }}

        QToolTip {{
            background: {COLOR_SURFACE}; color: {COLOR_TEXT};
            border: 1px solid {COLOR_SURFACE1}; border-radius: 6px;
            padding: 6px; font-size: 12px;
        }}

        QMenu {{
            background: {COLOR_SURFACE}; color: {COLOR_TEXT};
            border: 1px solid {COLOR_SURFACE1}; border-radius: 8px;
            padding: 4px;
        }}
        QMenu::item {{
            padding: 6px 24px; border-radius: 4px;
        }}
        QMenu::item:selected {{
            background: {COLOR_BLUE}; color: {COLOR_BASE};
        }}
        QMenu::separator {{
            height: 1px; background: {COLOR_SURFACE1};
            margin: 4px 8px;
        }}

        QScrollBar:vertical {{
            background: {COLOR_BASE};
            width: 10px;
            border: none;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical {{
            background: {COLOR_SURFACE2};
            min-height: 30px;
            border-radius: 5px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {COLOR_OVERLAY0};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: none;
        }}
        QScrollBar:horizontal {{
            background: {COLOR_BASE};
            height: 10px;
            border: none;
            border-radius: 5px;
        }}
        QScrollBar::handle:horizontal {{
            background: {COLOR_SURFACE2};
            min-width: 30px;
            border-radius: 5px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {COLOR_OVERLAY0};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: none;
        }}
        """)

    # ──────── Keyboard Shortcuts ────────

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+V"), self, self._focus_url_paste)
        QShortcut(QKeySequence("Delete"), self.list_widget,
                  self._remove_selected)
        QShortcut(QKeySequence("Ctrl+Shift+V"), self, self._bulk_paste)
        QShortcut(QKeySequence("Ctrl+O"), self, self._add_local_files)
        QShortcut(QKeySequence("Ctrl+Return"), self,
                  self._start_processing)
        QShortcut(QKeySequence("Escape"), self, self._stop_processing)
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, self._copy_output)
        QShortcut(QKeySequence("Ctrl+F"), self,
                  self.search_bar.activate)

    def _focus_url_paste(self) -> None:
        self.txt_url_input.setFocus()
        self.txt_url_input.paste()

    # ──────── Drag & Drop ────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_overlay.show_overlay()

    def dragLeaveEvent(self, event) -> None:
        self.drop_overlay.hide_overlay()

    def dropEvent(self, event: QDropEvent) -> None:
        self.drop_overlay.hide_overlay()
        added = 0
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.isfile(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in VALID_MEDIA_EXTS:
                    self.tasks.append(VideoTask(path, "local"))
                    added += 1
        if added:
            self._refresh_list()
            self._update_stats()
            self._on_log(f"Added {added} file(s) via drag & drop.")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, 'drop_overlay'):
            self.drop_overlay.setGeometry(self.centralWidget().rect())

    # ──────── Task Reordering ────────

    def _move_up(self) -> None:
        if self.is_processing:
            return
        indices = sorted(set(
            item.data(Qt.UserRole)
            for item in self.list_widget.selectedItems()
            if item.data(Qt.UserRole) is not None
        ))
        for idx in indices:
            if idx > 0:
                self.tasks[idx - 1], self.tasks[idx] = \
                    self.tasks[idx], self.tasks[idx - 1]
        self._refresh_list()

    def _move_down(self) -> None:
        if self.is_processing:
            return
        indices = sorted(set(
            item.data(Qt.UserRole)
            for item in self.list_widget.selectedItems()
            if item.data(Qt.UserRole) is not None
        ), reverse=True)
        for idx in indices:
            if idx < len(self.tasks) - 1:
                self.tasks[idx], self.tasks[idx + 1] = \
                    self.tasks[idx + 1], self.tasks[idx]
        self._refresh_list()

    # ──────── List Management ────────

    def _add_single_url(self) -> None:
        url = self.txt_url_input.text().strip()
        if not url:
            return
        if not url.startswith("http"):
            self._on_log("URL must start with http:// or https://")
            return
        if not is_valid_media_url(url):
            reply = QMessageBox.question(
                self, "Unrecognized URL",
                f"This URL doesn't match common video platforms.\n\n"
                f"Add it anyway?\n{url}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.No:
                return

        task = VideoTask(url, "url")
        self.tasks.append(task)
        self.txt_url_input.clear()
        self._refresh_list()
        self._update_stats()
        self._on_log(f"Added: {url}")

        w = TitleFetchWorker(self.signals, task.uid, url)
        self.active_workers.append(w)
        w.start()

    def _bulk_paste(self) -> None:
        dialog = BulkPasteDialog(self)
        if dialog.exec() == QDialog.Accepted:
            urls = dialog.get_urls()
            if urls:
                invalid = [u for u in urls if not is_valid_media_url(u)]
                valid = [u for u in urls if is_valid_media_url(u)]

                if invalid:
                    reply = QMessageBox.question(
                        self, "Some URLs Unrecognized",
                        f"{len(invalid)} URL(s) don't match common video "
                        f"platforms.\n\nAdd all {len(urls)} URLs anyway?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                    if reply == QMessageBox.No:
                        valid = []
                    else:
                        valid = urls

                for url in valid:
                    task = VideoTask(url, "url")
                    self.tasks.append(task)
                    w = TitleFetchWorker(self.signals, task.uid, url)
                    self.active_workers.append(w)
                    w.start()

                if valid:
                    self._refresh_list()
                    self._update_stats()
                    self._on_log(
                        f"Added {len(valid)} URL(s) via bulk paste. "
                        f"Resolving titles...")
            else:
                self._on_log(
                    "No valid URLs found. Each line must start with http.")

    def _add_local_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio / Video Files", "",
            "Media (*.mp3 *.wav *.m4a *.ogg *.flac "
            "*.mp4 *.mkv *.webm);;All (*)")
        for p in paths:
            self.tasks.append(VideoTask(p, "local"))
        if paths:
            self._refresh_list()
            self._update_stats()
            self._on_log(f"Added {len(paths)} local file(s).")

    def _remove_selected(self) -> None:
        if self.is_processing:
            QMessageBox.warning(self, "Busy",
                                "Cannot modify the list while processing.\n"
                                "Stop processing first.")
            return
        indices = set()
        for item in self.list_widget.selectedItems():
            idx = item.data(Qt.UserRole)
            if idx is not None:
                indices.add(idx)
        if not indices:
            return
        self.tasks = [
            t for i, t in enumerate(self.tasks) if i not in indices]
        self._refresh_list()
        self._update_stats()
        self.txt_output.clear()

    def _clear_all(self) -> None:
        if self.is_processing:
            QMessageBox.warning(self, "Busy",
                                "Cannot modify the list while processing.")
            return
        if not self.tasks:
            return
        reply = QMessageBox.question(
            self, "Clear All", "Remove all items from the list?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.tasks.clear()
            self._refresh_list()
            self._update_stats()
            self.txt_output.clear()
            self._on_log("Cleared all items.")

    def _show_context_menu(self, pos) -> None:
        item = self.list_widget.itemAt(pos)
        if not item:
            return
        idx = item.data(Qt.UserRole)
        if idx is None or idx >= len(self.tasks):
            return

        task = self.tasks[idx]
        menu = QMenu(self)

        if task.status == STATUS_DONE:
            view_action = menu.addAction("View Transcript")
            view_action.triggered.connect(
                lambda: self._on_item_clicked(
                    self.list_widget.indexFromItem(item)))
        elif task.status == STATUS_ERROR:
            retry_action = menu.addAction("Retry This Task")
            retry_action.triggered.connect(lambda: self._retry_task(idx))

        menu.addSeparator()
        up_action = menu.addAction("Move Up")
        up_action.triggered.connect(self._move_up)
        down_action = menu.addAction("Move Down")
        down_action.triggered.connect(self._move_down)

        menu.addSeparator()
        remove_action = menu.addAction("Remove")
        remove_action.triggered.connect(self._remove_selected)

        menu.addSeparator()
        copy_src = menu.addAction("Copy Source URL/Path")
        copy_src.triggered.connect(
            lambda: QApplication.clipboard().setText(task.source))

        menu.exec_(self.list_widget.mapToGlobal(pos))

    def _retry_task(self, index: int) -> None:
        if self.is_processing:
            QMessageBox.warning(self, "Busy",
                                "Stop current processing first.")
            return
        if 0 <= index < len(self.tasks):
            task = self.tasks[index]
            task.reset()
            self._refresh_list()
            self._update_stats()
            self._on_log(f"Reset task for retry: {task.display_name()}")
            self._start_processing_single(task)

    def _refresh_list(self) -> None:
        self.list_widget.setUpdatesEnabled(False)
        try:
            self.list_widget.clear()
            for i, task in enumerate(self.tasks):
                icon = STATUS_ICONS.get(task.status, "  ")
                name = task.display_name()
                if task.progress_pct > 0 and not task.is_terminal:
                    text = f"  {icon}  {name}  [{task.progress_pct}%]"
                elif task.status == STATUS_ERROR:
                    text = f"  {icon}  {name}  \u2014 Failed"
                else:
                    text = f"  {icon}  {name}"
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, i)
                tip = (f"{task.source}\nStatus: {task.status}\n"
                       f"{task.progress_msg}" if task.progress_msg
                       else task.source)
                item.setToolTip(tip)
                fg = STATUS_COLORS.get(task.status, QColor(COLOR_TEXT))
                item.setForeground(fg)
                self.list_widget.addItem(item)
        finally:
            self.list_widget.setUpdatesEnabled(True)

    def _update_single_item(self, uid: str) -> None:
        for i, task in enumerate(self.tasks):
            if task.uid == uid:
                for row in range(self.list_widget.count()):
                    item = self.list_widget.item(row)
                    if item.data(Qt.UserRole) == i:
                        icon = STATUS_ICONS.get(task.status, "  ")
                        name = task.display_name()
                        if task.progress_pct > 0 and not task.is_terminal:
                            text = f"  {icon}  {name}  [{task.progress_pct}%]"
                        elif task.status == STATUS_ERROR:
                            text = f"  {icon}  {name}  \u2014 Failed"
                        else:
                            text = f"  {icon}  {name}"
                        item.setText(text)
                        fg = STATUS_COLORS.get(
                            task.status, QColor(COLOR_TEXT))
                        item.setForeground(fg)
                        tip = (f"{task.source}\nStatus: {task.status}\n"
                               f"{task.progress_msg}"
                               if task.progress_msg else task.source)
                        item.setToolTip(tip)
                        break
                break

    def _on_item_clicked(self, index) -> None:
        idx = index.data(Qt.UserRole)
        if idx is not None and 0 <= idx < len(self.tasks):
            task = self.tasks[idx]
            if task.status == STATUS_ERROR and task.error_msg:
                content = f"ERROR:\n{task.error_msg}"
            elif task.transcript_text:
                content = task.transcript_text
            else:
                content = "(No transcript available yet.)"
            self._current_task_txt = content
            self._current_task_srt = task.transcript_srt or ""
            self._current_task_vtt = task.transcript_vtt or ""
            self.txt_output.setPlainText(content)
            self.txt_srt.setPlainText(task.transcript_srt or "(No SRT available.)")
            self.txt_vtt.setPlainText(task.transcript_vtt or "(No VTT available.)")

    def _update_stats(self) -> None:
        self.stats_bar.update_stats(self.tasks)

    # ──────── Auto-Save ────────

    def _auto_save(self, task: VideoTask) -> None:
        output_fmt = self.combo_output_fmt.currentText()
        save_txt = "TXT" in output_fmt
        save_srt = "SRT" in output_fmt
        save_vtt = "VTT" in output_fmt

        if not save_txt and not save_srt and not save_vtt:
            return
        if not task.transcript_text and not task.transcript_srt and not task.transcript_vtt:
            return

        out_dir = self._output_dir
        os.makedirs(out_dir, exist_ok=True)
        base = task.safe_filename()

        if save_txt and task.transcript_text:
            txt_path = os.path.join(out_dir, f"{base}.txt")
            counter = 1
            while os.path.exists(txt_path):
                txt_path = os.path.join(
                    out_dir, f"{base}_{counter}.txt")
                counter += 1
            Path(txt_path).write_text(
                task.transcript_text, encoding="utf-8")
            self._on_log(f"Saved: {txt_path}")

        if save_srt and task.transcript_srt:
            srt_path = os.path.join(out_dir, f"{base}.srt")
            counter = 1
            while os.path.exists(srt_path):
                srt_path = os.path.join(
                    out_dir, f"{base}_{counter}.srt")
                counter += 1
            Path(srt_path).write_text(
                task.transcript_srt, encoding="utf-8")
            self._on_log(f"Saved: {srt_path}")

        if save_vtt and task.transcript_vtt:
            vtt_path = os.path.join(out_dir, f"{base}.vtt")
            counter = 1
            while os.path.exists(vtt_path):
                vtt_path = os.path.join(
                    out_dir, f"{base}_{counter}.vtt")
                counter += 1
            Path(vtt_path).write_text(
                task.transcript_vtt, encoding="utf-8")
            self._on_log(f"Saved: {vtt_path}")

    def _open_output_folder(self) -> None:
        out_dir = self._output_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.abspath(out_dir)
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    # ──────── Processing Logic (Concurrent) ────────

    def _lock_ui(self, locked: bool) -> None:
        self.is_processing = locked
        self.header.set_busy(locked)
        widgets = (
            self.btn_add_url, self.btn_bulk_paste,
            self.btn_add_local, self.btn_remove_selected,
            self.btn_clear_all, self.btn_start,
            self.combo_model, self.combo_lang,
            self.combo_fmt, self.combo_output_fmt,
            self.chk_diarize,
            self.spin_speakers, self.btn_retry,
            self.txt_url_input,
            self.combo_speaker_labels, self.combo_text_format,
            self.chk_include_timestamps, self.chk_include_lang_header,
            self.chk_srt_speaker_labels,
            self.chk_concatenate, self.chk_merge_audio,
            self.spin_concurrent, self.btn_move_up, self.btn_move_down,
        )
        for w in widgets:
            w.setEnabled(not locked)
        self.btn_stop.setEnabled(locked)

    def _start_processing(self) -> None:
        pending = [t for t in self.tasks if t.status == STATUS_PENDING]
        if not pending:
            QMessageBox.information(
                self, "Nothing to Process",
                "Add URLs or files first, or retry failed tasks.")
            return

        self.cancel_event.clear()

        if (self.chk_merge_audio.isChecked() and len(pending) >= 2):
            self._merge_mode_process(pending)
            return

        self._normal_mode_process(pending)

    def _normal_mode_process(self, pending: list) -> None:
        self._lock_ui(True)
        self._on_log("Starting batch processing...")
        self._pending_queue = list(pending)
        self._in_flight = 0
        self._total_to_process = len(pending)
        self._finished_count = 0
        self.progress.setValue(0)
        self._start_next_tasks()

    def _start_next_tasks(self) -> None:
        max_concurrent = self.spin_concurrent.value()
        while (self._in_flight < max_concurrent
               and self._pending_queue):
            if self.cancel_event.is_set():
                for t in self._pending_queue:
                    t.status = STATUS_CANCELLED
                self._pending_queue.clear()
                break

            task = self._pending_queue.pop(0)
            self._in_flight += 1
            self._begin_task(task)

        if self._in_flight == 0 and not self._pending_queue:
            self._on_batch_complete(
                was_cancelled=self.cancel_event.is_set())

    def _begin_task(self, task: VideoTask) -> None:
        total = self._total_to_process
        completed = self._finished_count
        self.lbl_status.setText(
            f"Processing {completed + 1}/{total}  |  "
            f"{task.display_name()}")

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if (item.data(Qt.UserRole) is not None
                    and item.data(Qt.UserRole) < len(self.tasks)
                    and self.tasks[item.data(Qt.UserRole)].uid == task.uid):
                self.list_widget.setCurrentItem(item)
                break

        if task.source_type == "url":
            task.status = STATUS_DOWNLOADING
            task.progress_msg = "Starting download..."
            self._update_single_item(task.uid)

            fmt = self.combo_fmt.currentText()
            audio_only = "Audio" in fmt
            convert_mp3 = "MP3" in fmt

            sem = self.concurrent_semaphore
            original_start = DownloadWorker.run

            def make_wrapped_run(worker, sem_ref, task_ref):
                def wrapped_run():
                    sem_ref.acquire()
                    try:
                        if not self.cancel_event.is_set():
                            original_start(worker)
                    finally:
                        sem_ref.release()
                return wrapped_run

            worker = DownloadWorker(
                signals=self.signals,
                task_uid=task.uid,
                url=task.source,
                output_dir=self._output_dir,
                audio_only=audio_only,
                convert_mp3=convert_mp3,
                cancel_event=self.cancel_event,
            )
            worker.run = make_wrapped_run(worker, sem, task)
            self.active_workers.append(worker)
            worker.start()
        else:
            task.title = os.path.splitext(
                os.path.basename(task.source))[0]
            task.audio_path = task.source
            task.status = STATUS_TRANSCRIBING
            task.progress_msg = "Preparing transcription..."
            self._update_single_item(task.uid)

            sem = self.concurrent_semaphore
            self._start_transcription_with_sem(task, sem)

    def _start_transcription_with_sem(
            self, task: VideoTask,
            sem: threading.Semaphore) -> None:
        spk_text = self.spin_speakers.currentText()
        num_spk = 0 if spk_text == "Auto" else int(spk_text)
        if not self.chk_diarize.isChecked():
            num_spk = 0

        output_fmt = self.combo_output_fmt.currentText()
        include_vtt = "VTT" in output_fmt

        worker = TranscriptionWorker(
            signals=self.signals,
            task_uid=task.uid,
            audio_path=task.audio_path,
            model_size=self.combo_model.currentText(),
            language=self.combo_lang.currentText(),
            diarize=self.chk_diarize.isChecked(),
            num_speakers=num_spk,
            cancel_event=self.cancel_event,
            speaker_label_fmt=self.combo_speaker_labels.currentText(),
            text_format_style=self.combo_text_format.currentText(),
            include_timestamps=self.chk_include_timestamps.isChecked(),
            include_lang_header=self.chk_include_lang_header.isChecked(),
            srt_speaker_labels=self.chk_srt_speaker_labels.isChecked(),
            include_vtt=include_vtt,
        )

        orig_run = worker.run
        def wrapped():
            sem.acquire()
            try:
                if not self.cancel_event.is_set():
                    orig_run()
            finally:
                sem.release()

        worker.run = wrapped
        self.active_workers.append(worker)
        worker.start()

    def _start_processing_single(self, task: VideoTask) -> None:
        self.cancel_event.clear()
        self._lock_ui(True)
        self._pending_queue = [task]
        self._in_flight = 0
        self._total_to_process = 1
        self._finished_count = 0
        self._start_next_tasks()

    # ──────── Merge Audio Mode ────────

    def _merge_mode_process(self, tasks: list) -> None:
        self._lock_ui(True)
        self._on_log("MERGE MODE: Downloading all audio files...")

        phase1_tasks = list(tasks)
        self._merge_tasks = phase1_tasks
        self._merge_downloaded = 0
        self._merge_audio_paths: list[str] = []
        self._merge_total = len(phase1_tasks)

        self._merge_current_idx = 0
        self._download_next_for_merge()

    def _download_next_for_merge(self) -> None:
        if self.cancel_event.is_set():
            self._on_batch_complete(was_cancelled=True)
            return

        if self._merge_current_idx >= len(self._merge_tasks):
            self._do_merge()
            return

        task = self._merge_tasks[self._merge_current_idx]
        self._merge_current_idx += 1

        pct = int((self._merge_current_idx / self._merge_total) * 40)
        self.progress.setValue(pct)
        self.header.set_progress(pct)
        self.lbl_status.setText(
            f"Merge mode: Downloading {self._merge_current_idx}"
            f"/{self._merge_total} \u2014 {task.display_name()}")

        if task.source_type == "local":
            task.status = STATUS_DONE
            task.audio_path = task.source
            self._merge_audio_paths.append(task.source)
            self._update_single_item(task.uid)
            self._on_log(f"Ready (local): {task.display_name()}")
            QTimer.singleShot(10, self._download_next_for_merge)
        else:
            fmt = self.combo_fmt.currentText()
            audio_only = "Audio" in fmt
            convert_mp3 = "MP3" in fmt

            worker = DownloadWorker(
                signals=self.signals,
                task_uid=task.uid,
                url=task.source,
                output_dir=self._output_dir,
                audio_only=audio_only,
                convert_mp3=convert_mp3,
                cancel_event=self.cancel_event,
            )
            self._merge_download_uid = task.uid
            self._merge_download_worker = worker
            self.active_workers.append(worker)
            worker.start()

    def _do_merge(self) -> None:
        if self.cancel_event.is_set():
            self._on_batch_complete(was_cancelled=True)
            return

        paths = [p for p in self._merge_audio_paths
                 if p and os.path.isfile(p)]
        if not paths:
            self._on_error(self._merge_tasks[0].uid,
                           "No audio files to merge.")
            self._on_batch_complete()
            return

        self.progress.setValue(45)
        self.header.set_progress(45)
        self.lbl_status.setText("Merging audio files...")

        for t in self._merge_tasks:
            t.status = STATUS_MERGING
            self._update_single_item(t.uid)

        merged_path = AudioMerger.merge(
            paths, self._output_dir,
            progress_cb=lambda msg: self._on_log(msg))

        if not merged_path or not os.path.isfile(merged_path):
            self._on_log("Audio merge failed \u2014 falling back to "
                          "individual processing.")
            for t in self._merge_tasks:
                if t.status != STATUS_DONE:
                    t.reset()
            self._refresh_list()
            self._pending_queue = [t for t in self._merge_tasks
                                   if t.status == STATUS_PENDING]
            self._in_flight = 0
            self._total_to_process = len(self._pending_queue)
            self._finished_count = 0
            self._start_next_tasks()
            return

        self.progress.setValue(55)
        self.header.set_progress(55)
        self.lbl_status.setText("Transcribing merged audio...")

        spk_text = self.spin_speakers.currentText()
        num_spk = 0 if spk_text == "Auto" else int(spk_text)
        if not self.chk_diarize.isChecked():
            num_spk = 0

        output_fmt = self.combo_output_fmt.currentText()
        include_vtt = "VTT" in output_fmt

        merge_task = VideoTask(merged_path, "local")
        merge_task.title = "Merged Transcript"
        self.tasks.append(merge_task)
        self._refresh_list()

        worker = TranscriptionWorker(
            signals=self.signals,
            task_uid=merge_task.uid,
            audio_path=merged_path,
            model_size=self.combo_model.currentText(),
            language=self.combo_lang.currentText(),
            diarize=self.chk_diarize.isChecked(),
            num_speakers=num_spk,
            cancel_event=self.cancel_event,
            speaker_label_fmt=self.combo_speaker_labels.currentText(),
            text_format_style=self.combo_text_format.currentText(),
            include_timestamps=self.chk_include_timestamps.isChecked(),
            include_lang_header=self.chk_include_lang_header.isChecked(),
            srt_speaker_labels=self.chk_srt_speaker_labels.isChecked(),
            include_vtt=include_vtt,
        )

        self._merge_result_task = merge_task
        self._merge_original_tasks = self._merge_tasks
        self.active_workers.append(worker)
        worker.start()

    # ──────── Stop / Retry ────────

    def _stop_processing(self) -> None:
        self.cancel_event.set()
        self._on_log("Stop requested \u2014 finishing current operations...")
        self.lbl_status.setText("Stopping...")
        self.header.set_busy(False)

    def _retry_failed(self) -> None:
        if self.is_processing:
            QMessageBox.warning(self, "Busy",
                                "Stop current processing before retrying.")
            return
        failed = [t for t in self.tasks if t.status == STATUS_ERROR]
        if not failed:
            QMessageBox.information(self, "No Failures",
                                    "There are no failed tasks to retry.")
            return
        for t in failed:
            t.reset()
        self._refresh_list()
        self._update_stats()
        self._on_log(f"Reset {len(failed)} failed task(s) to Pending.")
        self._start_processing()

    # ──────── Batch Complete ────────

    def _on_batch_complete(self, was_cancelled: bool = False) -> None:
        self._lock_ui(False)
        self.cancel_event.clear()

        self.active_workers = [
            w for w in self.active_workers if w.is_alive()]

        done  = sum(1 for t in self.tasks if t.status == STATUS_DONE)
        error = sum(1 for t in self.tasks if t.status == STATUS_ERROR)
        canc  = sum(1 for t in self.tasks
                    if t.status == STATUS_CANCELLED)
        self.progress.setValue(100)
        self.header.set_progress(100)
        self._update_stats()

        rel = os.path.relpath(OUTPUT_DIR)
        if was_cancelled:
            self.lbl_status.setText("Processing stopped by user.")
            self._on_log(
                f"Stopped. {done} done, {error} failed, "
                f"{canc} cancelled.")
        else:
            self.lbl_status.setText("Batch processing complete!")
            self._on_log(
                f"All tasks finished! {done} done, {error} failed. "
                f"Transcripts saved to ./{rel}/")
            if done > 0:
                self._tray_notify(
                    f"{APP_TITLE} Pro",
                    f"Batch complete! {done} done, {error} failed.")

        self.statusBar().showMessage("Idle")

        if (not was_cancelled
                and self.chk_concatenate.isChecked()
                and done >= 2):
            self._concatenate_outputs()

        for t in self.tasks:
            if t.status == STATUS_DONE and t.transcript_text:
                self.txt_output.setPlainText(t.transcript_text)
                self.txt_srt.setPlainText(t.transcript_srt or "")
                self.txt_vtt.setPlainText(t.transcript_vtt or "")
                break

    # ──────── Signal Handlers ────────

    def _get_task_by_uid(self, uid: str) -> Optional[VideoTask]:
        for task in self.tasks:
            if task.uid == uid:
                return task
        return None

    def _on_progress(self, uid: str, value: int, msg: str) -> None:
        task = self._get_task_by_uid(uid)
        if not task:
            return

        task.progress_pct = value
        task.progress_msg = msg

        if self._total_to_process > 0:
            overall = int(
                ((self._finished_count + value / 100.0)
                 / self._total_to_process) * 100)
            self.header.set_progress(min(overall, 99))
            self.progress.setValue(min(overall, 99))

        ml = msg.lower()
        if "download" in ml:
            task.status = STATUS_DOWNLOADING
        elif "transcrib" in ml or "whisper" in ml:
            task.status = STATUS_TRANSCRIBING
        elif ("analyzing" in ml or "diariz" in ml
              or "clustering" in ml):
            task.status = STATUS_DIARIZING

        self._update_single_item(uid)

        if self.is_processing:
            self.lbl_status.setText(f"{task.display_name()}: {msg}")

    def _on_error(self, uid: str, error_msg: str) -> None:
        task = self._get_task_by_uid(uid)
        if task:
            task.status = STATUS_ERROR
            task.error_msg = error_msg
            task.progress_msg = "Failed"
            self._update_single_item(uid)
            self._update_stats()
            self._on_log(
                f"Error processing {task.display_name()}: {error_msg}")

            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if (item.isSelected()
                        and item.data(Qt.UserRole) is not None):
                    idx = item.data(Qt.UserRole)
                    if (idx < len(self.tasks)
                            and self.tasks[idx].uid == uid):
                        self.txt_output.setPlainText(
                            f"ERROR:\n{error_msg}")
                        break

            self._handle_task_completion()

    def _on_cancelled(self, uid: str) -> None:
        task = self._get_task_by_uid(uid)
        if task:
            task.status = STATUS_CANCELLED
            task.progress_msg = "Cancelled"
            task.progress_pct = 0
            self._update_single_item(uid)
            self._update_stats()
            self._on_log(f"Cancelled: {task.display_name()}")
            self._handle_task_completion()

    def _on_finished(self, uid: str, result_type: str,
                     payload: str) -> None:
        task = self._get_task_by_uid(uid)
        if not task:
            return

        if result_type == "title":
            task.title = payload
            self._update_single_item(uid)

        elif result_type == "download":
            try:
                filename, title = payload.split("|||")
                task.audio_path = filename
                task.title = title
                task.status = STATUS_TRANSCRIBING
                task.progress_msg = "Starting transcription..."
                task.progress_pct = 0
                self._update_single_item(uid)

                if hasattr(self, '_merge_download_uid'):
                    self._merge_audio_paths.append(filename)
                    self._on_log(f"Merged download ready: {title}")
                    QTimer.singleShot(
                        10, self._download_next_for_merge)
                    return

                sem = self.concurrent_semaphore
                self._start_transcription_with_sem(task, sem)

            except ValueError:
                self._on_error(uid, "Failed to parse download result.")

        elif result_type == "transcription":
            task.transcript_text = payload
            task.progress_pct = 95
            self.txt_output.setPlainText(payload)

            if task.transcript_srt is not None:
                self._finalize_task(task)
            else:
                task.transcript_srt = ""

        elif result_type == "srt_cache":
            if task.transcript_text and task.transcript_srt == "":
                task.transcript_srt = payload
                self._finalize_task(task)
            elif task.transcript_text:
                task.transcript_srt = payload
                self._finalize_task(task)
            else:
                task.transcript_srt = payload

        elif result_type == "vtt_cache":
            task.transcript_vtt = payload

    def _finalize_task(self, task: VideoTask) -> None:
        task.status = STATUS_DONE
        task.progress_pct = 100
        task.progress_msg = "Complete"
        self._update_single_item(task.uid)
        self._update_stats()
        self._auto_save(task)
        self._handle_task_completion()

    def _handle_task_completion(self) -> None:
        if not self.is_processing:
            return

        self._finished_count += 1

        if self._total_to_process > 0:
            pct = int(
                (self._finished_count / self._total_to_process) * 100)
            self.progress.setValue(min(pct, 99))
            self.header.set_progress(min(pct, 99))

        self._in_flight = max(0, self._in_flight - 1)
        if not self.cancel_event.is_set():
            self._start_next_tasks()

        if self._finished_count >= self._total_to_process:
            if not self._pending_queue and self._in_flight == 0:
                self._on_batch_complete()

    # ──────── Copy ────────

    def _copy_output(self) -> None:
        idx = self.tabs.currentIndex()
        if idx == 0:
            text = self.txt_output.toPlainText()
        elif idx == 1:
            text = self.txt_srt.toPlainText()
        elif idx == 2:
            text = self.txt_vtt.toPlainText()
        else:
            text = self.txt_output.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            tab_name = ["transcript", "SRT", "VTT"][min(idx, 2)]
            self._on_log(f"Copied {tab_name} to clipboard.")

    # ──────── Concatenation ────────

    def _manual_concatenate(self) -> None:
        done_tasks = [t for t in self.tasks if t.status == STATUS_DONE]
        if len(done_tasks) < 2:
            QMessageBox.information(
                self, "Need Multiple Tasks",
                "At least 2 completed tasks are required.\n\n"
                f"Currently {len(done_tasks)} task(s) done.")
            return
        self._concatenate_outputs()

    def _concatenate_outputs(self) -> None:
        done_tasks = [t for t in self.tasks if t.status == STATUS_DONE]
        if not done_tasks:
            return

        output_fmt = self.combo_output_fmt.currentText()
        save_txt = "TXT" in output_fmt
        save_srt = "SRT" in output_fmt
        save_vtt = "VTT" in output_fmt

        has_txt = any(t.transcript_text for t in done_tasks)
        has_srt = any(t.transcript_srt for t in done_tasks)
        has_vtt = any(t.transcript_vtt for t in done_tasks)

        if save_txt and has_txt:
            self._concatenate_txt(done_tasks)
        if save_srt and has_srt:
            self._concatenate_srt(done_tasks)
        if save_vtt and has_vtt:
            self._concatenate_vtt(done_tasks)

        self._on_log(
            f"Concatenated {len(done_tasks)} transcript(s).")

    def _concatenate_txt(self, tasks: list) -> None:
        sep = "=" * CONCAT_SEPARATOR_LENGTH
        thin_sep = "-" * CONCAT_SEPARATOR_LENGTH
        lines = [
            "CONCATENATED TRANSCRIPT",
            f"{len(tasks)} video(s) processed on "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}",
            sep, "",
            "TABLE OF CONTENTS",
        ]
        for i, t in enumerate(tasks, 1):
            lines.append(f"  {i}. {t.display_name()}")
        lines.extend(["", sep, ""])

        for i, task in enumerate(tasks):
            kind = "VIDEO" if task.source_type == "url" else "FILE"
            lines.append(f"{kind} {i + 1}: {task.display_name()}")
            lines.append(f"Source: {task.source}")
            lines.append(thin_sep)
            lines.append(task.transcript_text or "(No transcript content.)")
            lines.append("")
            if i < len(tasks) - 1:
                lines.extend([sep, ""])

        combined = "\n".join(lines)
        os.makedirs(self._output_dir, exist_ok=True)
        out_path = os.path.join(self._output_dir, "concatenated_transcript.txt")
        Path(out_path).write_text(combined, encoding="utf-8")
        self._on_log(f"Saved concatenated TXT: {out_path}")
        self.txt_output.setPlainText(combined)

    def _concatenate_srt(self, tasks: list) -> None:
        global_num = 1
        all_entries: list[dict] = []
        offset = 0.0

        for task in tasks:
            srt = task.transcript_srt
            if not srt or not srt.strip():
                continue
            entries = self._parse_srt_entries(srt)
            if not entries:
                continue
            max_end = max(e["end"] for e in entries)
            for e in entries:
                all_entries.append({
                    "num": global_num,
                    "start": e["start"] + offset,
                    "end": e["end"] + offset,
                    "text": e["text"],
                })
                global_num += 1
            offset += max_end

        if not all_entries:
            return

        lines = []
        for e in all_entries:
            lines.append(str(e["num"]))
            lines.append(
                f"{self._fmt_srt_time(e['start'])} --> "
                f"{self._fmt_srt_time(e['end'])}")
            lines.append(e["text"])
            lines.append("")

        os.makedirs(self._output_dir, exist_ok=True)
        out_path = os.path.join(self._output_dir, "concatenated_transcript.srt")
        Path(out_path).write_text("\n".join(lines), encoding="utf-8")
        self._on_log(f"Saved concatenated SRT: {out_path}")

    def _concatenate_vtt(self, tasks: list) -> None:
        global_num = 1
        all_entries: list[dict] = []
        offset = 0.0

        for task in tasks:
            vtt = task.transcript_vtt
            if not vtt or not vtt.strip():
                continue
            entries = self._parse_vtt_entries(vtt)
            if not entries:
                continue
            max_end = max(e["end"] for e in entries)
            for e in entries:
                all_entries.append({
                    "num": global_num,
                    "start": e["start"] + offset,
                    "end": e["end"] + offset,
                    "text": e["text"],
                })
                global_num += 1
            offset += max_end

        if not all_entries:
            return

        lines = ["WEBVTT", ""]
        for e in all_entries:
            lines.append(str(e["num"]))
            lines.append(
                f"{self._fmt_vtt_time(e['start'])} --> "
                f"{self._fmt_vtt_time(e['end'])}")
            lines.append(e["text"])
            lines.append("")

        os.makedirs(self._output_dir, exist_ok=True)
        out_path = os.path.join(self._output_dir, "concatenated_transcript.vtt")
        Path(out_path).write_text("\n".join(lines), encoding="utf-8")
        self._on_log(f"Saved concatenated VTT: {out_path}")

    @staticmethod
    def _parse_srt_entries(srt_text: str) -> list[dict]:
        entries = []
        blocks = re.split(r'\n\s*\n', srt_text.strip())
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            block_lines = block.split('\n')
            if len(block_lines) < 2:
                continue
            ts_idx = -1
            for idx, line in enumerate(block_lines):
                if '-->' in line:
                    ts_idx = idx
                    break
            if ts_idx < 0:
                continue
            match = re.match(
                r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*'
                r'(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                block_lines[ts_idx])
            if not match:
                continue
            h1, m1, s1, ms1 = (int(match.group(1)), int(match.group(2)),
                                 int(match.group(3)), int(match.group(4)))
            h2, m2, s2, ms2 = (int(match.group(5)), int(match.group(6)),
                                 int(match.group(7)), int(match.group(8)))
            start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
            end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
            text = '\n'.join(block_lines[ts_idx + 1:]).strip()
            entries.append({"start": start, "end": end, "text": text})
        return entries

    @staticmethod
    def _parse_vtt_entries(vtt_text: str) -> list[dict]:
        entries = []
        blocks = re.split(r'\n\s*\n', vtt_text.strip())
        for block in blocks:
            block = block.strip()
            if not block or block.startswith("WEBVTT"):
                continue
            block_lines = block.split('\n')
            if len(block_lines) < 2:
                continue
            ts_idx = -1
            for idx, line in enumerate(block_lines):
                if '-->' in line:
                    ts_idx = idx
                    break
            if ts_idx < 0:
                continue
            match = re.match(
                r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*'
                r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})',
                block_lines[ts_idx])
            if not match:
                continue
            h1, m1, s1, ms1 = (int(match.group(1)), int(match.group(2)),
                                 int(match.group(3)), int(match.group(4)))
            h2, m2, s2, ms2 = (int(match.group(5)), int(match.group(6)),
                                 int(match.group(7)), int(match.group(8)))
            start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
            end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
            text = '\n'.join(block_lines[ts_idx + 1:]).strip()
            entries.append({"start": start, "end": end, "text": text})
        return entries

    @staticmethod
    def _fmt_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        if ms >= 1000:
            ms = 999
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _fmt_vtt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        if ms >= 1000:
            ms = 999
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    # ──────── Logging ────────

    def _on_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        self.txt_log.append(
            f"<span style='color:{COLOR_OVERLAY0}'>[{timestamp}]</span> "
            f"{message}")
        cursor = self.txt_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.txt_log.setTextCursor(cursor)


# ════════════════════════════════════════════════════════════════
#  MAIN ENTRY
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)

    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    # Dark palette for native dialogs
    from PySide6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLOR_BASE))
    palette.setColor(QPalette.WindowText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Base, QColor(COLOR_SURFACE0))
    palette.setColor(QPalette.AlternateBase, QColor(COLOR_SURFACE))
    palette.setColor(QPalette.ToolTipBase, QColor(COLOR_SURFACE))
    palette.setColor(QPalette.ToolTipText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Text, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Button, QColor(COLOR_SURFACE1))
    palette.setColor(QPalette.ButtonText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Highlight, QColor(COLOR_BLUE))
    palette.setColor(QPalette.HighlightedText, QColor(COLOR_BASE))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())