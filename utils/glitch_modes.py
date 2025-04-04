from utils.image_glitch import (
    glitch_bass_treble,
    glitch_pitch_shift,
    glitch_change_speed,
    glitch_change_tempo,
)

GLITCH_MODES = {
    "change_speed": {
        "func": glitch_change_speed,
        "description": "Change playback speed (faster/slower).",
        "per_channel": True,
    },
    "change_tempo": {
        "func": glitch_change_tempo,
        "description": "Stretch/compress tempo while preserving pitch.",
        "per_channel": False,
        "needs_width": True
    },
    "bass_treble": {
    "func": glitch_bass_treble,
    "description": "Block-based bass and treble filter for RGB smearing.",
    "needs_width": True
    },
    "pitch_shift": {
        "func": glitch_pitch_shift,
        "description": "Shift pitch up/down by resampling.",
        "per_channel": True,
    }
}
