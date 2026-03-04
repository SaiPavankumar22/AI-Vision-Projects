# Face Puzzle

A real-time gesture-controlled sliding puzzle game that captures your face from a webcam and lets you solve it using hand gestures — no mouse or keyboard needed during gameplay.

## Demo

| Step | What happens |
|------|-------------|
| 1 | Hold an open palm in front of the camera for 1 second |
| 2 | Your face is automatically detected and captured |
| 3 | The image is sliced into a 3×3 grid and shuffled |
| 4 | Pinch your fingers to grab and drag blocks into place |
| 5 | Solve the puzzle — confetti and timer on completion |

## Features

- **Zero-touch UI** — fully controlled by hand gestures via MediaPipe
- **Stable pinch detection** — requires 2 consecutive pinch frames to grab, 3 to release, eliminating flicker
- **Smooth animation** — lerped visual positions for blocks, pulsing glow on selected piece, drop-shadow
- **Particle effects** — snap burst on every correct placement, confetti rain on solve
- **Live debug overlay** — FPS, hand/palm/pinch state, face detection status (toggle with `d`)
- **Auto camera scan** — tries indices 0–3 and picks the first working webcam

## Requirements

- Python 3.8+
- Webcam

## Installation

```bash
pip install opencv-python mediapipe==0.10.9 numpy
```

## Usage

```bash
python face_puzzle.py
```

## Controls

| Gesture / Key | Action |
|---------------|--------|
| Open palm held for 1 s | Start game / Reset |
| Pinch (thumb + index) | Grab a puzzle block |
| Drag while pinching | Move the grabbed block |
| Release pinch | Drop block (snaps to nearest slot) |
| `q` | Quit |
| `r` | Reset immediately |
| `d` | Toggle debug overlay |

## How It Works

### State Machine

```
WAITING_FOR_PALM → CAPTURE_FACE → SHUFFLE_PUZZLE → PLAYING → SOLVED
        ↑_____________________reset___________________________|
```

| State | Description |
|-------|-------------|
| `WAITING_FOR_PALM` | Shows a progress bar while the user holds an open palm |
| `CAPTURE_FACE` | Runs MediaPipe face detection; crops and stores the face image |
| `SHUFFLE_PUZZLE` | Slices the face into a 3×3 grid, randomises positions, shows "Get Ready!" |
| `PLAYING` | Gesture-driven drag-and-drop; tracks elapsed time |
| `SOLVED` | Shows solve time and animated confetti; auto-resets after 5 s |

### Gesture Detection

- **Open palm** — thumb extended outward and all four fingertips above their respective PIP joints
- **Pinch** — normalised Euclidean distance between thumb tip (landmark 4) and index tip (landmark 8) below `PINCH_THRESH = 0.055`

### Tunable Constants

| Constant | Default | Description |
|----------|---------|-------------|
| `GRID` | `3` | Puzzle grid size (3 → 3×3 = 9 pieces) |
| `PUZZLE_SIZE` | `450` | Puzzle canvas size in pixels |
| `PINCH_THRESH` | `0.055` | Normalised distance to confirm a pinch |
| `PINCH_RELEASE` | `0.075` | Normalised distance to confirm a release |
| `PALM_HOLD_SEC` | `1.0` | Seconds palm must be held to trigger state change |
| `SNAP_DIST` | `CELL × 0.6` | Max pixel distance to snap a block to a slot |
| `SOLVED_SHOW` | `5.0` | Seconds the solved screen is shown before auto-reset |

## Project Structure

```
face-puzzle/
├── face_puzzle.py   # Single-file application
└── README.md
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | latest | Camera capture, drawing, image processing |
| `mediapipe` | 0.10.9 | Hand tracking and face detection |
| `numpy` | latest | Array operations for landmark data |

## Troubleshooting

**No webcam found** — ensure a webcam is connected and not in use by another application. The app scans indices 0–3 automatically.

**Face not detected** — ensure good, even lighting and that your face is fully visible and centred in the frame.

**Pinch too sensitive / not sensitive enough** — adjust `PINCH_THRESH` (lower = tighter pinch required) and `PINCH_RELEASE` in the constants section at the top of `face_puzzle.py`.

**Low FPS** — close other applications using the webcam or GPU. Reducing `PUZZLE_SIZE` or `GRID` also lowers rendering cost.

## License

MIT
