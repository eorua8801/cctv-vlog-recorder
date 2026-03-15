# CCTV-Style Vlog Recorder

Real-time CCTV-style vlog recorder with object detection and face anonymization for NVIDIA Jetson Orin Nano Super.

![Demo](https://img.shields.io/badge/FPS-30-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10-blue) ![YOLO](https://img.shields.io/badge/YOLO-v8n-orange)

## Features

- 🎥 **Real-time Object Detection** - YOLOv8n/YOLO11n for detecting people, objects
- 😎 **Face Anonymization** - Multiple glitch effects (mosaic, RGB shift, pixel shuffle)
- 🚀 **30 FPS Performance** - Optimized frame-skip detection on Jetson Orin Nano
- 📹 **CCTV Aesthetic** - Timestamp, FPS counter, surveillance-style overlay
- ⚡ **Multiple Detection Methods** - YOLO-Face, MediaPipe, or Hybrid
- 🎨 **Customizable Effects** - 8+ different glitch/mosaic styles
- 💾 **Video Recording** - MP4 output with configurable resolution
- 🖥️ **GUI Application** - Easy-to-use camera app interface with one-click recording

## Hardware Requirements

- **NVIDIA Jetson Orin Nano Super** (tested)
- Camera (USB webcam, CSI camera, RealSense, etc.)
- JetPack 6.1+

## Installation

### 1. Install Dependencies

```bash
pip3 install ultralytics opencv-python mediapipe huggingface_hub pillow
```

### 2. Download YOLO Face Detection Model

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='arnabdhar/YOLOv8-Face-Detection',
    filename='model.pt',
    local_dir='.'
)
mv model.pt yolov8-face.pt
"
```

### 3. Clone Repository

```bash
git clone <your-repo-url>
cd <repo-directory>
chmod +x vlog_recorder.py
```

## Quick Start

### GUI Mode (Recommended - Camera App Style)

**Launch the GUI application**:
```bash
python3 vlog_recorder_gui.py
```

Or double-click the desktop launcher: **CCTV Vlog Recorder**

**Features**:
- 📹 Real-time preview with all effects applied
- ⏺️ One-click start/stop recording
- 📷 Screenshot button
- ⚙️ Settings panel for camera, model, and effect selection
- 📊 Live FPS and status display
- 💾 Auto-generated filenames with timestamp

**Controls**:
- Click **"⏺ START RECORDING"** to begin recording
- Click **"⏹ STOP RECORDING"** to save video
- Click **"📷 Screenshot"** to capture a frame
- Click **"⚙ Settings"** to adjust camera/detection settings

### Command-Line Mode (Advanced)

**Basic Usage (Preview Only)**:
```bash
python3 vlog_recorder.py --camera 1
```

**Record with Auto-Generated Filename**:
```bash
python3 vlog_recorder.py --camera 1 --record
```

**Headless Mode (SSH/Remote)**:
```bash
python3 vlog_recorder.py --headless --camera 1 --duration 60 --output my_vlog.mp4
```

## Performance Optimization

### 30 FPS Setup (Recommended)

```bash
python3 vlog_recorder.py \
  --camera 1 \
  --model yolov8n.pt \
  --resolution 640x480 \
  --face-detector yolo \
  --glitch mosaic_heavy \
  --detection-interval 3 \
  --record
```

### High Quality (13 FPS)

```bash
python3 vlog_recorder.py \
  --camera 1 \
  --model yolo11n.pt \
  --resolution 1280x720 \
  --detection-interval 1 \
  --glitch mosaic_shuffle \
  --record
```

## Configuration Options

### Camera & Model

| Option | Default | Description |
|--------|---------|-------------|
| `--camera` | 0 | Camera device ID |
| `--model` | yolo11n.pt | YOLO model (yolov8n.pt, yolo11n.pt, yolo26n.pt) |
| `--resolution` | 640x480 | Camera resolution (320x240, 640x480, 1280x720, 1920x1080) |

### Detection

| Option | Default | Description |
|--------|---------|-------------|
| `--face-detector` | yolo | Face detection method (yolo, mediapipe, hybrid) |
| `--confidence` | 0.3 | Object detection confidence threshold |
| `--detection-interval` | 1 | Run detection every N frames (1=every frame, 3=30 FPS boost) |

### Effects

| Option | Default | Description |
|--------|---------|-------------|
| `--glitch` | combined | Glitch effect type |

**Available Effects:**
- `rgb_shift` - RGB channel shift (cyberpunk style)
- `pixelate` - Standard pixelation
- `noise` - Digital noise
- `scanlines` - VHS scanlines
- `mosaic_heavy` - Large pixel mosaic
- `pixelate_shuffle` - Shuffled pixel blocks
- `mosaic_shuffle` - Heavy mosaic + shuffle (strongest anonymization)
- `combined` - All effects combined

### Output

| Option | Default | Description |
|--------|---------|-------------|
| `--record` | False | Enable recording with auto-generated filename |
| `--output` | None | Custom output video path |
| `--duration` | None | Recording duration in seconds (headless mode) |
| `--headless` | False | Run without GUI (for SSH/remote) |

## Performance Comparison

### Resolution Impact (YOLOv8n, detection_interval=1)

| Resolution | FPS | Quality | Use Case |
|------------|-----|---------|----------|
| 320x240 | 13 FPS | Low | Speed priority |
| 640x480 | 13 FPS | Medium | ⭐ Balanced (recommended) |
| 1280x720 | 10 FPS | High | Quality priority |
| 1920x1080 | ~7 FPS | Highest | Max quality |

### Detection Interval Impact (640x480, YOLOv8n)

| Interval | FPS | Tracking Smoothness |
|----------|-----|---------------------|
| 1 (every frame) | 13 FPS | Perfect |
| 2 (every 2nd) | 20 FPS | Excellent |
| 3 (every 3rd) | **30 FPS** | Good ⭐ |
| 5 (every 5th) | 35+ FPS | Fair |

### Model Comparison (640x480, single inference)

| Model | Inference Time | Size | Notes |
|-------|----------------|------|-------|
| YOLOv8n | 40ms | 6.2 MB | ⭐ Fastest |
| YOLO11n | 43ms | 5.4 MB | Balanced |
| YOLO26n | 45ms | 5.3 MB | Latest, edge-optimized |

## Examples

### Maximum Speed (30 FPS)

```bash
python3 vlog_recorder.py \
  --camera 1 \
  --model yolov8n.pt \
  --resolution 640x480 \
  --detection-interval 3 \
  --glitch mosaic_heavy \
  --record
```

### Maximum Quality

```bash
python3 vlog_recorder.py \
  --camera 1 \
  --model yolo11n.pt \
  --resolution 1920x1080 \
  --detection-interval 1 \
  --glitch mosaic_shuffle \
  --record
```

### Lightweight Setup (Low Power)

```bash
python3 vlog_recorder.py \
  --camera 1 \
  --model yolov8n.pt \
  --resolution 320x240 \
  --detection-interval 5 \
  --glitch mosaic_heavy \
  --record
```

### Cyberpunk Style

```bash
python3 vlog_recorder.py \
  --camera 1 \
  --glitch rgb_shift \
  --record
```

## Keyboard Controls

### GUI Application
- Use on-screen buttons for all controls
- Window close button prompts to save recording if in progress

### Command-Line Mode
- `q` - Quit
- `s` - Save screenshot

## Face Detection Methods

### YOLO-Face (Default, Recommended)
- Accurate at all angles
- Works with side/downward-facing faces
- ~50ms inference time
- Best for vlogging

### MediaPipe
- Very accurate for frontal faces
- Struggles with side angles
- ~40ms inference time
- Good for static shots

### Hybrid
- Tries YOLO first, falls back to MediaPipe
- Best accuracy but slowest
- Use with detection-interval ≥ 3

## Project Structure

```
.
├── vlog_recorder_gui.py      # GUI application (recommended)
├── vlog_recorder.py          # Command-line version
├── yolov8-face.pt            # YOLO face detection model
├── yolo11n.pt                # YOLO object detection model (auto-downloaded)
├── yolov8n.pt                # Alternative YOLO model (auto-downloaded)
├── CCTV_Vlog_Recorder.desktop # Desktop launcher
├── README.md                 # This file
└── vlog_recorder_README.md   # Detailed Korean documentation
```

## Troubleshooting

### Camera Not Found

```bash
# List available cameras
ls /dev/video*

# Test camera
python3 test_camera.py
```

### Low FPS

1. **Increase detection interval**: `--detection-interval 3`
2. **Lower resolution**: `--resolution 320x240`
3. **Use faster model**: `--model yolov8n.pt`
4. **Simpler effect**: `--glitch mosaic_heavy`

### Out of Memory

```bash
# Use smaller resolution
python3 vlog_recorder.py --resolution 320x240 --record
```

### No Face Detection

- Ensure face is visible and well-lit
- Try different detector: `--face-detector hybrid`
- Lower confidence: `--confidence 0.2`

## Performance Tips

1. **30 FPS Goal**: Use `--detection-interval 3` with 640x480
2. **Best Quality**: Use 1280x720 with `--detection-interval 1`
3. **Lowest Latency**: Use 320x240 with `--detection-interval 5`
4. **Battery Saving**: Use 320x240 with YOLOv8n

## Credits

- **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **YOLO-Face**: [arnabdhar/YOLOv8-Face-Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)
- **MediaPipe**: [Google MediaPipe](https://mediapipe.dev/)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Changelog

### v1.1.0 (2026-03-15)
- **NEW**: GUI application with camera app-style interface
- **NEW**: One-click start/stop recording
- **NEW**: Real-time preview with settings panel
- **NEW**: Desktop launcher for easy access
- **IMPROVED**: Live FPS and status monitoring
- **IMPROVED**: Auto-generated filenames
- **IMPROVED**: Screenshot functionality in GUI

### v1.0.0 (2026-03-15)
- Initial release
- YOLO object detection
- YOLO-Face & MediaPipe face detection
- 8 glitch effects
- Frame-skip optimization (30 FPS)
- Headless mode support
- Configurable resolution

---

**Optimized for NVIDIA Jetson Orin Nano Super** 🚀
