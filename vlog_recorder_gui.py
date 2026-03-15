#!/usr/bin/env python3
"""
CCTV-style Vlog Recorder GUI - Camera App Style
Real-time monitoring with Start/Stop recording buttons
Optimized for NVIDIA Jetson Orin Nano Super
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk


class GlitchEffect:
    """Glitch blur effect generator for face anonymization"""

    @staticmethod
    def rgb_shift(image, offset=5):
        """RGB channel shift effect"""
        result = np.zeros_like(image)
        result[:, :, 0] = np.roll(image[:, :, 0], offset, axis=1)
        result[:, :, 1] = image[:, :, 1]
        result[:, :, 2] = np.roll(image[:, :, 2], -offset, axis=1)
        return result

    @staticmethod
    def pixelate(image, pixel_size=10):
        """Pixelation effect"""
        h, w = image.shape[:2]
        temp = cv2.resize(image, (w // pixel_size, h // pixel_size),
                         interpolation=cv2.INTER_LINEAR)
        result = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        return result

    @staticmethod
    def add_noise(image, intensity=50):
        """Add digital noise"""
        noise = np.random.randint(-intensity, intensity, image.shape, dtype=np.int16)
        result = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def scanlines(image, line_height=2):
        """Add scanline effect"""
        result = image.copy()
        h = image.shape[0]
        for i in range(0, h, line_height * 2):
            end_idx = min(i + line_height, h)
            result[i:end_idx] = result[i:end_idx] * 0.5
        return result.astype(np.uint8)

    @staticmethod
    def mosaic_heavy(image, block_size=30):
        """Heavy mosaic effect"""
        h, w = image.shape[:2]
        block_size = min(block_size, w // 2, h // 2)
        if block_size < 2:
            return image

        result = image.copy()
        blocks_y = h // block_size
        blocks_x = w // block_size

        for i in range(blocks_y):
            for j in range(blocks_x):
                y1, x1 = i * block_size, j * block_size
                y2, x2 = y1 + block_size, x1 + block_size
                avg_color = np.mean(image[y1:y2, x1:x2], axis=(0, 1)).astype(np.uint8)
                result[y1:y2, x1:x2] = avg_color

        return result

    @staticmethod
    def pixelate_shuffle(image, block_size=20):
        """Pixelate with shuffled blocks"""
        h, w = image.shape[:2]
        block_size = min(block_size, w // 2, h // 2)
        if block_size < 2:
            return image

        result = image.copy()
        blocks_y = h // block_size
        blocks_x = w // block_size

        blocks = []
        block_positions = []

        for i in range(blocks_y):
            for j in range(blocks_x):
                y1, x1 = i * block_size, j * block_size
                y2, x2 = y1 + block_size, x1 + block_size
                avg_color = np.mean(image[y1:y2, x1:x2], axis=(0, 1)).astype(np.uint8)
                blocks.append(avg_color)
                block_positions.append((y1, y2, x1, x2))

        np.random.shuffle(blocks)

        for (y1, y2, x1, x2), color in zip(block_positions, blocks):
            result[y1:y2, x1:x2] = color

        return result

    @staticmethod
    def mosaic_shuffle(image, block_size=20):
        """Heavy mosaic + shuffle combination"""
        result = GlitchEffect.mosaic_heavy(image, block_size=block_size)
        result = GlitchEffect.pixelate_shuffle(result, block_size=block_size)
        return result

    @staticmethod
    def combined(image):
        """Combined glitch effect"""
        result = GlitchEffect.rgb_shift(image, offset=8)
        result = GlitchEffect.pixelate(result, pixel_size=12)
        result = GlitchEffect.add_noise(result, intensity=30)
        result = GlitchEffect.scanlines(result, line_height=3)
        return result


class VlogRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CCTV Vlog Recorder")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2b2b2b')

        # Recording state
        self.is_recording = False
        self.is_running = True
        self.video_writer = None
        self.output_path = None

        # Camera settings
        self.camera_id = tk.IntVar(value=0)
        self.resolution = tk.StringVar(value="640x480")
        self.model_name = tk.StringVar(value="yolov8n.pt")
        self.glitch_type = tk.StringVar(value="mosaic_heavy")
        self.detection_interval = tk.IntVar(value=3)
        self.confidence = tk.DoubleVar(value=0.3)

        # Initialize models (lazy loading)
        self.object_model = None
        self.face_model = None
        self.cap = None
        self.models_loaded = False

        # Frame cache for detection
        self.frame_count = 0
        self.cached_face_boxes = []
        self.cached_object_detections = []

        # FPS tracking
        self.fps = 0
        self.frame_times = []

        # Setup UI
        self.setup_ui()

        # Start camera (models loaded later)
        self.init_camera()

        # Schedule frame update after a short delay
        self.root.after(100, self.update_frame)

    def setup_ui(self):
        """Setup the GUI layout"""
        # Top control panel
        control_frame = tk.Frame(self.root, bg='#1e1e1e', padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Recording controls
        rec_frame = tk.Frame(control_frame, bg='#1e1e1e')
        rec_frame.pack(side=tk.LEFT, padx=5)

        self.record_btn = tk.Button(
            rec_frame,
            text="⏺ START RECORDING",
            command=self.toggle_recording,
            bg='#d32f2f',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)

        self.screenshot_btn = tk.Button(
            rec_frame,
            text="📷 Screenshot",
            command=self.take_screenshot,
            bg='#455a64',
            fg='white',
            font=('Arial', 10),
            padx=15,
            pady=10,
            cursor='hand2'
        )
        self.screenshot_btn.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="⚫ Ready",
            bg='#1e1e1e',
            fg='#aaaaaa',
            font=('Arial', 12)
        )
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Settings panel (right side)
        settings_btn = tk.Button(
            control_frame,
            text="⚙ Settings",
            command=self.show_settings,
            bg='#37474f',
            fg='white',
            font=('Arial', 10),
            padx=15,
            pady=10,
            cursor='hand2'
        )
        settings_btn.pack(side=tk.RIGHT, padx=5)

        # Video display
        self.video_label = tk.Label(self.root, bg='#000000')
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Bottom info panel
        info_frame = tk.Frame(self.root, bg='#1e1e1e', pady=5)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.fps_label = tk.Label(
            info_frame,
            text="FPS: 0",
            bg='#1e1e1e',
            fg='#4caf50',
            font=('Courier', 10)
        )
        self.fps_label.pack(side=tk.LEFT, padx=10)

        self.resolution_label = tk.Label(
            info_frame,
            text="Resolution: 640x480",
            bg='#1e1e1e',
            fg='#aaaaaa',
            font=('Courier', 10)
        )
        self.resolution_label.pack(side=tk.LEFT, padx=10)

        self.model_label = tk.Label(
            info_frame,
            text="Model: YOLOv8n",
            bg='#1e1e1e',
            fg='#aaaaaa',
            font=('Courier', 10)
        )
        self.model_label.pack(side=tk.LEFT, padx=10)

    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x500")
        settings_window.configure(bg='#2b2b2b')
        settings_window.transient(self.root)

        # Camera settings
        tk.Label(
            settings_window,
            text="Camera Settings",
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 12, 'bold')
        ).pack(pady=10)

        # Camera ID
        cam_frame = tk.Frame(settings_window, bg='#2b2b2b')
        cam_frame.pack(pady=5, padx=20, fill=tk.X)
        tk.Label(cam_frame, text="Camera ID:", bg='#2b2b2b', fg='white').pack(side=tk.LEFT)
        tk.Spinbox(
            cam_frame,
            from_=0,
            to=10,
            textvariable=self.camera_id,
            width=10
        ).pack(side=tk.RIGHT)

        # Resolution
        res_frame = tk.Frame(settings_window, bg='#2b2b2b')
        res_frame.pack(pady=5, padx=20, fill=tk.X)
        tk.Label(res_frame, text="Resolution:", bg='#2b2b2b', fg='white').pack(side=tk.LEFT)
        ttk.Combobox(
            res_frame,
            textvariable=self.resolution,
            values=["320x240", "640x480", "1280x720", "1920x1080"],
            width=15
        ).pack(side=tk.RIGHT)

        # Model
        tk.Label(
            settings_window,
            text="Detection Settings",
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 12, 'bold')
        ).pack(pady=10)

        model_frame = tk.Frame(settings_window, bg='#2b2b2b')
        model_frame.pack(pady=5, padx=20, fill=tk.X)
        tk.Label(model_frame, text="YOLO Model:", bg='#2b2b2b', fg='white').pack(side=tk.LEFT)
        ttk.Combobox(
            model_frame,
            textvariable=self.model_name,
            values=["yolov8n.pt", "yolo11n.pt", "yolo26n.pt"],
            width=15
        ).pack(side=tk.RIGHT)

        # Glitch effect
        glitch_frame = tk.Frame(settings_window, bg='#2b2b2b')
        glitch_frame.pack(pady=5, padx=20, fill=tk.X)
        tk.Label(glitch_frame, text="Glitch Effect:", bg='#2b2b2b', fg='white').pack(side=tk.LEFT)
        ttk.Combobox(
            glitch_frame,
            textvariable=self.glitch_type,
            values=[
                "rgb_shift",
                "pixelate",
                "noise",
                "scanlines",
                "mosaic_heavy",
                "pixelate_shuffle",
                "mosaic_shuffle",
                "combined"
            ],
            width=15
        ).pack(side=tk.RIGHT)

        # Detection interval
        interval_frame = tk.Frame(settings_window, bg='#2b2b2b')
        interval_frame.pack(pady=5, padx=20, fill=tk.X)
        tk.Label(interval_frame, text="Detection Interval:", bg='#2b2b2b', fg='white').pack(side=tk.LEFT)
        tk.Spinbox(
            interval_frame,
            from_=1,
            to=10,
            textvariable=self.detection_interval,
            width=10
        ).pack(side=tk.RIGHT)

        # Confidence
        conf_frame = tk.Frame(settings_window, bg='#2b2b2b')
        conf_frame.pack(pady=5, padx=20, fill=tk.X)
        tk.Label(conf_frame, text="Confidence:", bg='#2b2b2b', fg='white').pack(side=tk.LEFT)
        tk.Scale(
            conf_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            variable=self.confidence,
            orient=tk.HORIZONTAL,
            bg='#2b2b2b',
            fg='white'
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Apply button
        tk.Button(
            settings_window,
            text="Apply & Restart Camera",
            command=lambda: self.apply_settings(settings_window),
            bg='#4caf50',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=10,
            cursor='hand2'
        ).pack(pady=20)

    def apply_settings(self, window):
        """Apply settings and restart camera"""
        window.destroy()

        # Release old resources
        if self.cap:
            self.cap.release()

        # Clear models for reload
        self.models_loaded = False
        self.object_model = None
        self.face_model = None

        # Reinitialize
        self.init_camera()
        messagebox.showinfo("Settings", "Settings applied! Camera restarted.")

    def init_camera(self):
        """Initialize camera only (models loaded on first frame)"""
        try:
            # Open camera first
            print(f"Opening camera {self.camera_id.get()}...")
            self.cap = cv2.VideoCapture(self.camera_id.get())

            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera {self.camera_id.get()}")

            # Set resolution
            width, height = map(int, self.resolution.get().split('x'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Update UI
            self.resolution_label.config(text=f"Resolution: {self.resolution.get()}")
            self.model_label.config(text=f"Model: {self.model_name.get()}")

            print("Camera initialized. Models will load on first frame.")
            self.models_loaded = False

        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize camera:\n{str(e)}")
            self.is_running = False

    def load_models(self):
        """Load detection models (called on first frame)"""
        if self.models_loaded:
            return

        try:
            print("Loading models (this may take a moment)...")
            self.status_label.config(text="⏳ Loading models...", fg='#ffa726')
            self.root.update()

            # Load YOLO object detection model
            print(f"Loading YOLO model: {self.model_name.get()}...")
            self.object_model = YOLO(self.model_name.get())

            # Load YOLO-Face model
            print("Loading YOLO-Face model...")
            self.face_model = YOLO('yolov8-face.pt')

            self.models_loaded = True
            self.status_label.config(text="✓ Models loaded", fg='#4caf50')
            print("Models loaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
            self.is_running = False

    def detect_faces_yolo(self, frame):
        """Detect faces using YOLO-Face"""
        results = self.face_model(frame, conf=self.confidence.get(), verbose=False)
        face_boxes = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_boxes.append((x1, y1, x2, y2))

        return face_boxes

    def apply_glitch_effect(self, face_region):
        """Apply selected glitch effect"""
        effect_map = {
            "rgb_shift": lambda img: GlitchEffect.rgb_shift(img, offset=8),
            "pixelate": lambda img: GlitchEffect.pixelate(img, pixel_size=15),
            "noise": lambda img: GlitchEffect.add_noise(img, intensity=60),
            "scanlines": lambda img: GlitchEffect.scanlines(img, line_height=3),
            "mosaic_heavy": lambda img: GlitchEffect.mosaic_heavy(img, block_size=30),
            "pixelate_shuffle": lambda img: GlitchEffect.pixelate_shuffle(img, block_size=20),
            "mosaic_shuffle": lambda img: GlitchEffect.mosaic_shuffle(img, block_size=20),
            "combined": GlitchEffect.combined
        }

        effect_func = effect_map.get(self.glitch_type.get(), GlitchEffect.mosaic_heavy)
        return effect_func(face_region)

    def process_frame(self, frame):
        """Process frame with detection and effects"""
        # Load models on first frame
        if not self.models_loaded:
            self.load_models()
            if not self.models_loaded:  # If loading failed
                return frame

        self.frame_count += 1
        should_detect = (self.frame_count % self.detection_interval.get()) == 0

        if should_detect:
            try:
                # Detect faces
                self.cached_face_boxes = self.detect_faces_yolo(frame)

                # Detect objects
                results = self.object_model(frame, conf=self.confidence.get(), verbose=False)
                self.cached_object_detections = []

                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            class_name = self.object_model.names[cls]

                            # Skip person class (already detected by face detector)
                            if class_name.lower() == 'person':
                                continue

                            self.cached_object_detections.append({
                                'box': (x1, y1, x2, y2),
                                'conf': conf,
                                'class': class_name
                            })
            except Exception as e:
                print(f"Detection error: {e}")

        # Apply face glitch
        for (x1, y1, x2, y2) in self.cached_face_boxes:
            face_region = frame[y1:y2, x1:x2]
            if face_region.size > 0:
                glitched = self.apply_glitch_effect(face_region)
                frame[y1:y2, x1:x2] = glitched

        # Draw object bounding boxes
        for det in self.cached_object_detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']} {det['conf']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def add_overlay(self, frame):
        """Add CCTV-style overlay"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add timestamp
        cv2.putText(frame, timestamp, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add recording indicator
        if self.is_recording:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 80, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    def update_frame(self):
        """Update video frame in GUI"""
        if not self.is_running or not self.cap or not self.cap.isOpened():
            return

        try:
            start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                self.root.after(10, self.update_frame)
                return

            # Process frame
            frame = self.process_frame(frame)
            frame = self.add_overlay(frame)

            # Write to video if recording
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)

            # Convert to PhotoImage for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Resize to fit display (with safety check)
            h, w = frame.shape[:2]
            if w > 0 and h > 0:
                display_width = 960
                display_height = int(h * (display_width / w))
                img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)

                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            # Calculate FPS
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")

        except Exception as e:
            print(f"Frame update error: {e}")

        # Schedule next update
        self.root.after(10, self.update_frame)

    def toggle_recording(self):
        """Start/Stop recording"""
        if not self.is_recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = f"vlog_{timestamp}.mp4"

            width, height = map(int, self.resolution.get().split('x'))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                30.0,
                (width, height)
            )

            self.is_recording = True
            self.record_btn.config(
                text="⏹ STOP RECORDING",
                bg='#1976d2'
            )
            self.status_label.config(
                text=f"🔴 Recording: {self.output_path}",
                fg='#f44336'
            )
        else:
            # Stop recording
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            self.record_btn.config(
                text="⏺ START RECORDING",
                bg='#d32f2f'
            )
            self.status_label.config(
                text=f"✓ Saved: {self.output_path}",
                fg='#4caf50'
            )

            messagebox.showinfo(
                "Recording Saved",
                f"Video saved to:\n{self.output_path}"
            )

    def take_screenshot(self):
        """Take a screenshot"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                messagebox.showinfo("Screenshot", f"Screenshot saved:\n{filename}")

    def on_closing(self):
        """Handle window close"""
        if self.is_recording:
            if messagebox.askokcancel("Recording in Progress", "Stop recording and quit?"):
                self.is_recording = False
                if self.video_writer:
                    self.video_writer.release()
            else:
                return

        self.is_running = False
        if self.cap:
            self.cap.release()

        # Clean up models
        self.object_model = None
        self.face_model = None

        self.root.destroy()


def main():
    root = tk.Tk()
    app = VlogRecorderGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
