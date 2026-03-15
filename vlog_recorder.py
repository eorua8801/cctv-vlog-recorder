#!/usr/bin/env python3
"""
CCTV-style Vlog Recorder with Object Detection and Face Glitch Effect
Optimized for NVIDIA Jetson Orin Nano Super
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time
import argparse
import mediapipe as mp


class GlitchEffect:
    """Glitch blur effect generator for face anonymization"""

    @staticmethod
    def rgb_shift(image, offset=5):
        """RGB channel shift effect"""
        h, w = image.shape[:2]
        result = np.zeros_like(image)

        # Shift each channel differently
        result[:, :, 0] = np.roll(image[:, :, 0], offset, axis=1)  # Blue
        result[:, :, 1] = image[:, :, 1]  # Green (no shift)
        result[:, :, 2] = np.roll(image[:, :, 2], -offset, axis=1)  # Red

        return result

    @staticmethod
    def pixelate(image, pixel_size=10):
        """Pixelation effect"""
        h, w = image.shape[:2]

        # Resize down
        temp = cv2.resize(image, (w // pixel_size, h // pixel_size),
                         interpolation=cv2.INTER_LINEAR)
        # Resize back up
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
    def pixelate_shuffle(image, block_size=20):
        """Pixelate with shuffled blocks - heavy mosaic effect"""
        h, w = image.shape[:2]

        # Adjust block_size if image is too small
        block_size = min(block_size, w // 2, h // 2)
        if block_size < 2:
            return image

        result = image.copy()

        # Calculate number of blocks
        blocks_y = h // block_size
        blocks_x = w // block_size

        if blocks_y == 0 or blocks_x == 0:
            return image

        # Create list of all blocks
        blocks = []
        block_positions = []

        for i in range(blocks_y):
            for j in range(blocks_x):
                y1 = i * block_size
                x1 = j * block_size
                y2 = y1 + block_size
                x2 = x1 + block_size

                # Extract block and average color
                block = image[y1:y2, x1:x2]
                avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)

                blocks.append(avg_color)
                block_positions.append((y1, y2, x1, x2))

        # Shuffle blocks
        np.random.shuffle(blocks)

        # Place shuffled blocks back
        for idx, (y1, y2, x1, x2) in enumerate(block_positions):
            result[y1:y2, x1:x2] = blocks[idx]

        return result

    @staticmethod
    def mosaic_heavy(image, block_size=25):
        """Heavy mosaic with large blocks"""
        h, w = image.shape[:2]

        # Adjust block_size if image is too small
        block_size = min(block_size, w // 2, h // 2)
        if block_size < 2:
            return image

        # Pixelate to large blocks
        temp = cv2.resize(image, (max(1, w // block_size), max(1, h // block_size)),
                         interpolation=cv2.INTER_LINEAR)
        result = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        return result

    @staticmethod
    def mosaic_shuffle(image, block_size=20):
        """Combination of heavy pixelation and shuffle"""
        # First apply heavy mosaic
        result = GlitchEffect.mosaic_heavy(image, block_size=block_size)
        # Then shuffle the blocks
        result = GlitchEffect.pixelate_shuffle(result, block_size=block_size)

        return result

    @staticmethod
    def apply_glitch(image, effect_type='combined'):
        """Apply glitch effect to image region"""
        if effect_type == 'rgb_shift':
            return GlitchEffect.rgb_shift(image)
        elif effect_type == 'pixelate':
            return GlitchEffect.pixelate(image)
        elif effect_type == 'noise':
            return GlitchEffect.add_noise(image)
        elif effect_type == 'scanlines':
            return GlitchEffect.scanlines(image)
        elif effect_type == 'mosaic_heavy':
            return GlitchEffect.mosaic_heavy(image, block_size=25)
        elif effect_type == 'pixelate_shuffle':
            return GlitchEffect.pixelate_shuffle(image, block_size=20)
        elif effect_type == 'mosaic_shuffle':
            return GlitchEffect.mosaic_shuffle(image, block_size=20)
        elif effect_type == 'combined':
            # Combine multiple effects
            result = GlitchEffect.rgb_shift(image, offset=3)
            result = GlitchEffect.pixelate(result, pixel_size=8)
            result = GlitchEffect.add_noise(result, intensity=30)
            result = GlitchEffect.scanlines(result, line_height=2)
            return result
        else:
            return image


class VlogRecorder:
    """Main vlog recorder class"""

    def __init__(self, camera_id=0, model_name='yolo11n.pt',
                 confidence_threshold=0.3, glitch_effect='combined',
                 output_path=None, headless=False, face_detector_type='yolo',
                 resolution='640x480', detection_interval=1):
        """
        Initialize the vlog recorder

        Args:
            camera_id: Camera device ID (default: 0)
            model_name: YOLO model name (default: 'yolo11n.pt')
            confidence_threshold: Detection confidence threshold
            glitch_effect: Type of glitch effect for faces
            output_path: Output video file path (None = don't record)
            headless: Run without GUI display (default: False)
            face_detector_type: Face detection method ('yolo', 'mediapipe', 'hybrid')
            resolution: Camera resolution (WIDTHxHEIGHT)
            detection_interval: Run detection every N frames (1=every frame, 3=every 3rd frame)
        """
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.glitch_effect = glitch_effect
        self.output_path = output_path
        self.headless = headless
        self.face_detector_type = face_detector_type
        self.detection_interval = detection_interval

        # Cache for detection results
        self.frame_count = 0
        self.cached_face_boxes = []
        self.cached_object_detections = []

        # Initialize YOLO model for object detection
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)

        # Initialize face detection
        self.yolo_face_model = None
        self.mediapipe_face_detector = None

        if face_detector_type in ['yolo', 'hybrid']:
            print("Loading YOLOv8-Face model...")
            self.yolo_face_model = YOLO('yolov8-face.pt')

        if face_detector_type in ['mediapipe', 'hybrid']:
            print("Loading MediaPipe Face Detection...")
            self.mp_face_detection = mp.solutions.face_detection
            self.mediapipe_face_detector = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )

        # Initialize camera
        print(f"Opening camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)

        # Parse resolution
        width, height = map(int, resolution.split('x'))

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Get actual camera properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps")

        # Video writer
        self.video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.fps,
                (self.frame_width, self.frame_height)
            )
            print(f"Recording to: {output_path}")

        # FPS calculation
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def detect_faces_yolo(self, frame):
        """
        Detect faces using YOLO face detection model

        Args:
            frame: Input image frame

        Returns:
            List of face bounding boxes as [x1, y1, x2, y2]
        """
        face_boxes = []
        if self.yolo_face_model:
            results = self.yolo_face_model(frame, conf=0.3, verbose=False)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    face_boxes.append([x1, y1, x2, y2])

        return face_boxes

    def detect_faces_mediapipe(self, frame):
        """
        Detect faces using MediaPipe Face Detection

        Args:
            frame: Input image frame

        Returns:
            List of face bounding boxes as [x1, y1, x2, y2]
        """
        face_boxes = []
        if self.mediapipe_face_detector:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            results = self.mediapipe_face_detector.process(rgb_frame)

            if results.detections:
                h, w = frame.shape[:2]
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box

                    # Convert relative coordinates to absolute
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)

                    # Ensure coordinates are within frame bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    face_boxes.append([x1, y1, x2, y2])

        return face_boxes

    def detect_faces(self, frame):
        """
        Detect faces using selected method

        Args:
            frame: Input image frame

        Returns:
            List of face bounding boxes as [x1, y1, x2, y2]
        """
        if self.face_detector_type == 'yolo':
            return self.detect_faces_yolo(frame)
        elif self.face_detector_type == 'mediapipe':
            return self.detect_faces_mediapipe(frame)
        elif self.face_detector_type == 'hybrid':
            # Try YOLO first, fallback to MediaPipe if no faces found
            faces = self.detect_faces_yolo(frame)
            if not faces:
                faces = self.detect_faces_mediapipe(frame)
            return faces
        else:
            return []

    def draw_bounding_box(self, frame, box, label, confidence, color=(0, 255, 0)):
        """Draw bounding box with label"""
        x1, y1, x2, y2 = map(int, box)

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        text = f"{label} {confidence:.2f}"

        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            frame,
            text,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )

    def add_overlay(self, frame, fps):
        """Add timestamp and FPS overlay"""
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Overlay properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (0, 255, 0)  # Green

        # Add timestamp (top-left)
        cv2.putText(
            frame,
            timestamp,
            (10, 30),
            font,
            font_scale,
            color,
            thickness
        )

        # Add FPS (top-right)
        fps_text = f"FPS: {fps:.1f}"
        (text_width, text_height), _ = cv2.getTextSize(
            fps_text, font, font_scale, thickness
        )
        cv2.putText(
            frame,
            fps_text,
            (self.frame_width - text_width - 10, 30),
            font,
            font_scale,
            color,
            thickness
        )

    def estimate_face_from_person(self, person_box):
        """Estimate face region from person box (fallback method)"""
        x1, y1, x2, y2 = map(int, person_box)
        box_width = x2 - x1
        box_height = y2 - y1

        # Face is typically in upper 30% of person box
        face_height = int(box_height * 0.30)
        face_width = int(box_width * 0.7)

        # Calculate face coordinates (centered horizontally)
        face_x1 = x1 + (box_width - face_width) // 2
        face_y1 = y1
        face_x2 = face_x1 + face_width
        face_y2 = y1 + face_height

        # Ensure within frame bounds
        face_x1 = max(0, face_x1)
        face_y1 = max(0, face_y1)
        face_x2 = min(self.frame_width, face_x2)
        face_y2 = min(self.frame_height, face_y2)

        return [face_x1, face_y1, face_x2, face_y2]

    def box_contains_point(self, box, point):
        """Check if a box contains a point"""
        x1, y1, x2, y2 = box
        px, py = point
        return x1 <= px <= x2 and y1 <= py <= y2

    def process_frame(self, frame):
        """Process a single frame with object detection and face glitch"""
        self.frame_count += 1

        # Run detection every N frames, otherwise use cached results
        should_detect = (self.frame_count % self.detection_interval) == 0

        if should_detect:
            # Detect faces
            mediapipe_faces = self.detect_faces(frame)

            # Run YOLO detection for objects
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)

            # Collect all faces to blur (MediaPipe + fallback from person boxes)
            all_face_boxes = list(mediapipe_faces)

            # Process YOLO detections
            person_boxes = []
            all_detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.model.names[cls]

                    all_detections.append((xyxy, conf, cls, label))

                    if label.lower() == 'person':
                        person_boxes.append((xyxy, conf, label))

            # For each person, check if we have a face detection
            # If not, use fallback estimation
            for person_box, conf, label in person_boxes:
                px1, py1, px2, py2 = person_box

                # Check if any face is inside this person box
                has_face = False

                for face_box in mediapipe_faces:
                    fx1, fy1, fx2, fy2 = face_box
                    face_center = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)

                    # Check if face center is inside person box
                    if self.box_contains_point(person_box, face_center):
                        has_face = True
                        break

                # If no face found for this person, use fallback
                if not has_face:
                    estimated_face = self.estimate_face_from_person(person_box)
                    all_face_boxes.append(estimated_face)

            # Cache the results for next frames
            self.cached_face_boxes = all_face_boxes
            self.cached_object_detections = all_detections
        else:
            # Use cached results
            all_face_boxes = self.cached_face_boxes
            all_detections = self.cached_object_detections

        # Apply glitch effect to all collected faces
        for face_box in all_face_boxes:
            fx1, fy1, fx2, fy2 = map(int, face_box)

            # Extract face region
            if fx2 > fx1 and fy2 > fy1:
                face_region = frame[fy1:fy2, fx1:fx2].copy()

                # Apply glitch effect
                glitched_face = GlitchEffect.apply_glitch(
                    face_region, self.glitch_effect
                )

                # Replace face region with glitched version
                frame[fy1:fy2, fx1:fx2] = glitched_face

        # Draw bounding boxes for all detected objects (using cached detections)
        for xyxy, conf, cls, label in all_detections:
            # Draw bounding box
            if label.lower() == 'person':
                color = (0, 255, 255)  # Yellow for person
            else:
                color = (0, 255, 0)  # Green for others

            self.draw_bounding_box(frame, xyxy, label, conf, color=color)

        return frame

    def run(self, duration=None):
        """Main loop for processing video"""
        print("\nStarting vlog recorder...")
        if self.headless:
            print("Running in headless mode (no GUI)")
            if duration:
                print(f"Recording for {duration} seconds...")
            else:
                print("Press Ctrl+C to stop")
        else:
            print("Press 'q' to quit, 's' to save screenshot")

        start_time = time.time()
        frame_count = 0

        try:
            while True:
                # Check duration limit (for headless mode)
                if duration and (time.time() - start_time) >= duration:
                    print(f"\nRecording completed ({duration}s)")
                    break

                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Calculate FPS
                self.fps_counter += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.current_fps = self.fps_counter / elapsed

                    # Print status in headless mode
                    if self.headless:
                        frame_count += self.fps_counter
                        print(f"Processing: {frame_count} frames, {self.current_fps:.1f} FPS")

                    self.fps_counter = 0
                    self.fps_start_time = time.time()

                # Add overlay
                self.add_overlay(processed_frame, self.current_fps)

                # Write to video file
                if self.video_writer:
                    self.video_writer.write(processed_frame)

                # Display frame (only if not headless)
                if not self.headless:
                    cv2.imshow('CCTV Vlog Recorder', processed_frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(screenshot_path, processed_frame)
                        print(f"Screenshot saved: {screenshot_path}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources"""
        print("\nCleaning up...")
        if self.video_writer:
            self.video_writer.release()
        self.cap.release()
        if self.mediapipe_face_detector:
            self.mediapipe_face_detector.close()
        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except:
                pass  # Ignore errors if GUI not available
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='CCTV-style Vlog Recorder with Object Detection and Face Glitch'
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--model', type=str, default='yolo11n.pt',
        help='YOLO model name (default: yolo11n.pt)'
    )
    parser.add_argument(
        '--confidence', type=float, default=0.3,
        help='Detection confidence threshold (default: 0.3)'
    )
    parser.add_argument(
        '--glitch', type=str, default='combined',
        choices=['rgb_shift', 'pixelate', 'noise', 'scanlines', 'combined',
                 'mosaic_heavy', 'pixelate_shuffle', 'mosaic_shuffle'],
        help='Glitch effect type (default: combined)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output video file path (default: None - no recording)'
    )
    parser.add_argument(
        '--record', action='store_true',
        help='Enable recording with auto-generated filename'
    )
    parser.add_argument(
        '--headless', action='store_true',
        help='Run without GUI display (for SSH/headless systems)'
    )
    parser.add_argument(
        '--duration', type=int, default=None,
        help='Recording duration in seconds (headless mode only)'
    )
    parser.add_argument(
        '--face-detector', type=str, default='yolo',
        choices=['yolo', 'mediapipe', 'hybrid'],
        help='Face detection method (default: yolo)'
    )
    parser.add_argument(
        '--resolution', type=str, default='640x480',
        help='Camera resolution WIDTHxHEIGHT (default: 640x480, options: 320x240, 640x480, 1280x720, 1920x1080)'
    )
    parser.add_argument(
        '--detection-interval', type=int, default=1,
        help='Run detection every N frames for speed boost (1=every frame, 3=every 3rd frame, default: 1)'
    )

    args = parser.parse_args()

    # Auto-generate output filename if --record is used
    output_path = args.output
    if args.record and not output_path:
        output_path = f"vlog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

    # Headless mode requires recording
    if args.headless and not output_path:
        print("Warning: Headless mode enabled but no output specified.")
        print("Auto-enabling recording...")
        output_path = f"vlog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

    # Create and run recorder
    recorder = VlogRecorder(
        camera_id=args.camera,
        model_name=args.model,
        confidence_threshold=args.confidence,
        glitch_effect=args.glitch,
        output_path=output_path,
        headless=args.headless,
        face_detector_type=args.face_detector,
        resolution=args.resolution,
        detection_interval=args.detection_interval
    )

    recorder.run(duration=args.duration)


if __name__ == '__main__':
    main()
