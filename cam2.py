import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QMessageBox, QGroupBox, QProgressBar, QFileDialog, QDialog)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QMutex, QMutexLocker, QMetaObject, Slot
from PySide6.QtGui import QImage, QPixmap, QFont
import time
from datetime import datetime
import platform
import threading
import json
import os
from pathlib import Path
import concurrent.futures

# Suppress OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ELP Camera Control")
        self.setWindowFlags((self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint) & ~Qt.WindowContextHelpButtonHint)
        self.setModal(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        logo_label = QLabel()
        logo_path = Path(__file__).resolve().parent / 'logo.png'
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            logo_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        self.message_label = QLabel("Starting application...")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.message_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.setFixedSize(420, 520)

    def update_progress(self, value, message=None):
        self.progress_bar.setValue(value)
        if message:
            self.message_label.setText(message)
        QApplication.processEvents()


class UnifiedCameraThread(QThread):
    frameReady = Signal(np.ndarray)
    connectionLost = Signal()
    recordingStarted = Signal()
    recordingStopped = Signal(str)
    framesCaptured = Signal(int)
    recordingProgress = Signal(str)
    recordingError = Signal(str)
    cameraReinitialized = Signal()

    def __init__(self):
        super().__init__()
        self.cap = None
        self.is_running = False
        self.is_recording = False
        self.needs_reinit = False
        self.camera_index = 0
        self.mutex = QMutex()
        self.current_resolution = (1280, 720)
        self.current_fps = 120
        self.codec = "XVID"
        self.video_save_path = ""
        self.out = None
        self.frames_captured = 0
        self.recording_filename = ""
        self.start_time = 0

    def set_camera_mode(self, resolution, fps):
        with QMutexLocker(self.mutex):
            if resolution != self.current_resolution or fps != self.current_fps:
                self.current_resolution = resolution
                self.current_fps = fps
                self.needs_reinit = True

    def set_recording_params(self, codec, save_path):
        with QMutexLocker(self.mutex):
            self.codec = codec
            self.video_save_path = save_path

    def initialize_camera(self):
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            time.sleep(0.2)

            backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.camera_index, backend)

            if not self.cap.isOpened():
                return False

            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.current_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.current_resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.current_fps)

            ret, frame = self.cap.read()
            if not ret:
                return False

            self.cameraReinitialized.emit()
            return True
        except Exception as e:
            print(f"Camera init error: {e}")
            return False

    def run(self):
        if not self.initialize_camera():
            self.connectionLost.emit()
            return

        self.is_running = True
        consecutive_failures = 0

        while self.is_running:
            try:
                with QMutexLocker(self.mutex):
                    if self.needs_reinit and not self.is_recording:
                        self.needs_reinit = False
                        if not self.initialize_camera():
                            self.connectionLost.emit()
                            break
                        consecutive_failures = 0
                        continue

                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        consecutive_failures = 0
                        self.frameReady.emit(frame)

                        with QMutexLocker(self.mutex):
                            if self.is_recording and self.out:
                                self.out.write(frame)
                                self.frames_captured += 1
                                self.framesCaptured.emit(self.frames_captured)
                    else:
                        consecutive_failures += 1
                        if consecutive_failures > 10:
                            self.connectionLost.emit()
                            break
                    time.sleep(0.01)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Thread error: {e}")
                time.sleep(0.01)

        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()

    def start_recording(self):
        with QMutexLocker(self.mutex):
            if self.is_recording:
                return

            videos_dir = Path(self.video_save_path)
            videos_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            resolution_str = f"{self.current_resolution[0]}x{self.current_resolution[1]}"
            self.recording_filename = str(videos_dir / f"recording_{timestamp}_{resolution_str}_{self.current_fps}fps_{self.codec}.avi")

            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.out = cv2.VideoWriter(self.recording_filename, fourcc, self.current_fps, self.current_resolution)

            if not self.out.isOpened():
                self.recordingError.emit("Failed to initialize video writer")
                return

            self.frames_captured = 0
            self.start_time = time.time()
            self.is_recording = True
            self.recordingStarted.emit()
            self.recordingProgress.emit("Recording started")

    def stop_recording(self):
        with QMutexLocker(self.mutex):
            if not self.is_recording:
                return

            self.is_recording = False
            if self.out:
                self.out.release()
                self.out = None

            duration = time.time() - self.start_time

            if self.frames_captured > 0:
                self.recordingStopped.emit(f"{self.recording_filename} (duration: {duration:.1f} sec)")
            else:
                try:
                    os.remove(self.recording_filename)
                except:
                    pass
                self.recordingStopped.emit("")

    def stop(self):
        self.stop_recording()
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.wait()


class CameraApp(QMainWindow):
    def __init__(self, loading_dialog=None):
        super().__init__()
        self.loading_dialog = loading_dialog
        self.unified_thread = None
        self.is_recording = False
        self.is_switching_mode = False
        self.rotation_angle = 0  # 0, 90, 180, 270

        self.update_loading(5, "Preparing settings...")
        self.settings_dir = self.get_settings_dir()
        self.settings_file = str(self.settings_dir / 'camera_settings.json')
        self.settings = self.load_settings()

        self.update_loading(15, "Checking save path...")
        self.video_save_path = self.check_video_save_path()

        self.update_loading(30, "Loading interface...")
        self.init_ui()
        self.show()

        self.update_loading(50, "Searching for camera...")
        threading.Thread(target=self.delayed_camera_search, daemon=True).start()

    def get_settings_dir(self):
        if platform.system() == 'Windows':
            app_data = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
            path = Path(app_data) / 'ELPCameraControl'
        else:
            path = Path.home() / '.config' / 'ELPCameraControl'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def update_loading(self, value, message=None):
        if self.loading_dialog:
            QTimer.singleShot(0, lambda: self.loading_dialog.update_progress(value, message))

    @Slot()
    def finish_loading(self):
        if self.loading_dialog:
            self.loading_dialog.update_progress(100, "Done")
            QTimer.singleShot(500, self.loading_dialog.close)
            self.loading_dialog = None

    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_settings(self):
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=2)

    def check_video_save_path(self):
        video_path = self.settings.get('video_save_path', '')
        if video_path and os.path.exists(video_path) and os.path.isdir(video_path):
            return video_path

        default_path = str(Path.home() / 'Videos' / 'ELPCameraRecordings')
        Path(default_path).mkdir(parents=True, exist_ok=True)
        self.settings['video_save_path'] = default_path
        self.save_settings()
        return default_path

    def delayed_camera_search(self):
        time.sleep(0.3)
        self.camera_index = self.load_or_find_camera()

        if self.camera_index == -1:
            QMetaObject.invokeMethod(self, "show_camera_not_found", Qt.QueuedConnection)
        else:
            QMetaObject.invokeMethod(self, "init_camera_thread", Qt.QueuedConnection)

    @Slot()
    def show_camera_not_found(self):
        self.finish_loading()
        QMessageBox.critical(self, "Camera Error",
                             "ELP camera not found!\n\n"
                             "Please check that the camera is connected and try again.")
        self.close()

    def load_or_find_camera(self):
        saved_index = self.settings.get('camera_index', -1)
        if saved_index >= 0 and self.check_camera_fast(saved_index):
            return saved_index
        return self.find_elp_camera_parallel()

    def check_camera_fast(self, index):
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY)
            if not cap.isOpened():
                return False

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            cap.set(cv2.CAP_PROP_FPS, 260)

            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            return w == 640 and h == 360 and fps >= 200
        except:
            return False

    def find_elp_camera_parallel(self):
        priority_indices = [1, 2, 0, 3, 4]

        def check_index(i):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY)
                if not cap.isOpened():
                    return None

                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                cap.set(cv2.CAP_PROP_FPS, 260)

                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)

                if w == 640 and h == 360 and fps >= 200:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_FPS, 60)
                    if cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1920:
                        cap.release()
                        return i
                cap.release()
            except:
                pass
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for idx in executor.map(check_index, priority_indices):
                if idx is not None:
                    self.settings['camera_index'] = idx
                    self.save_settings()
                    return idx
        return -1

    @Slot()
    def init_camera_thread(self):
        self.unified_thread = UnifiedCameraThread()
        self.unified_thread.camera_index = self.camera_index

        resolution = self.resolution_combo.currentData()
        fps = int(self.fps_combo.currentText())

        self.unified_thread.current_resolution = resolution
        self.unified_thread.current_fps = fps
        self.unified_thread.set_recording_params(self.codec_combo.currentData(), self.video_save_path)

        self.unified_thread.frameReady.connect(self.update_frame)
        self.unified_thread.connectionLost.connect(self.on_camera_connection_lost)
        self.unified_thread.recordingStarted.connect(self.on_recording_started)
        self.unified_thread.recordingStopped.connect(self.on_recording_stopped)
        self.unified_thread.framesCaptured.connect(self.update_frames_count)
        self.unified_thread.recordingProgress.connect(self.update_recording_status)
        self.unified_thread.recordingError.connect(self.on_recording_error)
        self.unified_thread.cameraReinitialized.connect(self.on_camera_reinitialized)

        resolution_str = f"{resolution[0]}x{resolution[1]}"
        self.info_label.setText(f"Preview mode: {resolution_str} @ {fps}fps")
        self.path_label.setText(self.video_save_path)

        self.ready_indicator.setStyleSheet("color: green; font-size: 24px;")
        self.record_button.setEnabled(True)
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.unified_thread.start()
        self.finish_loading()

    def init_ui(self):
        self.setWindowTitle('ELP Camera Control')
        self.setGeometry(100, 100, 1000, 800)

        font = QFont()
        font.setPointSize(14)

        main_widget = QWidget()
        main_widget.setFont(font)
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setStyleSheet("border: 2px solid black; font-size: 16px; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Searching for camera...")
        layout.addWidget(self.video_label)

        self.recording_indicator = QLabel()
        self.recording_indicator.setAlignment(Qt.AlignCenter)
        self.recording_indicator.setStyleSheet("color: red; font-weight: bold; font-size: 18px;")
        self.recording_indicator.setVisible(False)
        layout.addWidget(self.recording_indicator)

        control_panel = QGroupBox("Control")
        control_panel.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        resolution_label = QLabel("Resolution:")
        resolution_label.setStyleSheet("font-size: 16px;")
        control_layout.addWidget(resolution_label)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("640×360", (640, 360))
        self.resolution_combo.addItem("1280×720", (1280, 720))
        self.resolution_combo.addItem("1920×1080", (1920, 1080))
        self.resolution_combo.setCurrentIndex(1)
        self.resolution_combo.setStyleSheet("font-size: 16px; padding: 5px;")
        self.resolution_combo.currentIndexChanged.connect(self.on_resolution_changed)
        control_layout.addWidget(self.resolution_combo)

        fps_label = QLabel("FPS:")
        fps_label.setStyleSheet("font-size: 16px;")
        control_layout.addWidget(fps_label)

        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["60", "120"])
        self.fps_combo.setCurrentText("120")
        self.fps_combo.setStyleSheet("font-size: 16px; padding: 5px;")
        self.fps_combo.currentIndexChanged.connect(self.on_fps_changed)
        control_layout.addWidget(self.fps_combo)

        control_layout.addStretch()

        # Кнопка поворота превью
        self.rotate_button = QPushButton("⟳ Rotate")
        self.rotate_button.setStyleSheet("font-size: 14px; padding: 5px;")
        self.rotate_button.clicked.connect(self.rotate_preview)
        control_layout.addWidget(self.rotate_button)

        control_layout.addStretch()

        codec_label = QLabel("Codec:")
        codec_label.setStyleSheet("font-size: 16px;")
        control_layout.addWidget(codec_label)

        self.codec_combo = QComboBox()
        self.codec_combo.addItem("XVID (compressed, small size)", "XVID")
        self.codec_combo.addItem("MJPG (no transcoding)", "MJPG")
        self.codec_combo.setCurrentIndex(0)
        self.codec_combo.setStyleSheet("font-size: 16px; padding: 5px;")
        control_layout.addWidget(self.codec_combo)

        control_layout.addStretch()

        self.ready_indicator = QLabel("●")
        self.ready_indicator.setStyleSheet("color: gray; font-size: 24px;")
        control_layout.addWidget(self.ready_indicator)

        # КНОПКА С ФИКСИРОВАННОЙ ШИРИНОЙ
        self.record_button = QPushButton("Start Recording")
        self.record_button.setEnabled(False)
        self.record_button.setFixedWidth(200)
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #cccccc;
                color: #666666;
                font-size: 18px;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #bbbbbb;
            }
        """)
        self.record_button.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.record_button)

        layout.addWidget(control_panel)

        info_panel = QGroupBox("Information")
        info_panel.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        info_layout = QVBoxLayout()
        info_panel.setLayout(info_layout)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setStyleSheet("font-size: 16px;")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()
        self.camera_label = QLabel("Camera: ELP-USBFHD08S")
        self.camera_label.setStyleSheet("color: #666; font-size: 16px;")
        status_layout.addWidget(self.camera_label)
        info_layout.addLayout(status_layout)

        self.info_label = QLabel("Preview mode: 1280x720 @ 120fps")
        self.info_label.setStyleSheet("font-size: 16px;")
        info_layout.addWidget(self.info_label)

        self.frames_label = QLabel("Frames recorded: 0")
        self.frames_label.setStyleSheet("font-size: 16px;")
        info_layout.addWidget(self.frames_label)

        self.recording_progress = QProgressBar()
        self.recording_progress.setVisible(False)
        self.recording_progress.setTextVisible(False)
        info_layout.addWidget(self.recording_progress)

        save_path_layout = QHBoxLayout()
        save_path_label = QLabel("Save to:")
        save_path_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        save_path_layout.addWidget(save_path_label)

        path_button_layout = QHBoxLayout()
        self.path_label = QLabel(self.video_save_path)
        self.path_label.setStyleSheet("color: #666; font-size: 16px;")
        self.path_label.setWordWrap(True)
        path_button_layout.addWidget(self.path_label, 1)

        self.change_path_button = QPushButton("Change")
        self.change_path_button.setMaximumWidth(100)
        self.change_path_button.setStyleSheet("font-size: 14px; padding: 5px;")
        self.change_path_button.clicked.connect(self.change_save_path)
        path_button_layout.addWidget(self.change_path_button)

        save_path_layout.addLayout(path_button_layout)
        save_path_layout.addStretch()
        info_layout.addLayout(save_path_layout)

        layout.addWidget(info_panel)

        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.blink_indicator)
        self.blink_state = True

    def rotate_preview(self):
        """Поворот превью на 90 градусов по часовой стрелке"""
        self.rotation_angle = (self.rotation_angle + 90) % 360

    def closeEvent(self, event):
        if self.unified_thread:
            self.unified_thread.stop()
        self.blink_timer.stop()
        event.accept()

    def on_resolution_changed(self, index):
        if not self.unified_thread or self.is_recording:
            return

        resolution = self.resolution_combo.currentData()
        current_fps = self.fps_combo.currentText()

        self.fps_combo.blockSignals(True)
        self.fps_combo.clear()

        if resolution == (640, 360):
            self.fps_combo.addItems(["60", "120", "260"])
        elif resolution == (1280, 720):
            self.fps_combo.addItems(["60", "120"])
        else:
            self.fps_combo.addItems(["60"])

        index = self.fps_combo.findText(current_fps)
        if index >= 0:
            self.fps_combo.setCurrentIndex(index)
        else:
            self.fps_combo.setCurrentIndex(0)

        self.fps_combo.blockSignals(False)
        self.apply_camera_mode()

    def on_fps_changed(self, index):
        if not self.unified_thread or self.is_recording:
            return
        self.apply_camera_mode()

    def apply_camera_mode(self):
        if self.is_recording or self.is_switching_mode or not self.unified_thread:
            return

        resolution = self.resolution_combo.currentData()
        fps = int(self.fps_combo.currentText())

        resolution_str = f"{resolution[0]}x{resolution[1]}"
        self.info_label.setText(f"Preview mode: {resolution_str} @ {fps}fps")

        self.set_controls_enabled(False)
        self.video_label.setText("Switching camera mode...")
        self.is_switching_mode = True

        self.unified_thread.set_camera_mode(resolution, fps)

    def on_camera_reinitialized(self):
        self.is_switching_mode = False
        self.set_controls_enabled(True)

    def set_controls_enabled(self, enabled):
        self.resolution_combo.setEnabled(enabled and not self.is_recording)
        self.fps_combo.setEnabled(enabled and not self.is_recording)
        self.codec_combo.setEnabled(enabled and not self.is_recording)
        self.record_button.setEnabled(enabled)
        self.change_path_button.setEnabled(enabled and not self.is_recording)

    def on_camera_connection_lost(self):
        if self.unified_thread:
            self.unified_thread.stop()
        QMessageBox.critical(self, "Camera Error", "Camera connection lost!")
        self.close()

    def on_recording_error(self, error_message):
        if self.is_recording:
            self.unified_thread.stop_recording()
        self.is_recording = False
        self.record_button.setText("Start Recording")
        self.record_button.setEnabled(True)
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.set_controls_enabled(True)
        self.blink_timer.stop()
        self.ready_indicator.setStyleSheet("color: green; font-size: 24px;")
        self.recording_progress.setVisible(False)
        QMessageBox.critical(self, "Recording Error", error_message)

    def update_frame(self, frame):
        if not self.is_switching_mode:
            # Применяем поворот к кадру для превью
            if self.rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)

    def toggle_recording(self):
        if not self.unified_thread:
            return

        if not self.is_recording:
            selected_codec = self.codec_combo.currentData()
            self.unified_thread.set_recording_params(selected_codec, self.video_save_path)
            self.unified_thread.start_recording()
        else:
            self.unified_thread.stop_recording()

    def on_recording_started(self):
        self.is_recording = True
        self.record_button.setText("Stop Recording")
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 18px;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)

        resolution = self.resolution_combo.currentData()
        fps = self.fps_combo.currentText()
        codec = self.codec_combo.currentData()

        self.status_label.setText(f"Status: Recording ({resolution[0]}x{resolution[1]} @ {fps} FPS, {codec})")
        self.resolution_combo.setEnabled(False)
        self.fps_combo.setEnabled(False)
        self.codec_combo.setEnabled(False)
        self.change_path_button.setEnabled(False)

        self.blink_timer.start(500)
        self.recording_progress.setVisible(True)
        self.recording_progress.setRange(0, 0)

    def on_recording_stopped(self, filename):
        self.is_recording = False
        self.record_button.setText("Start Recording")
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.status_label.setText("Status: Ready")
        self.set_controls_enabled(True)
        self.frames_label.setText("Frames recorded: 0")

        self.blink_timer.stop()
        self.ready_indicator.setStyleSheet("color: green; font-size: 24px;")
        self.recording_progress.setVisible(False)

        if filename:
            QMessageBox.information(self, "Recording completed", f"Video saved:\n{filename}")

    def update_frames_count(self, count):
        self.frames_label.setText(f"Frames recorded: {count}")

    def update_recording_status(self, status):
        self.status_label.setText(f"Status: {status}")

    def blink_indicator(self):
        self.ready_indicator.setStyleSheet("color: red; font-size: 24px;" if self.blink_state else "color: darkred; font-size: 24px;")
        self.blink_state = not self.blink_state

    def change_save_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Save Folder", self.video_save_path)
        if folder:
            video_path = str(Path(folder) / 'ELPCameraRecordings')
            Path(video_path).mkdir(parents=True, exist_ok=True)
            self.video_save_path = video_path
            self.settings['video_save_path'] = video_path
            self.save_settings()
            self.path_label.setText(self.video_save_path)
            if self.unified_thread:
                self.unified_thread.set_recording_params(self.codec_combo.currentData(), self.video_save_path)


def main():
    app = QApplication(sys.argv)
    loading_dialog = LoadingDialog()
    loading_dialog.show()
    app.processEvents()
    camera_app = CameraApp(loading_dialog)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
