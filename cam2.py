import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QComboBox,
                            QMessageBox, QGroupBox, QProgressBar, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap, QFont
import time
from datetime import datetime
import platform
import threading
import json
import os
from pathlib import Path

# Suppress OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

class UnifiedCameraThread(QThread):
    """Single thread for both preview and recording"""
    frameReady = pyqtSignal(np.ndarray)
    connectionLost = pyqtSignal()
    recordingStarted = pyqtSignal()
    recordingStopped = pyqtSignal(str)
    framesCaptured = pyqtSignal(int)
    recordingProgress = pyqtSignal(str)
    recordingError = pyqtSignal(str)
    cameraReinitialized = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.is_running = False
        self.is_recording = False
        self.needs_reinit = False
        self.camera_index = 0
        self.mutex = QMutex()
        
        # Default camera mode - start with 720p
        self.current_resolution = (1280, 720)
        self.current_fps = 120
        
        # Recording parameters
        self.codec = "XVID"
        self.video_save_path = ""
        self.out = None
        self.frames_captured = 0
        self.recording_filename = ""
        self.start_time = 0
        
    def set_camera_mode(self, resolution, fps):
        """Request camera mode change"""
        with QMutexLocker(self.mutex):
            if resolution != self.current_resolution or fps != self.current_fps:
                self.current_resolution = resolution
                self.current_fps = fps
                self.needs_reinit = True
                print(f"Camera mode change requested: {resolution[0]}x{resolution[1]} @ {fps}fps")
                
    def set_recording_params(self, codec, save_path):
        """Set recording parameters"""
        with QMutexLocker(self.mutex):
            self.codec = codec
            self.video_save_path = save_path
            
    def initialize_camera(self):
        """Initialize or reinitialize camera with current settings"""
        try:
            # Release existing camera if open
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                time.sleep(0.2)  # Give camera time to release
            
            print(f"Initializing camera: {self.current_resolution[0]}x{self.current_resolution[1]} @ {self.current_fps}fps")
            
            # Open camera with DirectShow on Windows
            if platform.system() == 'Windows':
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print("Failed to open camera")
                return False
            
            # Set camera parameters
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.current_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.current_resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.current_fps)
            
            # Verify actual settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera actually initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Read a test frame to ensure camera is working
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read test frame")
                return False
            
            # Signal that camera is ready
            self.cameraReinitialized.emit()
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            return False
        
    def run(self):
        """Main thread loop"""
        # Initial camera setup
        if not self.initialize_camera():
            self.connectionLost.emit()
            return
        
        self.is_running = True
        consecutive_failures = 0
        
        while self.is_running:
            try:
                # Check if we need to reinitialize camera
                with QMutexLocker(self.mutex):
                    if self.needs_reinit and not self.is_recording:
                        self.needs_reinit = False
                        print("Reinitializing camera...")
                        if not self.initialize_camera():
                            self.connectionLost.emit()
                            break
                        consecutive_failures = 0
                        continue
                
                # Read frame
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        consecutive_failures = 0
                        
                        # Always emit frame for preview
                        self.frameReady.emit(frame)
                        
                        # If recording, write frame to file
                        with QMutexLocker(self.mutex):
                            if self.is_recording and self.out:
                                self.out.write(frame)
                                self.frames_captured += 1
                                self.framesCaptured.emit(self.frames_captured)
                            
                    else:
                        consecutive_failures += 1
                        if consecutive_failures > 10:
                            print(f"Too many consecutive failures: {consecutive_failures}")
                            self.connectionLost.emit()
                            break
                        time.sleep(0.01)
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Error in camera thread: {str(e)}")
                time.sleep(0.01)
                
        # Cleanup
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
            
    def start_recording(self):
        """Start recording immediately"""
        with QMutexLocker(self.mutex):
            if self.is_recording:
                return
                
            # Create filename
            videos_dir = Path(self.video_save_path)
            videos_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            resolution_str = f"{self.current_resolution[0]}x{self.current_resolution[1]}"
            self.recording_filename = str(videos_dir / f"recording_{timestamp}_{resolution_str}_{self.current_fps}fps_{self.codec}.avi")
            
            # Initialize VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.out = cv2.VideoWriter(self.recording_filename, fourcc, self.current_fps, self.current_resolution)
            
            if not self.out.isOpened():
                self.recordingError.emit("Failed to initialize video writer")
                return
                
            # Reset counters
            self.frames_captured = 0
            self.start_time = time.time()
            
            # Set recording flag
            self.is_recording = True
            
            # Signal recording started
            self.recordingStarted.emit()
            self.recordingProgress.emit("Recording started")
            
    def stop_recording(self):
        """Stop recording"""
        with QMutexLocker(self.mutex):
            if not self.is_recording:
                return
                
            self.is_recording = False
            
            # Close video writer
            if self.out:
                self.out.release()
                self.out = None
                
            # Calculate duration
            duration = time.time() - self.start_time
            
            # Send completion signal
            if self.frames_captured > 0:
                self.recordingStopped.emit(f"{self.recording_filename} (duration: {duration:.1f} sec)")
            else:
                # Remove empty file
                try:
                    os.remove(self.recording_filename)
                except:
                    pass
                self.recordingStopped.emit("")
                
    def stop(self):
        """Stop the thread"""
        self.stop_recording()  # Stop recording if active
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.wait()

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.unified_thread = None
        self.is_recording = False
        self.is_switching_mode = False
        
        # Setup settings path
        if platform.system() == 'Windows':
            app_data = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
            settings_dir = Path(app_data) / 'ELPCameraControl'
        else:
            settings_dir = Path.home() / '.config' / 'ELPCameraControl'
        
        settings_dir.mkdir(parents=True, exist_ok=True)
        self.settings_file = str(settings_dir / 'camera_settings.json')
        print(f"Settings file location: {self.settings_file}")
        
        # Load settings
        self.settings = self.load_settings()
        
        # Check video save path
        self.video_save_path = self.check_video_save_path()
        
        # Find camera
        self.camera_index = self.load_or_find_camera()
        
        # Exit if no camera found
        if self.camera_index == -1:
            QMessageBox.critical(None, "Camera Error",
                               "ELP camera not found!\n\n"
                               "Please check that the camera is connected and try again.")
            sys.exit(1)
        
        self.init_ui()
        
        # Initialize and start camera thread AFTER UI is ready
        self.init_camera_thread()
        
    def init_camera_thread(self):
        """Initialize camera thread with current settings"""
        # Create thread
        self.unified_thread = UnifiedCameraThread()
        
        # Set camera index
        self.unified_thread.camera_index = self.camera_index
        
        # Get initial mode from UI
        resolution = self.resolution_combo.currentData()
        fps = int(self.fps_combo.currentText())
        
        # Set initial mode
        self.unified_thread.current_resolution = resolution
        self.unified_thread.current_fps = fps
        
        # Set recording parameters
        self.unified_thread.set_recording_params(
            self.codec_combo.currentData(),
            self.video_save_path
        )
        
        # Connect signals
        self.unified_thread.frameReady.connect(self.update_frame)
        self.unified_thread.connectionLost.connect(self.on_camera_connection_lost)
        self.unified_thread.recordingStarted.connect(self.on_recording_started)
        self.unified_thread.recordingStopped.connect(self.on_recording_stopped)
        self.unified_thread.framesCaptured.connect(self.update_frames_count)
        self.unified_thread.recordingProgress.connect(self.update_recording_status)
        self.unified_thread.recordingError.connect(self.on_recording_error)
        self.unified_thread.cameraReinitialized.connect(self.on_camera_reinitialized)
        
        # Update info label
        resolution_str = f"{resolution[0]}x{resolution[1]}"
        self.info_label.setText(f"Preview mode: {resolution_str} @ {fps}fps")
        
        # Start thread
        self.unified_thread.start()
        
    def load_settings(self):
        """Load settings from file"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_settings(self):
        """Save settings to file"""
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=2)
    
    def check_video_save_path(self):
        """Check and set video save path"""
        video_path = self.settings.get('video_save_path', '')
        
        if video_path and os.path.exists(video_path) and os.path.isdir(video_path):
            print(f"Using saved video path: {video_path}")
            return video_path
        
        # Ask user for path if not set
        msg = QMessageBox()
        msg.setWindowTitle("Select Video Save Location")
        msg.setText("Please select a folder where videos will be saved.")
        msg.setInformativeText("You can change this later in the settings.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
        default_path = str(Path.home() / 'Videos')
        folder = QFileDialog.getExistingDirectory(
            None,
            "Select Video Save Folder",
            default_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder:
            video_path = str(Path(folder) / 'ELPCameraRecordings')
            Path(video_path).mkdir(parents=True, exist_ok=True)
            
            self.settings['video_save_path'] = video_path
            self.save_settings()
            
            print(f"Video save path set to: {video_path}")
            return video_path
        else:
            # Use default path if user cancels
            default_video_path = str(Path.home() / 'Videos' / 'ELPCameraRecordings')
            Path(default_video_path).mkdir(parents=True, exist_ok=True)
            
            self.settings['video_save_path'] = default_video_path
            self.save_settings()
            
            QMessageBox.information(None, "Default Path",
                                  f"Videos will be saved to:\n{default_video_path}")
            return default_video_path
    
    def load_or_find_camera(self):
        """Load saved camera index or find new one"""
        saved_index = self.settings.get('camera_index', -1)
        
        if saved_index >= 0:
            # Quick check if camera is still available
            if platform.system() == 'Windows':
                cap = cv2.VideoCapture(saved_index, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(saved_index)
            
            if cap.isOpened():
                # Check if it's still ELP camera
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                cap.set(cv2.CAP_PROP_FPS, 260)
                
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                if width == 640 and height == 360 and fps >= 200:
                    print(f"Using saved camera: index {saved_index}")
                    return saved_index
        
        return self.find_elp_camera()
        
    def save_camera_settings(self, camera_index):
        """Save camera index for quick startup"""
        self.settings['camera_index'] = camera_index
        self.save_settings()
            
    def find_elp_camera(self):
        """Find ELP camera among available devices"""
        print("Searching for ELP camera...")
        
        priority_indices = [1, 2, 0, 3, 4]
        other_indices = [i for i in range(5, 10)]
        
        for i in priority_indices + other_indices:
            if platform.system() == 'Windows':
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                # Check for ELP-specific capabilities
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                cap.set(cv2.CAP_PROP_FPS, 260)
                
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                
                # ELP camera should support 640x360 @ 260fps
                if actual_width == 640 and actual_height == 360 and actual_fps >= 200:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Additional check - 1920x1080 support
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                        cap.set(cv2.CAP_PROP_FPS, 60)
                        
                        check_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        check_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        
                        if check_width == 1920 and check_height == 1080:
                            print(f"Found ELP camera at index {i}")
                            cap.release()
                            self.save_camera_settings(i)
                            return i
                
                cap.release()
        
        print("ELP camera not found")
        return -1
        
    def init_ui(self):
        self.setWindowTitle('ELP Camera Control')
        self.setGeometry(100, 100, 1000, 800)

        # Set global font
        font = QFont()
        font.setPointSize(14)

        # Main widget
        main_widget = QWidget()
        main_widget.setFont(font)
        self.setCentralWidget(main_widget)

        # Main layout
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Video display area
        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("border: 2px solid black; font-size: 16px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Camera initialization...")
        layout.addWidget(self.video_label)

        # Recording indicator
        self.recording_indicator = QLabel()
        self.recording_indicator.setAlignment(Qt.AlignCenter)
        self.recording_indicator.setStyleSheet("color: red; font-weight: bold; font-size: 18px;")
        self.recording_indicator.setVisible(False)
        layout.addWidget(self.recording_indicator)

        # Control panel
        control_panel = QGroupBox("Control")
        control_panel.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        # Resolution selector
        resolution_label = QLabel("Resolution:")
        resolution_label.setStyleSheet("font-size: 16px;")
        control_layout.addWidget(resolution_label)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("640×360", (640, 360))
        self.resolution_combo.addItem("1280×720", (1280, 720))
        self.resolution_combo.addItem("1920×1080", (1920, 1080))
        self.resolution_combo.setCurrentIndex(1)  # Default to 720p
        self.resolution_combo.setStyleSheet("font-size: 16px; padding: 5px;")
        self.resolution_combo.currentIndexChanged.connect(self.on_resolution_changed)
        control_layout.addWidget(self.resolution_combo)

        # FPS selector
        fps_label = QLabel("FPS:")
        fps_label.setStyleSheet("font-size: 16px;")
        control_layout.addWidget(fps_label)

        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["60", "120"])  # Will be updated based on resolution
        self.fps_combo.setCurrentText("120")
        self.fps_combo.setStyleSheet("font-size: 16px; padding: 5px;")
        self.fps_combo.currentIndexChanged.connect(self.on_fps_changed)
        control_layout.addWidget(self.fps_combo)

        control_layout.addStretch()

        # Codec selector
        codec_label = QLabel("Codec:")
        codec_label.setStyleSheet("font-size: 16px;")
        control_layout.addWidget(codec_label)

        self.codec_combo = QComboBox()
        self.codec_combo.addItem("XVID (compressed, small size)", "XVID")
        self.codec_combo.addItem("MJPG (no transcoding)", "MJPG")
        self.codec_combo.setCurrentIndex(0)
        self.codec_combo.setToolTip("XVID - good balance of quality and size\nMJPG - original quality from camera")
        self.codec_combo.setStyleSheet("font-size: 16px; padding: 5px;")
        control_layout.addWidget(self.codec_combo)

        control_layout.addStretch()

        # Ready indicator
        self.ready_indicator = QLabel("●")
        self.ready_indicator.setStyleSheet("color: green; font-size: 24px;")
        control_layout.addWidget(self.ready_indicator)

        # Record button
        self.record_button = QPushButton("Start Recording")
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                padding: 12px;
                border-radius: 5px;
                min-width: 180px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.record_button.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.record_button)

        layout.addWidget(control_panel)

        # Information panel
        info_panel = QGroupBox("Information")
        info_panel.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        info_layout = QVBoxLayout()
        info_panel.setLayout(info_layout)

        # Status and camera info
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Ready")
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

        # Recording progress bar
        self.recording_progress = QProgressBar()
        self.recording_progress.setVisible(False)
        self.recording_progress.setTextVisible(False)
        self.recording_progress.setStyleSheet("QProgressBar { font-size: 16px; }")
        info_layout.addWidget(self.recording_progress)

        # Save path info
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

        # Timer for recording indicator blink
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.blink_indicator)
        self.blink_state = True
        
    def closeEvent(self, event):
        """Handle application close"""
        if self.unified_thread:
            self.unified_thread.stop()
        self.blink_timer.stop()
        event.accept()
        
    def on_resolution_changed(self, index):
        """Handle resolution change"""
        if not self.unified_thread:
            return
            
        resolution = self.resolution_combo.currentData()
        current_fps = self.fps_combo.currentText()
        
        # Update available FPS options
        self.fps_combo.blockSignals(True)  # Block signals to prevent double update
        self.fps_combo.clear()
        
        if resolution == (640, 360):
            self.fps_combo.addItems(["60", "120", "260"])
        elif resolution == (1280, 720):
            self.fps_combo.addItems(["60", "120"])
        else:  # 1920x1080
            self.fps_combo.addItems(["60"])
        
        # Try to restore previous FPS selection
        index = self.fps_combo.findText(current_fps)
        if index >= 0:
            self.fps_combo.setCurrentIndex(index)
        else:
            self.fps_combo.setCurrentIndex(0)
            
        self.fps_combo.blockSignals(False)  # Re-enable signals
            
        # Apply camera mode change
        self.apply_camera_mode()
    
    def on_fps_changed(self, index):
        """Handle FPS change"""
        if not self.unified_thread:
            return
        self.apply_camera_mode()
    
    def apply_camera_mode(self):
        """Apply selected camera mode"""
        if self.is_recording or self.is_switching_mode:
            return  # Don't change mode during recording or while already switching
            
        resolution = self.resolution_combo.currentData()
        fps_text = self.fps_combo.currentText()
        if not fps_text:  # Safety check
            return
        fps = int(fps_text)
        
        # Update info label
        resolution_str = f"{resolution[0]}x{resolution[1]}"
        self.info_label.setText(f"Preview mode: {resolution_str} @ {fps}fps")
        
        # Disable controls during mode switch
        self.set_controls_enabled(False)
        self.video_label.setText("Switching camera mode...")
        self.is_switching_mode = True
        
        # Request camera mode change
        self.unified_thread.set_camera_mode(resolution, fps)
        
    def on_camera_reinitialized(self):
        """Handle camera reinitialization complete"""
        self.is_switching_mode = False
        self.set_controls_enabled(True)
        
    def set_controls_enabled(self, enabled):
        """Enable/disable UI controls"""
        self.resolution_combo.setEnabled(enabled and not self.is_recording)
        self.fps_combo.setEnabled(enabled and not self.is_recording)
        self.codec_combo.setEnabled(enabled and not self.is_recording)
        self.record_button.setEnabled(enabled)
        self.change_path_button.setEnabled(enabled and not self.is_recording)
        
    def on_camera_connection_lost(self):
        """Handle camera connection loss"""
        if self.unified_thread:
            self.unified_thread.stop()
        
        QMessageBox.critical(self, "Camera Error",
                           "Camera connection lost!\n\n"
                           "Please check the camera connection and restart the application.")
        self.close()
        
    def on_recording_error(self, error_message):
        """Handle recording errors"""
        if self.is_recording:
            self.unified_thread.stop_recording()
            self.is_recording = False
        
        # Reset UI
        self.record_button.setText("Start Recording")
        self.record_button.setEnabled(True)
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                padding: 12px;
                border-radius: 5px;
                min-width: 180px;
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
        """Update preview display"""
        if not self.is_switching_mode:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # Scale for display
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
        
    def toggle_recording(self):
        """Toggle recording - INSTANT START!"""
        if not self.unified_thread:
            return
            
        if not self.is_recording:
            # Get selected codec
            selected_codec = self.codec_combo.currentData()
            
            # Update recording parameters
            self.unified_thread.set_recording_params(selected_codec, self.video_save_path)
            
            # Start recording immediately
            self.unified_thread.start_recording()
            
        else:
            # Stop recording
            self.unified_thread.stop_recording()
        
    def on_recording_started(self):
        """Handle recording start"""
        self.is_recording = True
        self.record_button.setText("Stop Recording")
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 18px;
                padding: 12px;
                border-radius: 5px;
                min-width: 180px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        
        resolution = self.resolution_combo.currentData()
        resolution_str = f"{resolution[0]}x{resolution[1]}"
        fps = self.fps_combo.currentText()
        codec = self.codec_combo.currentData()
        self.status_label.setText(f"Status: Recording ({resolution_str} @ {fps} FPS, {codec})")
        
        self.resolution_combo.setEnabled(False)
        self.fps_combo.setEnabled(False)
        self.codec_combo.setEnabled(False)
        self.change_path_button.setEnabled(False)
        
        # Start blink animation
        self.blink_timer.start(500)
        self.recording_progress.setVisible(True)
        self.recording_progress.setRange(0, 0)  # Indeterminate progress
        
    def on_recording_stopped(self, filename):
        """Handle recording stop"""
        self.is_recording = False
        self.record_button.setText("Start Recording")
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                padding: 12px;
                border-radius: 5px;
                min-width: 180px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.status_label.setText("Status: Ready")
        self.resolution_combo.setEnabled(True)
        self.fps_combo.setEnabled(True)
        self.codec_combo.setEnabled(True)
        self.change_path_button.setEnabled(True)
        self.frames_label.setText("Frames recorded: 0")
        
        # Stop animation
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
        if self.blink_state:
            self.ready_indicator.setStyleSheet("color: red; font-size: 24px;")
        else:
            self.ready_indicator.setStyleSheet("color: darkred; font-size: 24px;")
        self.blink_state = not self.blink_state
        
    def change_save_path(self):
        """Change video save path"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Video Save Folder",
            self.video_save_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
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
    camera_app = CameraApp()
    camera_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()