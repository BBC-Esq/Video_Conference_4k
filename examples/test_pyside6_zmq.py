import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from videoconference4k import SyncTransport
from videoconference4k.utils import has_nvidia_codec, get_nvidia_info


class ReceiverThread(QThread):
    frame_received = Signal(np.ndarray)
    error_occurred = Signal(str)

    def __init__(self, transport):
        super().__init__()
        self.transport = transport
        self.running = True

    def run(self):
        while self.running:
            try:
                frame = self.transport.recv()
                if frame is None:
                    break
                self.frame_received.emit(frame.copy())
            except Exception as e:
                self.error_occurred.emit(str(e))
                break

    def stop(self):
        self.running = False


class SenderThread(QThread):
    error_occurred = Signal(str)

    def __init__(self, transport, capture):
        super().__init__()
        self.transport = transport
        self.capture = capture
        self.running = True

    def run(self):
        while self.running:
            try:
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    continue
                self.transport.send(frame)
            except Exception as e:
                self.error_occurred.emit(str(e))
                break

    def stop(self):
        self.running = False


class TransportTestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SyncTransport Test")
        self.setMinimumSize(700, 600)

        self.transport = None
        self.capture = None
        self.receiver_thread = None
        self.sender_thread = None

        self.local_timer = QTimer()
        self.local_timer.timeout.connect(self.update_local_video)

        self.setup_ui()
        self.check_gpu_status()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.gpu_status_label = QLabel("Checking GPU status...")
        self.gpu_status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.gpu_status_label)

        video_layout = QHBoxLayout()

        local_container = QVBoxLayout()
        local_title = QLabel("Local Video")
        local_title.setAlignment(Qt.AlignCenter)
        self.local_video_label = QLabel()
        self.local_video_label.setMinimumSize(320, 240)
        self.local_video_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        self.local_video_label.setAlignment(Qt.AlignCenter)
        local_container.addWidget(local_title)
        local_container.addWidget(self.local_video_label)

        remote_container = QVBoxLayout()
        remote_title = QLabel("Remote Video")
        remote_title.setAlignment(Qt.AlignCenter)
        self.remote_video_label = QLabel()
        self.remote_video_label.setMinimumSize(320, 240)
        self.remote_video_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        self.remote_video_label.setAlignment(Qt.AlignCenter)
        remote_container.addWidget(remote_title)
        remote_container.addWidget(self.remote_video_label)

        video_layout.addLayout(local_container)
        video_layout.addLayout(remote_container)
        layout.addLayout(video_layout)

        config_layout = QHBoxLayout()

        mode_layout = QVBoxLayout()
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Sender", "Receiver"])
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        config_layout.addLayout(mode_layout)

        address_layout = QVBoxLayout()
        address_label = QLabel("Address:")
        self.address_input = QLineEdit("localhost")
        address_layout.addWidget(address_label)
        address_layout.addWidget(self.address_input)
        config_layout.addLayout(address_layout)

        port_layout = QVBoxLayout()
        port_label = QLabel("Port:")
        self.port_input = QLineEdit("5555")
        port_layout.addWidget(port_label)
        port_layout.addWidget(self.port_input)
        config_layout.addLayout(port_layout)

        layout.addLayout(config_layout)

        self.gpu_checkbox = QCheckBox("Enable GPU Acceleration (NVIDIA NVENC)")
        self.gpu_checkbox.setChecked(False)
        layout.addWidget(self.gpu_checkbox)

        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_transport)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.clicked.connect(self.stop_transport)
        self.stop_btn.setEnabled(False)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)

        self.status_label = QLabel("Status: Not started")
        self.status_label.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(self.status_label)

    def check_gpu_status(self):
        if has_nvidia_codec():
            info = get_nvidia_info()
            self.gpu_status_label.setText("GPU Status: NVIDIA hardware encoding available")
            self.gpu_status_label.setStyleSheet("font-weight: bold; padding: 5px; color: green;")
            self.gpu_checkbox.setEnabled(True)
        else:
            self.gpu_status_label.setText("GPU Status: Not available (CPU encoding only)")
            self.gpu_status_label.setStyleSheet("font-weight: bold; padding: 5px; color: orange;")
            self.gpu_checkbox.setChecked(False)
            self.gpu_checkbox.setEnabled(False)

    def start_transport(self):
        mode = self.mode_combo.currentText()
        address = self.address_input.text().strip()
        port = self.port_input.text().strip()
        gpu_enabled = self.gpu_checkbox.isChecked()

        encoding_type = "GPU (NVENC)" if gpu_enabled else "CPU (JPEG)"

        try:
            if mode == "Sender":
                self.capture = cv2.VideoCapture(0)
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                self.transport = SyncTransport(
                    address=address,
                    port=port,
                    receive_mode=False,
                    gpu_accelerated=gpu_enabled,
                    logging=True,
                )

                self.sender_thread = SenderThread(self.transport, self.capture)
                self.sender_thread.error_occurred.connect(self.on_error)
                self.sender_thread.start()

                self.local_timer.start(33)
                self.status_label.setText(f"Status: Sending to {address}:{port} [{encoding_type}]")

            else:
                self.transport = SyncTransport(
                    address="*",
                    port=port,
                    receive_mode=True,
                    gpu_accelerated=gpu_enabled,
                    logging=True,
                )

                self.receiver_thread = ReceiverThread(self.transport)
                self.receiver_thread.frame_received.connect(self.update_remote_video)
                self.receiver_thread.error_occurred.connect(self.on_error)
                self.receiver_thread.start()

                self.status_label.setText(f"Status: Receiving on port {port} [{encoding_type}]")

            self.status_label.setStyleSheet("font-weight: bold; padding: 10px; color: green;")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.mode_combo.setEnabled(False)
            self.address_input.setEnabled(False)
            self.port_input.setEnabled(False)
            self.gpu_checkbox.setEnabled(False)

        except Exception as e:
            self.status_label.setText(f"Status: Error - {str(e)}")
            self.status_label.setStyleSheet("font-weight: bold; padding: 10px; color: red;")

    def stop_transport(self):
        self.local_timer.stop()

        if self.sender_thread:
            self.sender_thread.stop()
            self.sender_thread.wait()
            self.sender_thread = None

        if self.receiver_thread:
            self.receiver_thread.stop()
            self.receiver_thread.wait()
            self.receiver_thread = None

        if self.transport:
            self.transport.close()
            self.transport = None

        if self.capture:
            self.capture.release()
            self.capture = None

        self.local_video_label.clear()
        self.remote_video_label.clear()

        self.status_label.setText("Status: Stopped")
        self.status_label.setStyleSheet("font-weight: bold; padding: 10px;")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.address_input.setEnabled(True)
        self.port_input.setEnabled(True)
        if has_nvidia_codec():
            self.gpu_checkbox.setEnabled(True)

    def update_local_video(self):
        if self.capture:
            ret, frame = self.capture.read()
            if ret and frame is not None:
                self.display_frame(frame, self.local_video_label)

    def update_remote_video(self, frame):
        self.display_frame(frame, self.remote_video_label)

    def display_frame(self, frame, label):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def on_error(self, error_msg):
        self.status_label.setText(f"Status: Error - {error_msg}")
        self.status_label.setStyleSheet("font-weight: bold; padding: 10px; color: red;")

    def closeEvent(self, event):
        self.stop_transport()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TransportTestApp()
    window.show()
    sys.exit(app.exec())