import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from videoconference4k import PeerConference


class VideoSignals(QObject):
    frame_received = Signal(np.ndarray)
    connected = Signal()
    disconnected = Signal()


class VideoConferenceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Conference")
        self.setMinimumSize(900, 700)

        self.conference = None
        self.signals = VideoSignals()

        self.signals.frame_received.connect(self.update_remote_video)
        self.signals.connected.connect(self.on_connected)
        self.signals.disconnected.connect(self.on_disconnected)

        self.local_video_timer = QTimer()
        self.local_video_timer.timeout.connect(self.update_local_video)

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        video_layout = QHBoxLayout()

        local_video_container = QVBoxLayout()
        local_title = QLabel("Local Video")
        local_title.setAlignment(Qt.AlignCenter)
        self.local_video_label = QLabel()
        self.local_video_label.setMinimumSize(400, 300)
        self.local_video_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        self.local_video_label.setAlignment(Qt.AlignCenter)
        local_video_container.addWidget(local_title)
        local_video_container.addWidget(self.local_video_label)

        remote_video_container = QVBoxLayout()
        remote_title = QLabel("Remote Video")
        remote_title.setAlignment(Qt.AlignCenter)
        self.remote_video_label = QLabel()
        self.remote_video_label.setMinimumSize(400, 300)
        self.remote_video_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        self.remote_video_label.setAlignment(Qt.AlignCenter)
        remote_video_container.addWidget(remote_title)
        remote_video_container.addWidget(self.remote_video_label)

        video_layout.addLayout(local_video_container)
        video_layout.addLayout(remote_video_container)
        layout.addLayout(video_layout)

        control_layout = QHBoxLayout()
        self.create_invite_btn = QPushButton("Create Invite")
        self.create_invite_btn.setMinimumHeight(40)
        self.create_invite_btn.clicked.connect(self.create_invite)

        self.join_btn = QPushButton("Join / Complete")
        self.join_btn.setMinimumHeight(40)
        self.join_btn.clicked.connect(self.join_conference)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.clicked.connect(self.stop_conference)
        self.stop_btn.setEnabled(False)

        control_layout.addWidget(self.create_invite_btn)
        control_layout.addWidget(self.join_btn)
        control_layout.addWidget(self.stop_btn)
        layout.addLayout(control_layout)

        input_label = QLabel("Paste invite or response code here:")
        layout.addWidget(input_label)

        self.code_input = QLineEdit()
        self.code_input.setMinimumHeight(35)
        layout.addWidget(self.code_input)

        output_label = QLabel("Your code to share:")
        layout.addWidget(output_label)

        self.code_output = QTextEdit()
        self.code_output.setReadOnly(True)
        self.code_output.setMaximumHeight(80)
        layout.addWidget(self.code_output)

        self.status_label = QLabel("Status: Not connected")
        self.status_label.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(self.status_label)

    def create_invite(self):
        if self.conference:
            self.stop_conference()

        self.conference = PeerConference(resolution=(1280, 720), framerate=30)

        @self.conference.on_remote_video
        def handle_video(frame):
            self.signals.frame_received.emit(frame.copy())

        @self.conference.on_connected
        def handle_connected():
            self.signals.connected.emit()

        @self.conference.on_disconnected
        def handle_disconnected():
            self.signals.disconnected.emit()

        invite_code = self.conference.create_invite()
        self.code_output.setText(invite_code)
        self.status_label.setText("Status: Invite created. Share the code and wait for response.")
        self.stop_btn.setEnabled(True)
        self.create_invite_btn.setEnabled(False)
        self.local_video_timer.start(33)

    def join_conference(self):
        code = self.code_input.text().strip()
        if not code:
            self.status_label.setText("Status: Please enter a code first")
            return

        if self.conference is None:
            self.conference = PeerConference(resolution=(1280, 720), framerate=30)

            @self.conference.on_remote_video
            def handle_video(frame):
                self.signals.frame_received.emit(frame.copy())

            @self.conference.on_connected
            def handle_connected():
                self.signals.connected.emit()

            @self.conference.on_disconnected
            def handle_disconnected():
                self.signals.disconnected.emit()

            response_code = self.conference.accept_invite(code)
            self.code_output.setText(response_code)
            self.status_label.setText("Status: Response created. Share the code and wait for connection.")
            self.local_video_timer.start(33)
        else:
            self.conference.complete_connection(code)
            self.status_label.setText("Status: Connecting...")

        self.stop_btn.setEnabled(True)
        self.create_invite_btn.setEnabled(False)

    def stop_conference(self):
        self.local_video_timer.stop()

        if self.conference:
            self.conference.stop()
            self.conference = None

        self.status_label.setText("Status: Not connected")
        self.stop_btn.setEnabled(False)
        self.create_invite_btn.setEnabled(True)

        self.local_video_label.clear()
        self.remote_video_label.clear()
        self.code_output.clear()
        self.code_input.clear()

    def update_local_video(self):
        if self.conference:
            frame = self.conference.get_local_frame()
            if frame is not None:
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

    def on_connected(self):
        self.status_label.setText("Status: Connected!")
        self.status_label.setStyleSheet("font-weight: bold; padding: 10px; color: green;")

    def on_disconnected(self):
        self.status_label.setText("Status: Disconnected")
        self.status_label.setStyleSheet("font-weight: bold; padding: 10px; color: red;")

    def closeEvent(self, event):
        self.stop_conference()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoConferenceApp()
    window.show()
    sys.exit(app.exec())