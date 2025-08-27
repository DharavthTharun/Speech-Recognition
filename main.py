import sys
import gc
import os
import wave
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, 
                             QWidget, QTextEdit, QLineEdit)
from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QMovie, QFontDatabase, QFont, QIcon
import WhisperModel
from debugging import get_gif_for_emotion,detect_emotion
import gc

import sounddevice as sd
import numpy as np
import wave
import gc



import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Traverse up to find the repository root
while current_dir != os.path.dirname(current_dir):  # Stop at the root
    assets_path = os.path.join(current_dir, "Assets")
    if os.path.isdir(assets_path):  # Check if "Assets" exists
        ASSET_FOLDER = assets_path
        break
    current_dir = os.path.dirname(current_dir)
else:
    ASSET_FOLDER = None  # If not found

print("ASSET_FOLDER =", ASSET_FOLDER)



class AudioRecorder(QThread):
    recorded = pyqtSignal(str)
    spectrum_data = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.sample_rate = 44100  # Default sample rate (44.1 kHz)
        self.channels = 1        # Mono recording
        self.chunk = 1024        # Number of frames per buffer
        self.filename = "audio.wav"  # Using WAV format instead of MP3 for simplicity

    def run(self):
        frames = []
        self.is_recording = True

        # Define the callback function to handle audio data and process spectrum
        def callback(indata, frames, time, status):
            if status:
                print(status)
            if self.is_recording:
                # Append the audio data to frames
                frames.append(indata.copy())
                # Convert audio data to numpy array and compute the FFT spectrum
                audio_data = np.frombuffer(indata, dtype=np.int16)
                spectrum = np.abs(np.fft.rfft(audio_data))
                self.spectrum_data.emit(spectrum)

        # Open the stream and start recording
        with sd.InputStream(callback=callback, channels=self.channels, samplerate=self.sample_rate, blocksize=self.chunk):
            print("Recording...")
            while self.is_recording:
                # Wait for the recording to stop
                sd.sleep(100)

        # Save the recorded audio to a WAV file
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 bytes for 16-bit PCM format
            wf.setframerate(self.sample_rate)
            for frame in frames:
                wf.writeframes(frame)

        print(f"Recording saved to {self.filename}")
        self.recorded.emit(self.filename)
        del frames  # Free memory
        gc.collect()

    def stop(self):
        self.is_recording = False
        self.wait()


class TranscriptionThread(QThread):
    transcribed = pyqtSignal(str)

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        try:
            segments, _ = self.model.transcribe("audio.mp3", beam_size=5, language="en", condition_on_previous_text=False)
            text = "\n".join([f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}" for s in segments])
            self.transcribed.emit(text)
        except Exception as e:
            print(f"Transcription error: {e}")
        gc.collect()

class PlayerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice and Expression Recongnition")
        self.setGeometry(100, 100, 1920, 1080)
        self.is_recording = False
        self.model = WhisperModel("distil-large-v3", device="cpu", compute_type="int8")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setStyleSheet("background-color: black;")
        gc.collect()

        
        font_id = QFontDatabase.addApplicationFont(ASSET_FOLDER+"/FSEX300.ttf")
        font_family = QFontDatabase.applicationFontFamilies(font_id)[0] if font_id != -1 else "Arial"
        custom_font = QFont(font_family, 26)
        
        self.gif_label = QLabel(self)
        self.movie = QMovie(ASSET_FOLDER+"/Hello.gif")
        self.gif_label.setMovie(self.movie)
        self.movie.start()
        self.gif_label.setFixedSize(400, 400)
        
        self.input_field = QLineEdit(self)
        self.input_field.setStyleSheet("background-color: black; color: lightgreen; border: 1px solid lightgreen;")
        self.input_field.setFont(custom_font)
        self.input_field.setPlaceholderText("Type here to chat with bot")
        self.input_field.returnPressed.connect(self.send_message)
        
        self.record_button = QPushButton("Start Recording")
        self.record_button.setStyleSheet("background-color: #5CE65C; color: black; border-radius: 5px; padding: 10px;")
        self.record_button.setIcon(QIcon(ASSET_FOLDER+"/mic.png"))
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setIconSize(QSize(8, 8))  

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("background-color: black; color: lightgreen;")
        self.text_display.setFont(custom_font)
        
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setBackground("black")
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen("lightgreen", width=2))
        self.spectrum_plot.setYRange(0, 5500000)
        self.spectrum_plot.setXRange(0, 50)
        self.spectrum_plot.setFixedHeight(150)
        
        layout.addWidget(self.gif_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.input_field)
        layout.addWidget(self.record_button)
        layout.addWidget(self.spectrum_plot)
        layout.addWidget(self.text_display)
        self.setLayout(layout)

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_button.setText("Stop Recording")
            self.recorder = AudioRecorder()
            self.recorder.recorded.connect(self.start_transcription)
            self.recorder.spectrum_data.connect(self.update_spectrum)
            self.recorder.start()
        else:
            self.is_recording = False
            self.record_button.setText("Start Recording")
            self.recorder.stop()
            self.recorder.deleteLater()
            gc.collect()

    def send_message(self):
        text = self.input_field.text().strip()
        if text:
            emotion = detect_emotion(text) or "neutral"
            e= emotion.capitalize()
            self.input_field.clear()
            self.text_display.append(f"User Typed message: {text}     (***{e}***)")
    
        if e=="Anger" or e=="anger" or e=="ANGER":
            self.change_gif("upset")
        else :
            self.change_gif(get_gif_for_emotion(emotion))
            self.input_field.clear()
            self.change_gif(get_gif_for_emotion(emotion))
            #print(get_gif_for_emotion(emotion))
            gc.collect()

    def change_gif(self,emotion):
        #print("to be changed"+emotion)
        path=ASSET_FOLDER
        final_path=ASSET_FOLDER+"/"+emotion
        #print("Final path"+final_path)

        label_width = self.gif_label.width()
        label_height = self.gif_label.height()

        if self.movie:
            self.movie.stop()
            self.movie.deleteLater()  
            self.movie = None

        self.gif_label.clear()  # Clear QLabel
        QApplication.processEvents()
        self.movie = QMovie(final_path)

        if not self.movie.isValid():
            print(f"Invalid GIF file: {final_path}")
            return
        
        self.movie.setScaledSize(QSize(label_width, label_height))
        self.gif_label.setMovie(self.movie)
        self.movie.start()

    # Force UI update
        self.gif_label.repaint()
        QApplication.processEvents()

    def start_transcription(self, filename):
        if os.path.exists(filename):
            self.transcriber = TranscriptionThread(self.model)
            self.transcriber.transcribed.connect(self.display_transcription)
            self.transcriber.start()


    def display_transcription(self, text):
        
        self.record_button.setEnabled(True)
        emotion = detect_emotion(text) or "neutral"
        e= (emotion).capitalize()
        self.text_display.append(f"User voice message: {text}     (***{e}***)")
        
        if e=="Anger" or e=="anger" or e=="ANGER":
            self.change_gif("upset")
        else :
            self.change_gif(get_gif_for_emotion(emotion))
        #print(get_gif_for_emotion(emotion))

    def update_spectrum(self, spectrum):
        self.spectrum_curve.setData(spectrum)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = PlayerApp()
    player.show()
    sys.exit(app.exec_())
