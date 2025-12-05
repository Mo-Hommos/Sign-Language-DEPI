import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QShortcut)
from PyQt5.QtCore import Qt, QRect, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QRegion, QPainter, QPen, QKeySequence, QImage, QPixmap

import time
import speech_recognition as sr
import sys
import os

def get_model_path():
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = sys._MEIPASS
    else:
        # Running as script
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, 'Arabic_Sign_Language_CNN_Final.h5')


class SignImageDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Position at bottom right of screen
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen.width() - 320, screen.height() - 370, 300, 350)
        
        # Setup layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 3px solid #2196F3;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        self.image_label.setMinimumSize(280, 280)
        self.image_label.setMaximumSize(280, 280)
        
        # Letter name label
        self.letter_label = QLabel("حرف")
        self.letter_label.setAlignment(Qt.AlignCenter)
        self.letter_label.setStyleSheet("""
            QLabel {
                background-color: #2196F3;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        layout.addWidget(self.image_label)
        layout.addWidget(self.letter_label)
        
    def show_image(self, image_path, letter):
        """Display a sign image"""
        try:
            # Read image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                print(f"Image not found: {image_path}")
                return False
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to fit label
            height, width = img_rgb.shape[:2]
            max_size = 260
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            
            img_resized = cv2.resize(img_rgb, (new_width, new_height))
            
            # Convert to QPixmap
            height, width, channel = img_resized.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Display
            self.image_label.setPixmap(pixmap)
            self.letter_label.setText(letter)
            self.show()
            return True
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            return False
    
    def clear_image(self):
        self.image_label.clear()
        self.hide()


class SignSequenceThread(QThread):
    image_ready = pyqtSignal(str, str)  # image_path, letter
    sequence_complete = pyqtSignal()
    
    def __init__(self, text, word_to_image, signs_path, display_time=1.0):
        super().__init__()
        self.text = text
        self.word_to_image = word_to_image
        self.signs_path = signs_path
        self.display_time = display_time
        self.is_running = False
    
    def run(self):
        """Display each letter's sign"""
        self.is_running = True
        for char in self.text:
            if not self.is_running:
                break
                
            if char in self.word_to_image:
                image_path = os.path.join(self.signs_path, self.word_to_image[char])
                self.image_ready.emit(image_path, char)
                time.sleep(self.display_time)
        
        self.sequence_complete.emit()
    
    def stop(self):
        self.is_running = False


class VoiceRecognitionThread(QThread):
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.is_running = False
    
    def run(self):
        self.is_running = True
        try:
            with sr.Microphone() as source:
                self.result_ready.emit("جاري الاستماع...")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                self.result_ready.emit("جاري التعرف...")
                
                # Try Arabic recognition first
                try:
                    text = self.recognizer.recognize_google(audio, language='ar-EG')
                    self.result_ready.emit(f"تم التعرف: {text}")
                except sr.UnknownValueError:
                    self.error_occurred.emit("لم يتم التعرف على الصوت")
                except sr.RequestError as e:
                    self.error_occurred.emit(f"خطأ في الخدمة: {str(e)}")
                    
        except Exception as e:
            self.error_occurred.emit(f"خطأ: {str(e)}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop voice recognition"""
        self.is_running = False


class VoiceIndicator(QWidget):
    """Visual indicator for voice recognition"""
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(100, 100, 150, 150)
        
        self.is_listening = False
        self.animation_value = 0
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate)
        
    def start_animation(self):
        """Start listening animation"""
        self.is_listening = True
        self.animation_timer.start(50)  # Update every 50ms
        self.show()
    
    def stop_animation(self):
        """Stop listening animation"""
        self.is_listening = False
        self.animation_timer.stop()
        self.hide()
    
    def animate(self):
        """Animate the indicator"""
        self.animation_value = (self.animation_value + 1) % 100
        self.update()
    
    def paintEvent(self, event):
        """Draw the voice indicator"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw pulsing circle
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # Pulsing effect
        radius = 40 + int(20 * abs(np.sin(self.animation_value * 0.1)))
        
        # Outer glow
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 0, 0, 50))
        painter.drawEllipse(center_x - radius - 10, center_y - radius - 10, 
                           (radius + 10) * 2, (radius + 10) * 2)
        
        # Main circle
        painter.setBrush(QColor(255, 0, 0, 200))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Microphone icon (simplified)
        painter.setPen(QPen(Qt.white, 3))
        painter.drawLine(center_x, center_y - 15, center_x, center_y + 15)
        painter.drawArc(center_x - 10, center_y - 15, 20, 25, 0, 180 * 16)


class ScreenCaptureWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color: rgba(0, 150, 255, 50);")
        
        self.mode = "resize"
        self.update_color()
        # Variables for resizing
        self.resizing = False
        self.resize_edge = None
        self.edge_margin = 10
        self.drag_position = None

        self.shortcut = QShortcut(QKeySequence("Ctrl+Space"), self)
        self.shortcut.activated.connect(self.toggle_mode)
    
    def toggle_mode(self):
        if self.mode == "resize":
            self.mode = "move"
        else:
            self.mode = "resize"
        self.update_color()
        self.update()

    def update_color(self):
        if self.mode == "resize":
            self.setStyleSheet("background-color: rgba(0, 150, 255, 50);")
            self.border_color = Qt.blue
        else:
            self.setStyleSheet("background-color: rgba(0, 255, 150, 50);")
            self.border_color = Qt.green
        
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(self.border_color, 3, Qt.SolidLine))
        painter.drawRect(0, 0, self.width()-1, self.height()-1)

        painter.setPen(QPen(Qt.white, 1))
        mode_text = "Resize Mode" if self.mode == "resize" else "Move Mode"
        painter.drawText(10, 20, mode_text)
        painter.drawText(10, 40, "Ctrl+Space to toggle mode")
        
    def get_resize_edge(self, pos):
        rect = self.rect()
        margin = self.edge_margin
        
        left = pos.x() < margin
        right = pos.x() > rect.width() - margin
        top = pos.y() < margin
        bottom = pos.y() > rect.height() - margin
        
        if bottom and right:
            return 'bottom-right'
        elif bottom and left:
            return 'bottom-left'
        elif top and right:
            return 'top-right'
        elif top and left:
            return 'top-left'
        elif bottom:
            return 'bottom'
        elif top:
            return 'top'
        elif right:
            return 'right'
        elif left:
            return 'left'
        return None
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.mode == "resize":
                self.resize_edge = self.get_resize_edge(event.pos())
                if self.resize_edge:
                    self.resizing = True
                    self.resize_start_pos = event.globalPos()
                    self.resize_start_geometry = self.geometry()
                else:
                    self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            else:
                self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            if self.mode == "resize":
                if self.resizing and self.resize_edge:
                    self.resize_window(event.globalPos())
                elif self.drag_position:
                    self.move(event.globalPos() - self.drag_position)
            else:
                if self.drag_position:
                    self.move(event.globalPos() - self.drag_position)
            event.accept()
        else:
            if self.mode == "resize":
                edge = self.get_resize_edge(event.pos())
                if edge in ['bottom-right', 'top-left']:
                    self.setCursor(Qt.SizeFDiagCursor)
                elif edge in ['bottom-left', 'top-right']:
                    self.setCursor(Qt.SizeBDiagCursor)
                elif edge in ['left', 'right']:
                    self.setCursor(Qt.SizeHorCursor)
                elif edge in ['top', 'bottom']:
                    self.setCursor(Qt.SizeVerCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
    
    def mouseReleaseEvent(self, event):
        self.resizing = False
        self.resize_edge = None
        self.drag_position = None
        event.accept()
    
    def resize_window(self, global_pos):
        delta = global_pos - self.resize_start_pos
        geo = self.resize_start_geometry
        
        x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()
        
        if 'right' in self.resize_edge:
            w = max(100, geo.width() + delta.x())
        if 'left' in self.resize_edge:
            new_w = max(100, geo.width() - delta.x())
            x = geo.x() + (geo.width() - new_w)
            w = new_w
        if 'bottom' in self.resize_edge:
            h = max(100, geo.height() + delta.y())
        if 'top' in self.resize_edge:
            new_h = max(100, geo.height() - delta.y())
            y = geo.y() + (geo.height() - new_h)
            h = new_h
        
        self.setGeometry(x, y, w, h)

class IconUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.7
        )
        try:
            model_path = get_model_path()
            self.model = load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
        self.arabic_letters = ['أ','ب','ت','ث','ج','ح','خ','د','ذ','ر','ز','س','ش','ص','ض','ط','ظ','ع','غ','ف','ق','ك','ل','م','ن','ه','و','ي','لا','ى','ة','ء']
        self.word_to_image = {
            "ا": "aleff.jpeg", "أ": "aleff.jpeg", "ب": "bb.jpeg", "ت": "ta.jpeg", 
            "ث": "thea.jpeg", "ج": "jeem.jpeg", "ح": "ha.jpeg", "خ": "khaa.jpeg", 
            "د": "daal.jpeg", "ذ": "thal.jpeg", "ر": "ra'.jpeg", "ز": "zay.jpeg", 
            "س": "seen.jpeg", "ش": "sheen.jpeg", "ص": "saad.jpeg", "ض": "daad.jpeg",
            "ط": "taah.jpeg", "ظ": "thaa.jpeg", "ع": "ain.jpeg", "غ": "ghain.jpeg", 
            "ف": "fa.jpeg", "ق": "k'af.jpeg", "ك": "kaaf.jpeg", "ل": "laam.jpeg", 
            "م": "meem.jpeg", "ن": "nun.jpeg", "ه": "haa'.jpeg", "و": "waaw.jpeg", 
            "ي": "yaa.jpeg", "لا": "la.jpeg", "ة": "ta marboota.jpeg", 
            "ال": "alif_lam.jpeg", "ء": "hamza.jpeg", "ى": "yaa.jpeg", " ": "space.jpeg"
        }
        
        # Path to sign images
        self.signs_path = r"D:\visual\Python\DEPI machine\Final Project\ARABIC"
        self.display_time = 1.0

        self.is_running = False
        self.capture_window = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        
        self.current_letter = None
        self.letter_start_time = None
        self.letter_hold_duration = 2.0  # seconds
        self.confidence_threshold = 0.90
        self.collected_word = ""
        self.last_added_letter = None

        self.voice_thread = None
        self.voice_indicator = VoiceIndicator()
        self.is_voice_mode = False
        self.sign_display = SignImageDisplay()
        self.sign_sequence_thread = None
        

        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Icon UI')
        
        # Make window frameless and stay on top
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        
        # Make window background transparent
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Get screen geometry and position at top-middle
        screen = QApplication.primaryScreen().geometry()
        window_width = 400
        window_height = 150
        x = (screen.width() - window_width) // 2
        y = 0
        self.setGeometry(x, y, window_width, window_height)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create icons container with horizontal layout
        icons_container = QWidget()
        icons_container.setStyleSheet("background-color: transparent;")
        icons_layout = QHBoxLayout(icons_container)
        icons_layout.setContentsMargins(20, 20, 20, 20)
        icons_layout.setSpacing(30)
        
        # Create three icon buttons
        self.create_icon_button(icons_layout, "Exit", "❌")
        self.create_icon_button(icons_layout, "Start", "▶️")
        self.create_icon_button(icons_layout, "Voice", "🎤")
        self.create_icon_button(icons_layout, "Show", "👁️")
        self.create_icon_button(icons_layout, "Clear", "🗑️")
        
        # Create caption label at bottom of screen
        screen = QApplication.primaryScreen().geometry()
        
        # Label for current letter and confidence (top label)
        self.letter_label = QLabel("جاهز للبدء")
        self.letter_label.setAlignment(Qt.AlignCenter)
        
        caption_width = screen.width()
        letter_height = 60
        letter_x = 0
        letter_y = screen.height() - 130  # 130px from bottom
        
        self.letter_label.setGeometry(letter_x, letter_y, caption_width, letter_height)
        self.letter_label.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.letter_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 100, 200, 255);
                color: white;
                padding: 10px;
                font-size: 40px;
                font-weight: bold;
                border-top: 2px solid white;
            }
        """)
        self.letter_label.show()
        
        # Label for collected word (bottom label)
        self.word_label = QLabel("")
        self.word_label.setAlignment(Qt.AlignCenter)
        
        word_height = 70
        word_x = 0
        word_y = screen.height() - word_height 
        
        self.word_label.setGeometry(word_x, word_y, caption_width, word_height)
        self.word_label.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.word_label.setStyleSheet("""
            QLabel {
                background-color: black;
                color: #00ff00;
                padding: 10px;
                font-size: 32px;
                font-weight: bold;
            }
        """)
        self.word_label.show()
        
        # Keep reference to caption_label for backward compatibility
        self.caption_label = self.letter_label

        # Add widgets to main layout
        main_layout.addWidget(icons_container, stretch=1)
        
        # Set window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: transparent;
            }
            QWidget {
                background-color: transparent;
            }
        """)
    
    def create_icon_button(self, layout, name, emoji):
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(5)
        container_layout.setAlignment(Qt.AlignCenter)
        
        button = QPushButton(emoji)
        button.setFixedSize(60, 60)
        button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 2px solid #ccc;
                border-radius: 30px;
                font-size: 24px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.3);
                border: 2px solid #999;
            }
        """)
        button.clicked.connect(lambda: self.button_clicked(name))
        

        
        # Add to container
        container_layout.addWidget(button, alignment=Qt.AlignCenter)
        
        # Add container to layout
        layout.addWidget(container)
    
    def button_clicked(self, name):
        print(f"{name} button clicked")
        if name == "Exit":
            self.stop_capture()
            self.close()
            if self.caption_label:
                self.caption_label.close()

            sys.exit()
        elif name == "Start":
            if self.is_running:
                self.stop_capture()
            else:
                self.start_capture()
        elif name == "Voice":
            self.stop_capture()
            self.start_voice_recognition()
        elif name == "Show":
            self.toggle_capture_window()
        elif name == "Clear":
            self.clear_word()
    
    def start_capture(self):
        if not self.is_running:
            if self.capture_window is None:
                self.capture_window = ScreenCaptureWindow()
                self.capture_window.show()
            
            self.is_running = True
            self.timer.start(30)  # 30ms = ~33 FPS
            self.update_caption("جاري التعرف...")
            self.capture_window.show()
            print("Capture started!")
    
    def stop_capture(self):
        if self.is_running:
            self.is_running = False
            self.timer.stop()
            self.update_caption("متوقف")
            self.capture_window.hide()
            print("Capture stopped!")
    
    def toggle_capture_window(self):
        if self.capture_window:
            if self.capture_window.isVisible():
                self.capture_window.hide()
            else:
                self.capture_window.show()
    
    def process_frame(self):
        if not self.capture_window or not self.model:
            return
        
        try:
            screen = QApplication.primaryScreen()
            geometry = self.capture_window.geometry()
            screenshot = screen.grabWindow(
                0,
                geometry.x(),
                geometry.y(),
                geometry.width(),
                geometry.height()
            ).toImage()
            
            width = screenshot.width()
            height = screenshot.height()
            ptr = screenshot.bits()
            ptr.setsize(height * width * 4)
            frame = np.array(ptr).reshape(height, width, 4)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                lm_list = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    lm_list.extend([lm.x, lm.y])
                
                landmarks = np.array(lm_list).reshape(1, 42, 1).astype(np.float32)
                
                pred = self.model.predict(landmarks, verbose=0)
                idx = np.argmax(pred)
                conf = pred[0][idx]
                
                predicted_letter = self.arabic_letters[idx]
                
                if conf >= self.confidence_threshold:
                    # Check if same letter as before
                    if predicted_letter == self.current_letter:
                        # Calculate hold duration
                        if self.letter_start_time is not None:
                            hold_time = time.time() - self.letter_start_time
                            
                            # Check if held long enough
                            if hold_time >= self.letter_hold_duration:
                                # Add letter to word if not already added
                                if predicted_letter != self.last_added_letter:
                                    self.add_letter_to_word(predicted_letter)
                                    self.last_added_letter = predicted_letter
                                
                                # Show confirmed letter
                                self.update_letter_display(f"{predicted_letter} ✓ ({conf:.0%})")
                                self.update_word_display(self.collected_word)
                            else:
                                # Show progress bar
                                progress = int((hold_time / self.letter_hold_duration) * 100)
                                self.update_letter_display(f"{predicted_letter} ({conf:.0%}) [{progress}%]")
                                self.update_word_display(self.collected_word)
                    else:
                        # New letter detected
                        self.current_letter = predicted_letter
                        self.letter_start_time = time.time()
                        self.update_letter_display(f"{predicted_letter} ({conf:.0%}) [0%]")
                        self.update_word_display(self.collected_word)
                else:
                    # Confidence too low, reset timer
                    self.reset_letter_timer()
                    if conf > 0.3:
                        self.update_letter_display(f"{predicted_letter} ({conf:.0%}) - ثقة منخفضة")
                        self.update_word_display(self.collected_word)
            else:
                self.update_caption("لا توجد يد")
                
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def update_caption(self, text):
        self.caption_label.setText(text)
    def update_letter_display(self, text):
        
        self.letter_label.setText(text)
    
    def update_word_display(self, word):
        """Update word label with Arabic reshaping"""
        try:
            if word:
                
                self.word_label.setText(f"الكلمة: {word}")
            else:
                self.word_label.setText("")
        except:
            self.word_label.setText(word if word else "")
    
    def reset_letter_timer(self):
        """Reset the letter hold timer"""
        self.current_letter = None
        self.letter_start_time = None
    
    def add_letter_to_word(self, letter):
        """Add a letter to the collected word"""
        self.collected_word += letter
        print(f"Letter added: {letter} | Word: {self.collected_word}")
    
    def clear_word(self):
        """Clear the collected word"""
        self.collected_word = ""
        self.last_added_letter = None
        self.reset_letter_timer()
        self.update_letter_display("تم المسح - جاهز للبدء")
        self.update_word_display("")
        print("Word cleared")

    def start_voice_recognition(self):
        """Start voice recognition mode"""
        if self.voice_thread and self.voice_thread.isRunning():
            print("Voice recognition already running")
            return
        
        self.is_voice_mode = True
        self.update_letter_display("🎤 اضغط على الزر مرة أخرى للبدء")
        self.update_word_display("")
        
        # Position voice indicator at center of screen
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - 150) // 2
        y = (screen.height() - 150) // 2
        self.voice_indicator.setGeometry(x, y, 150, 150)
        
        # Start voice recognition thread
        self.voice_thread = VoiceRecognitionThread()
        self.voice_thread.result_ready.connect(self.on_voice_result)
        self.voice_thread.error_occurred.connect(self.on_voice_error)
        self.voice_thread.start()
        
        # Show indicator
        self.voice_indicator.start_animation()
        self.update_letter_display("🎤 جاري الاستماع...")
        print("Voice recognition started")
    
    def stop_voice_recognition(self):
        """Stop voice recognition mode"""
        if self.voice_thread:
            self.voice_thread.stop()
            self.voice_thread = None
        
        self.voice_indicator.stop_animation()
        self.is_voice_mode = False
        print("Voice recognition stopped")
    
    def on_voice_result(self, text):
        """Handle voice recognition result"""
        if text.startswith("تم التعرف:"):
            # Extract the recognized text
            recognized_text = text.replace("تم التعرف:", "").strip()
            self.collected_word = recognized_text
            self.update_letter_display("✓ تم التعرف على الصوت")
            self.update_word_display(recognized_text)
            self.voice_indicator.stop_animation()
            
            # Start displaying sign sequence
            self.start_sign_sequence(recognized_text)
        else:
            # Status update
            self.update_letter_display(text)
    
    def on_voice_error(self, error):
        """Handle voice recognition error"""
        self.update_letter_display(f"❌ {error}")
        self.voice_indicator.stop_animation()
    
    def start_sign_sequence(self, text):
        """Start displaying sign language images for the text"""
        # Stop any existing sequence
        if self.sign_sequence_thread and self.sign_sequence_thread.isRunning():
            self.sign_sequence_thread.stop()
            self.sign_sequence_thread.wait()
        
        # Start new sequence
        self.sign_sequence_thread = SignSequenceThread(
            text, 
            self.word_to_image, 
            self.signs_path, 
            self.display_time
        )
        self.sign_sequence_thread.image_ready.connect(self.show_sign_image)
        self.sign_sequence_thread.sequence_complete.connect(self.on_sequence_complete)
        self.sign_sequence_thread.start()
        
    
    def show_sign_image(self, image_path, letter):
        """Display a sign image"""
        self.sign_display.show_image(image_path, letter)
    
    def on_sequence_complete(self):
        """Handle sequence completion"""
        print("Sign sequence completed")
        # Keep the last image visible for a moment, then hide
        QTimer.singleShot(1000, self.sign_display.clear_image)


def main():
    app = QApplication(sys.argv)
    window = IconUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
