import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QShortcut)
from PyQt5.QtCore import Qt, QRect, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QRegion, QPainter, QPen, QKeySequence

import time

import sys
import os

def get_model_path():
    """Get the correct path for the model file"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = sys._MEIPASS
    else:
        # Running as script
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, 'Arabic_Sign_Language_CNN_Final.h5')

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
def main():
    app = QApplication(sys.argv)
    window = IconUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
