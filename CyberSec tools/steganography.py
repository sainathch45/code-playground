import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QTextEdit
from PyQt6.QtGui import QPixmap
import os

class SteganographyTool(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Image Steganography Tool")
        self.setGeometry(100, 100, 500, 400)
        
        self.image_label = QLabel("Upload an Image")
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter text to hide...")
        
        self.encode_button = QPushButton("Hide Text in Image")
        self.encode_button.clicked.connect(self.encode_text)
        
        self.decode_button = QPushButton("Extract Hidden Text")
        self.decode_button.clicked.connect(self.decode_text)
        
        self.result_label = QLabel("")
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.encode_button)
        layout.addWidget(self.decode_button)
        layout.addWidget(self.result_label)
        
        self.setLayout(layout)
        self.image_path = ""
    
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.bmp)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(250, 250))
    
    def encode_text(self):
        if not self.image_path:
            self.result_label.setText("Please upload an image first!")
            return
        
        text = self.text_edit.toPlainText()
        if not text:
            self.result_label.setText("Enter text to hide!")
            return
        
        image = cv2.imread(self.image_path)
        binary_text = ''.join(format(ord(c), '08b') for c in text) + '1111111111111110'  # End marker
        data_index = 0
        binary_len = len(binary_text)

        for row in image:
            for pixel in row:
                for channel in range(3):  # Iterate over R, G, B
                    if data_index < binary_len:
                        pixel[channel] = (pixel[channel] & ~1) | int(binary_text[data_index])
                        data_index += 1
                    else:
                        break
            if data_index >= binary_len:
                break
        
        save_path = os.path.join(os.path.dirname(self.image_path), "encoded_image.png")
        cv2.imwrite(save_path, image)
        self.result_label.setText(f"Image saved with hidden text: {save_path}")
    
    def decode_text(self):
        if not self.image_path:
            self.result_label.setText("Please upload an image first!")
            return
        
        image = cv2.imread(self.image_path)
        binary_text = ""
        
        for row in image:
            for pixel in row:
                for channel in range(3):
                    binary_text += str(pixel[channel] & 1)

        # Split binary into 8-bit chunks and look for the stopping marker
        bytes_text = [binary_text[i:i+8] for i in range(0, len(binary_text), 8)]
        extracted_text = ""
        
        for byte in bytes_text:
            if byte == '11111111':  # Detect stop marker
                break
            extracted_text += chr(int(byte, 2))
        
        if extracted_text:
            self.result_label.setText(f"Extracted Text: {extracted_text}")
        else:
            self.result_label.setText("No hidden text found!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SteganographyTool()
    window.show()
    sys.exit(app.exec())
