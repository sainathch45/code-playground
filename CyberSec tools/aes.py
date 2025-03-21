import sys
import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QMessageBox

class AESTool(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("AES Encryptor & Decryptor")
        self.setGeometry(100, 100, 500, 400)

        layout = QVBoxLayout()

        self.key_label = QLabel("Enter Encryption Key:")
        layout.addWidget(self.key_label)
        self.key_input = QLineEdit(self)
        layout.addWidget(self.key_input)

        self.plain_text_label = QLabel("Enter Text to Encrypt:")
        layout.addWidget(self.plain_text_label)
        self.plain_text_input = QTextEdit(self)
        layout.addWidget(self.plain_text_input)

        self.encrypt_button = QPushButton("Encrypt", self)
        self.encrypt_button.clicked.connect(self.encrypt_text)
        layout.addWidget(self.encrypt_button)

        self.encrypted_label = QLabel("Encrypted Text:")
        layout.addWidget(self.encrypted_label)
        self.encrypted_output = QTextEdit(self)
        self.encrypted_output.setReadOnly(True)
        layout.addWidget(self.encrypted_output)

        self.decrypt_button = QPushButton("Decrypt", self)
        self.decrypt_button.clicked.connect(self.decrypt_text)
        layout.addWidget(self.decrypt_button)

        self.decrypted_label = QLabel("Decrypted Text:")
        layout.addWidget(self.decrypted_label)
        self.decrypted_output = QTextEdit(self)
        self.decrypted_output.setReadOnly(True)
        layout.addWidget(self.decrypted_output)

        self.setLayout(layout)

    def encrypt_text(self):
        key = self.key_input.text().strip()
        plain_text = self.plain_text_input.toPlainText().strip()

        if not key or not plain_text:
            QMessageBox.warning(self, "Error", "Key and Text cannot be empty!")
            return

        try:
            encrypted_text = self.aes_encrypt(key, plain_text)
            self.encrypted_output.setText(encrypted_text)
        except Exception as e:
            QMessageBox.warning(self, "Encryption Error", str(e))

    def decrypt_text(self):
        key = self.key_input.text().strip()
        encrypted_text = self.encrypted_output.toPlainText().strip()

        if not key or not encrypted_text:
            QMessageBox.warning(self, "Error", "Key and Encrypted Text cannot be empty!")
            return

        try:
            decrypted_text = self.aes_decrypt(key, encrypted_text)
            self.decrypted_output.setText(decrypted_text)
        except Exception as e:
            QMessageBox.warning(self, "Decryption Error", str(e))

    def aes_encrypt(self, key, plain_text):
        key = hashlib.sha256(key.encode()).digest()  # Ensure key is 32 bytes
        cipher = AES.new(key, AES.MODE_CBC)  # Generate a new AES cipher
        iv = cipher.iv
        encrypted_bytes = cipher.encrypt(pad(plain_text.encode(), AES.block_size))  # Encrypt text
        return base64.b64encode(iv + encrypted_bytes).decode()  # Encode with Base64

    def aes_decrypt(self, key, encrypted_text):
        key = hashlib.sha256(key.encode()).digest()  # Ensure key is 32 bytes
        encrypted_bytes = base64.b64decode(encrypted_text)  # Decode Base64
        iv = encrypted_bytes[:AES.block_size]  # Extract IV
        cipher = AES.new(key, AES.MODE_CBC, iv)  # Create cipher with IV
        decrypted_bytes = unpad(cipher.decrypt(encrypted_bytes[AES.block_size:]), AES.block_size)  # Decrypt
        return decrypted_bytes.decode()  # Convert bytes to string

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AESTool()
    window.show()
    sys.exit(app.exec())
