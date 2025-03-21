import os
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QLineEdit, QMessageBox

class AESFileEncryptorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("AES File Encryptor & Decryptor")
        self.setGeometry(100, 100, 400, 200)
        
        layout = QVBoxLayout()

        self.label = QLabel("Select a file and enter encryption key:")
        layout.addWidget(self.label)

        self.file_button = QPushButton("Choose File")
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Enter encryption key")
        layout.addWidget(self.key_input)

        self.encrypt_button = QPushButton("Encrypt File")
        self.encrypt_button.clicked.connect(self.encrypt_file)
        layout.addWidget(self.encrypt_button)

        self.decrypt_button = QPushButton("Decrypt File")
        self.decrypt_button.clicked.connect(self.decrypt_file)
        layout.addWidget(self.decrypt_button)

        self.setLayout(layout)
        self.file_path = None

    def select_file(self):
        file_dialog = QFileDialog()
        file_name, _ = file_dialog.getOpenFileName(self, "Select File")
        if file_name:
            self.file_path = file_name
            self.label.setText(f"Selected: {os.path.basename(file_name)}")

    def get_key(self):
        key = self.key_input.text().strip()
        if len(key) == 0:
            QMessageBox.warning(self, "Error", "Encryption key cannot be empty!")
            return None
        return hashlib.sha256(key.encode()).digest()  # Convert key to 256-bit AES key

    def encrypt_file(self):
        if not self.file_path:
            QMessageBox.warning(self, "Error", "No file selected!")
            return
        
        key = self.get_key()
        if key is None:
            return
        
        output_file = self.file_path + ".aes"
        iv = os.urandom(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)

        with open(self.file_path, 'rb') as f:
            plaintext = f.read()

        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

        with open(output_file, 'wb') as f:
            f.write(iv + ciphertext)

        QMessageBox.information(self, "Success", f"File encrypted: {output_file}")

    def decrypt_file(self):
        if not self.file_path:
            QMessageBox.warning(self, "Error", "No file selected!")
            return
        
        key = self.get_key()
        if key is None:
            return
        
        if not self.file_path.endswith(".aes"):
            QMessageBox.warning(self, "Error", "Please select an encrypted (.aes) file!")
            return
        
        output_file = self.file_path.replace(".aes", "_decrypted")

        with open(self.file_path, 'rb') as f:
            file_data = f.read()

        iv = file_data[:16]
        ciphertext = file_data[16:]

        cipher = AES.new(key, AES.MODE_CBC, iv)
        try:
            plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid encryption key!")
            return

        with open(output_file, 'wb') as f:
            f.write(plaintext)

        QMessageBox.information(self, "Success", f"File decrypted: {output_file}")

if __name__ == "__main__":
    app = QApplication([])
    window = AESFileEncryptorGUI()
    window.show()
    app.exec()
