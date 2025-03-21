import sys
import socket
import threading
import time
import subprocess
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout, QFrame
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
import ipaddress

class ScannerThread(QThread):
    scan_complete = pyqtSignal(list, float)

    def __init__(self, start_ip, end_ip):
        super().__init__()
        self.start_ip = start_ip
        self.end_ip = end_ip

    def ping_ip(self, ip):
        try:
            result = subprocess.run(["ping", "-n", "1", "-w", "500", ip], capture_output=True, text=True, shell=True)
            return "Reply from" in result.stdout
        except Exception:
            return False

    def scan_ip(self, ip, results):
        live = self.ping_ip(ip)
        active_ports = []
        if live:
            try:
                socket.setdefaulttimeout(1)
                for port in [22, 23, 53, 80, 443, 8080]:  # Common ports
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    result = s.connect_ex((ip, port))
                    if result == 0:
                        active_ports.append(port)
                    s.close()
            except:
                pass
        status = "Live" if live else "Dead"
        results.append((ip, status, ", ".join(map(str, active_ports)) if active_ports else "None"))

    def run(self):
        results = []
        start_ip_int = int(ipaddress.IPv4Address(self.start_ip))
        end_ip_int = int(ipaddress.IPv4Address(self.end_ip))
        threads = []
        
        start_time = time.time()
        
        for ip_int in range(start_ip_int, end_ip_int + 1):
            ip_str = str(ipaddress.IPv4Address(ip_int))
            thread = threading.Thread(target=self.scan_ip, args=(ip_str, results))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        self.scan_complete.emit(results, time_taken)

class IPScannerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('IP Scanner')
        self.resize(700, 500)
        
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        layout = QVBoxLayout()
        
        header = QLabel("Network IP Scanner")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        form_layout = QHBoxLayout()
        
        self.start_ip = QLineEdit(self)
        self.start_ip.setPlaceholderText("Start IP Address")
        self.end_ip = QLineEdit(self)
        self.end_ip.setPlaceholderText("End IP Address")
        
        form_layout.addWidget(self.start_ip)
        form_layout.addWidget(self.end_ip)
        layout.addLayout(form_layout)
        
        self.scan_button = QPushButton('Scan Network', self)
        self.scan_button.setStyleSheet("background-color: #0078D7; color: white; font-size: 14px; padding: 8px;")
        self.scan_button.clicked.connect(self.run_scan)
        layout.addWidget(self.scan_button)
        
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["IP Address", "Status", "Active Ports"])
        self.result_table.setStyleSheet("background-color: #1E1E1E; color: #00FF00; font-family: Consolas;")
        layout.addWidget(self.result_table)
        
        self.time_label = QLabel('Time taken: 0.00 seconds')
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)
        
        self.setLayout(layout)

    def run_scan(self):
        start_ip = self.start_ip.text().strip()
        end_ip = self.end_ip.text().strip()
        if start_ip and end_ip:
            self.result_table.setRowCount(0)
            self.time_label.setText("Scanning...")
            self.scanner_thread = ScannerThread(start_ip, end_ip)
            self.scanner_thread.scan_complete.connect(self.update_results)
            self.scanner_thread.start()

    def update_results(self, results, time_taken):
        self.result_table.setRowCount(len(results))
        for row, (ip, status, ports) in enumerate(results):
            self.result_table.setItem(row, 0, QTableWidgetItem(ip))
            self.result_table.setItem(row, 1, QTableWidgetItem(status))
            self.result_table.setItem(row, 2, QTableWidgetItem(ports))
        self.time_label.setText(f"Time taken: {time_taken:.2f} seconds")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    scanner = IPScannerApp()
    scanner.show()
    sys.exit(app.exec())
