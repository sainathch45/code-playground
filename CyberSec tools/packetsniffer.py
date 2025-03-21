import sys
import scapy.all as scapy
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
                             QTableWidget, QTableWidgetItem, QHeaderView, QComboBox)
from PyQt6.QtCore import QThread, pyqtSignal
import time

class PacketSniffer(QThread):
    packet_captured = pyqtSignal(object, object)

    def __init__(self, interface="Ethernet"):
        super().__init__()
        self.interface = interface
        self.running = False

    def run(self):
        self.running = True
        scapy.sniff(iface=self.interface, prn=self.process_packet, store=False)

    def process_packet(self, packet):
        if not self.running:
            return
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        src_ip = packet[scapy.IP].src if packet.haslayer(scapy.IP) else "Unknown"
        dst_ip = packet[scapy.IP].dst if packet.haslayer(scapy.IP) else "Unknown"
        protocol = "UDP" if packet.haslayer(scapy.UDP) else ("TCP" if packet.haslayer(scapy.TCP) else "Other")
        payload = str(packet.payload)

        self.packet_captured.emit((timestamp, src_ip, dst_ip, protocol, payload), packet)

    def stop(self):
        self.running = False
        self.quit()

class PacketSnifferGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Packet Sniffer - CyberSec Tool")
        self.setGeometry(100, 100, 800, 500)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("Press 'Start Capture' to begin sniffing packets")
        layout.addWidget(self.label)

        self.start_button = QPushButton("Start Capture")
        self.start_button.clicked.connect(self.start_sniffing)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Capture")
        self.stop_button.clicked.connect(self.stop_sniffing)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        self.protocol_filter = QComboBox()
        self.protocol_filter.addItem("All")
        self.protocol_filter.currentIndexChanged.connect(self.apply_filter)
        layout.addWidget(self.protocol_filter)

        self.packet_table = QTableWidget()
        self.packet_table.setColumnCount(5)
        self.packet_table.setHorizontalHeaderLabels(["Timestamp", "Source IP", "Destination IP", "Protocol", "Payload"])
        self.packet_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.packet_table)

        self.setLayout(layout)

        self.sniffer = None
        self.captured_packets = []  # Store all packets for filtering
        self.protocols_seen = set()

    def start_sniffing(self):
        self.packet_table.setRowCount(0)  # Clear table before new capture
        self.captured_packets = []
        self.protocols_seen.clear()
        self.protocol_filter.clear()
        self.protocol_filter.addItem("All")
        
        self.sniffer = PacketSniffer()
        self.sniffer.packet_captured.connect(self.store_packet)
        self.sniffer.start()

        self.label.setText("Capturing packets...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_sniffing(self):
        if self.sniffer:
            self.sniffer.stop()
            self.sniffer = None

        self.label.setText("Capture stopped.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def store_packet(self, packet_info, packet):
        self.captured_packets.append(packet_info)
        protocol = packet_info[3]
        
        if protocol not in self.protocols_seen:
            self.protocols_seen.add(protocol)
            self.protocol_filter.addItem(protocol)
        
        self.apply_filter()

    def apply_filter(self):
        selected_protocol = self.protocol_filter.currentText()
        self.packet_table.setRowCount(0)  # Clear table

        for packet_info in self.captured_packets:
            if selected_protocol == "All" or packet_info[3] == selected_protocol:
                self.add_packet_to_table(packet_info)

    def add_packet_to_table(self, packet_info):
        row = self.packet_table.rowCount()
        self.packet_table.insertRow(row)

        for col, data in enumerate(packet_info):
            self.packet_table.setItem(row, col, QTableWidgetItem(str(data)))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PacketSnifferGUI()
    window.show()
    sys.exit(app.exec())
