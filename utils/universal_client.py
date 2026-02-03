import sys
import zmq
import json

from PyQt5 import QtWidgets, QtGui, QtCore


sockets = [
    ("cam_server", 6667),
    ("DM_server", 6666),
    ("heimdallr", 6660),
    ("baldr1", 6662),
    ("baldr2", 6663),
    ("baldr3", 6664),
    ("baldr4", 6665),
]


class HistoryLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.history_index = -1

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Up:
            if self.history:
                if self.history_index == -1:
                    self.history_index = len(self.history) - 1
                elif self.history_index > 0:
                    self.history_index -= 1
                self.setText(self.history[self.history_index])
                self.setCursorPosition(len(self.text()))
        elif event.key() == QtCore.Qt.Key_Down:
            if self.history:
                if (
                    self.history_index != -1
                    and self.history_index < len(self.history) - 1
                ):
                    self.history_index += 1
                    self.setText(self.history[self.history_index])
                    self.setCursorPosition(len(self.text()))
                else:
                    self.history_index = -1
                    self.clear()
        elif event.key() == QtCore.Qt.Key_Tab:
            # Autocomplete command from dropdown if possible
            parent = self.parent()
            if hasattr(parent, "command_dropdown"):
                current_text = self.text()
                dropdown = parent.command_dropdown
                matches = []
                for i in range(dropdown.count()):
                    cmd = dropdown.itemText(i)
                    if cmd.startswith(current_text):
                        matches.append(cmd)
                if matches:
                    # If only one match, autocomplete fully
                    # If multiple, autocomplete to common prefix
                    if len(matches) == 1:
                        self.setText(matches[0])
                    else:
                        # Find common prefix
                        prefix = matches[0]
                        for m in matches[1:]:
                            i = 0
                            while i < len(prefix) and i < len(m) and prefix[i] == m[i]:
                                i += 1
                            prefix = prefix[:i]
                        self.setText(prefix)
                    self.setCursorPosition(len(self.text()))
                    # Optionally, show dropdown popup if multiple matches
                    if len(matches) > 1:
                        dropdown.showPopup()
                else:
                    # No match, do nothing or beep
                    # QtWidgets.QApplication.beep()
                    pass
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def add_history(self, cmd):
        if cmd and (not self.history or self.history[-1] != cmd):
            self.history.append(cmd)
        self.history_index = -1


class ServerTab(QtWidgets.QWidget):
    def __init__(self, server_name, zmq_socket, parent=None):
        super().__init__(parent)
        self.server_name = server_name
        self.zmq_socket = zmq_socket
        self.parent_window = parent  # Not used, but could be for future

        layout = QtWidgets.QVBoxLayout(self)
        self.command_dropdown = QtWidgets.QComboBox(self)
        self.command_dropdown.setEditable(False)
        self.command_dropdown.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.input_line = HistoryLineEdit(self)
        self.send_button = QtWidgets.QPushButton("Send", self)
        self.text_area = QtWidgets.QTextEdit(self)
        self.text_area.setReadOnly(True)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.command_dropdown)
        hlayout.addWidget(self.input_line)
        hlayout.addWidget(self.send_button)

        layout.addLayout(hlayout)
        layout.addWidget(self.text_area)

        self.send_button.clicked.connect(self.send_command)
        self.input_line.returnPressed.connect(self.send_command)
        self.command_dropdown.activated[str].connect(self.set_input_line)

    def set_input_line(self, cmd):
        self.input_line.setText(cmd)
        # Fetch and display arguments for the selected command
        self.display_command_arguments(cmd)

    def display_command_arguments(self, cmd):
        # Send 'arguments "<cmd>"' to the server and display the result as a table
        try:
            self.zmq_socket.send_string(f'arguments "{cmd}"')
            reply = self.zmq_socket.recv_string()
        except zmq.error.Again:
            self.text_area.append(
                "[Error] ZMQ request timed out while fetching arguments."
            )
            return
        except zmq.error.ZMQError as e:
            if "Operation cannot be accomplished in current state" in str(e):
                self.text_area.append(
                    "[Warning] Socket out of state, attempting to reconnect for arguments..."
                )
                self.reconnect_socket()
                try:
                    self.zmq_socket.send_string(f'arguments "{cmd}"')
                    reply = self.zmq_socket.recv_string()
                except Exception:
                    self.text_area.append(
                        "[Error] Could not fetch arguments after reconnect."
                    )
                    return
            else:
                self.text_area.append(f"[Error] {e}")
                return
        except Exception as e:
            self.text_area.append(f"[Error] {e}")
            return

        # Try to parse the reply and display as a table
        try:
            args = json.loads(reply)
            if args is None:
                self.text_area.append("No arguments required.")
            elif isinstance(args, list) and args and isinstance(args[0], dict):
                # Display as a table
                table = "<table border='1' cellspacing='0' cellpadding='2'><tr><th>Name</th><th>Type</th></tr>"
                for arg in args:
                    name = arg.get("name", "")
                    typ = arg.get("type", "")
                    table += f"<tr><td>{name}</td><td>{typ}</td></tr>"
                table += "</table>"
                self.text_area.append(table)
            else:
                self.text_area.append(str(args))
        except Exception:
            # Fallback: just show the reply
            self.text_area.append(reply.replace("\\n", "\n"))

    def reconnect_socket(self):
        # Recreate the socket and reconnect
        try:
            context = self.zmq_socket.context
            # Get the endpoint from the socket's last connection
            last_endpoint = self.zmq_socket.getsockopt(zmq.LAST_ENDPOINT).decode()
            self.zmq_socket.close(linger=0)
            new_socket = context.socket(zmq.REQ)
            new_socket.setsockopt(zmq.SNDTIMEO, 2000)
            new_socket.setsockopt(zmq.RCVTIMEO, 2000)
            new_socket.connect(last_endpoint)
            self.zmq_socket = new_socket
        except Exception as e:
            self.text_area.append(f"[Error] Could not reconnect socket: {e}")

    def send_command(self):
        cmd = self.input_line.text().strip()
        if not cmd:
            return
        self.input_line.add_history(cmd)
        self.text_area.append(f"> {cmd}")
        try:
            self.zmq_socket.send_string(cmd)
            reply = self.zmq_socket.recv_string()
        except zmq.error.Again:
            reply = "[Error] ZMQ request timed out."
        except zmq.error.ZMQError as e:
            if "Operation cannot be accomplished in current state" in str(e):
                self.text_area.append(
                    "[Warning] Socket out of state, attempting to reconnect..."
                )
                self.reconnect_socket()
                try:
                    self.zmq_socket.send_string(cmd)
                    reply = self.zmq_socket.recv_string()
                except Exception as e2:
                    reply = f"[Error] After reconnect: {e2}"
            else:
                reply = f"[Error] {e}"
        except Exception as e:
            reply = f"[Error] {e}"
        # Try to parse as JSON and pretty-print, else just show with \n expanded
        try:
            resp = json.loads(reply)
            if isinstance(resp, dict):
                pretty = json.dumps(resp, indent=4)
                self.text_area.append(pretty)
            else:
                self.text_area.append(str(resp).replace("\\n", "\n"))
        except json.JSONDecodeError:
            self.text_area.append(reply.replace("\\n", "\n"))
        self.input_line.clear()

    def populate_commands(self):
        # Send "command_names" and populate the dropdown
        self.command_dropdown.clear()
        try:
            self.zmq_socket.send_string("command_names")
            reply = self.zmq_socket.recv_string()
        except zmq.error.Again:
            self.text_area.append(
                "[Error] ZMQ request timed out while fetching commands."
            )
            return
        except zmq.error.ZMQError as e:
            if "Operation cannot be accomplished in current state" in str(e):
                self.text_area.append(
                    "[Warning] Socket out of state, attempting to reconnect for commands..."
                )
                self.reconnect_socket()
                try:
                    self.zmq_socket.send_string("command_names")
                    reply = self.zmq_socket.recv_string()
                except Exception:
                    return
            else:
                return
        except Exception:
            return
        try:
            commands = json.loads(reply)
            if isinstance(commands, list):
                self.command_dropdown.addItems(commands)
        except Exception:
            try:
                commands = eval(reply)
                if isinstance(commands, list):
                    self.command_dropdown.addItems([str(c) for c in commands])
            except Exception:
                pass


class UniversalClient(QtWidgets.QMainWindow):
    def __init__(self, ip_addr, servers, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Asgard DCS text interface: {ip_addr}")
        self.resize(700, 500)
        self.tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabs)
        self.context = zmq.Context()
        self.sockets = []
        self.tab_widgets = []

        for name, port in servers:
            socket = self.context.socket(zmq.REQ)
            socket.connect(f"tcp://{ip_addr}:{port}")
            # Set ZMQ send/recv timeouts (milliseconds)
            socket.setsockopt(zmq.SNDTIMEO, 2000)
            socket.setsockopt(zmq.RCVTIMEO, 2000)
            self.sockets.append(socket)
            tab = ServerTab(name, socket)
            self.tabs.addTab(tab, name)
            self.tab_widgets.append(tab)

        self.tabs.currentChanged.connect(self.on_tab_changed)
        # Populate commands for the first tab
        if self.tab_widgets:
            self.tab_widgets[0].populate_commands()

    def on_tab_changed(self, idx):
        if 0 <= idx < len(self.tab_widgets):
            self.tab_widgets[idx].populate_commands()


def main():
    if len(sys.argv) != 2:
        print("Usage: python universal_client.py <ip_address>")
        sys.exit(1)
    ip_addr = sys.argv[1]
    servers = sockets
    app = QtWidgets.QApplication(sys.argv)
    client = UniversalClient(ip_addr, servers)
    client.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()
