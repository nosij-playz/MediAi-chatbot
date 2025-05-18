from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import subprocess
import sys
import threading

app = Flask(__name__)
socketio = SocketIO(app)
available = False  # Flag to indicate if prescription is available
process = None  # Holds the chatbot process

def start_MediAi_script(force_restart=False):
    """Starts MediAi.py as a subprocess, optionally forcing a restart."""
    global process,available
    
    if process and process.poll() is None:
        if force_restart:  # If restart is requested, terminate first
            print("Restarting chatbot process...")
            process.terminate()
            process.wait()
        else:
            print("Chatbot is already running.")
            return  # Don't restart if it's already running

    process = subprocess.Popen(
        [sys.executable, "MediAi.py"],  # Use the same Python as Flask
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    threading.Thread(target=read_output, daemon=True).start()

def read_output():
    """Continuously reads output from MediAi.py and sends it to the frontend."""
    global process, available
    while process.poll() is None:  # While MediAi.py is running
        output = process.stdout.readline()
        if output:
            print(f"Output from MediAi.py: {output.strip()}")
            socketio.emit("output", {"message": output.strip()})

        # Example condition to toggle prescription availability
        if "Diagnosis completed sucessfully" in output:
            available = True
            socketio.emit("toggle_button", {"show": True})

    print("MediAi.py has terminated.")

@app.route("/")
def home():
    """Render the home page with a button to start chat."""
    return render_template("home.html")

@app.route("/chat")
def chat():
    """Starts chatbot process and renders the chat page."""
    print("Starting chatbot process...")
    start_MediAi_script()  # Start the chatbot when the button is clicked
    return render_template("index.html")

@socketio.on("user_input")
def handle_input(data):
    """Handles user input from frontend and sends it to MediAi.py."""
    global process
    user_text = data.get("message", "")
    if user_text and process:
        print(f"Sending to MediAi.py: {user_text}")
        process.stdin.write(user_text + "\n")
        process.stdin.flush()

@socketio.on("new_chat")
def restart_chat():
    """Handles the new chat button press by forcefully restarting MediAi.py and clearing chat history."""
    global available
    print("Restarting chat and MediAi.py...")
    available = False
    start_MediAi_script(force_restart=True)  # Forcefully restart MediAi.py
    socketio.emit("clear_chat")  # Notify frontend to clear chat history
    socketio.emit("toggle_button", {"show": False})  # Hide prescription box

@socketio.on("print")
def print_prescription(data):
    """Handles prescription viewing request."""
    global available
    if available==True:
        name = data.get("name")
        age = data.get("age")
        phone = data.get("phone")
        
        print(f"Generating prescription for {name}, Age: {age}, Phone: {phone}")# Generate prescription
        
        socketio.emit("prescription_ready")  # Notify frontend that prescription is ready
        
if __name__ == "__main__":
    socketio.run(app, debug=True)
