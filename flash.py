import os
import subprocess
import sys
import shutil

def main():
    sketch_dir = "esp32_camera"
    sketch_file = os.path.join(sketch_dir, "esp32_camera.ino")
    board_fqbn = "esp32:esp32:esp32s3"

    print("Checking for arduino-cli...")
    if not shutil.which("arduino-cli"):
        print("\n=======================================================")
        print("Error: arduino-cli is not installed or not in your PATH.")
        print("Since you are using the Arduino IDE, the easiest way to flash this is manually:")
        print(f"1. Open the file {sketch_file} in the Arduino IDE.")
        print("2. Select your ESP32-S3 board (e.g., 'ESP32S3 Dev Module').")
        print("3. Ensure 'PSRAM' is enabled in the Tools menu (crucial for the camera to work).")
        print("4. Select the correct COM port.")
        print("5. Click the 'Upload' button.")
        print("=======================================================\n")
        sys.exit(1)

    print("arduino-cli found! Compiling and flashing...")

    # We might need to install the core if it's not present, but for simplicity
    # we'll assume the user has ESP32 core installed in arduino-cli as well.
    # Usually users who use Arduino IDE will have to use the IDE if they haven't synced their CLI configs.
    
    # Prompt for port
    port = input("Enter the COM port of your ESP32-S3 (e.g. COM3): ").strip()
    if not port:
        print("Invalid port.")
        sys.exit(1)

    try:
        # Compile
        print("Compiling sketch...")
        subprocess.run(["arduino-cli", "compile", "--fqbn", board_fqbn, sketch_dir], check=True)
        
        # Upload
        print(f"Uploading to {port}...")
        subprocess.run(["arduino-cli", "upload", "-p", port, "--fqbn", board_fqbn, sketch_dir], check=True)
        print("Upload successful!")
        print("\nOpen the Serial Monitor (115200 baud) in Arduino IDE to see the camera IP address.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to compile or upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
