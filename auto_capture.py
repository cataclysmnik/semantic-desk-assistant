import urllib.request
import time
import os

# 1. Replace this with the actual IP address of your ESP32
ESP32_IP = "192.168.137.28" 
URL = f"http://{ESP32_IP}/capture"

# 2. This is the folder where images will be saved
SAVE_FOLDER = r"C:\Users\sagni\Downloads\esp32_captures"

def main():
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
        print(f"Created folder: {SAVE_FOLDER}")

    print(f"Starting to capture images every 5 seconds from {URL}")
    print(f"Saving to: {SAVE_FOLDER}")
    print("Press Ctrl+C in this terminal to stop.")

    try:
        while True:
            # Create a filename with the current date and time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(SAVE_FOLDER, f"capture_{timestamp}.jpg")
            
            try:
                # Download and save the image
                urllib.request.urlretrieve(URL, filename)
                print(f"[{time.strftime('%H:%M:%S')}] Saved: {filename}")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Error fetching image. Is the ESP32 on and is the IP correct? Error: {e}")
            
            # Wait 5 seconds before the next capture
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nStopped capturing.")

if __name__ == "__main__":
    main()
