import cv2
import urllib.request
import numpy as np
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python wifi_viewer.py <IP_ADDRESS>")
        print("Example: python wifi_viewer.py 192.168.1.50")
        sys.exit(1)

    ip = sys.argv[1]
    url = f"http://{ip}/"

    print(f"Connecting to camera stream at {url}...")
    
    try:
        stream = urllib.request.urlopen(url)
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    print("Connected! Press ESC to quit.")
    
    bytes_data = b''
    while True:
        try:
            bytes_data += stream.read(4096)
            a = bytes_data.find(b'\xff\xd8') # JPEG start marker
            b = bytes_data.find(b'\xff\xd9') # JPEG end marker
            
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                # Decode and display
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imshow("ESP32-S3 Wi-Fi Camera Stream", img)
                    if cv2.waitKey(1) == 27: # ESC key
                        break
        except Exception as e:
            print(f"Stream interrupted: {e}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
