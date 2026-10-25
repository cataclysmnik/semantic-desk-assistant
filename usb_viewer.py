import serial
import cv2
import numpy as np
import sys
import time

def main():
    if len(sys.argv) < 2:
        print("Usage: python usb_viewer.py <COM_PORT>")
        print("Example: python usb_viewer.py COM3")
        sys.exit(1)

    port = sys.argv[1]
    baud = 115200

    print(f"Connecting to {port} at {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
        # ESP32 auto-reset circuit: need to clear DTR and RTS to let it run
        ser.dtr = False
        ser.rts = False
    except Exception as e:
        print(f"Error opening serial port: {e}")
        sys.exit(1)

    print("Connected! Waiting for video stream... (Press ESC in the video window to quit)")

    debug_buffer = bytearray()
    while True:
        # 1. Find magic word "FRAME:"
        magic = b""
        while magic != b"FRAME:":
            c = ser.read(1)
            if not c:
                continue
            magic += c
            
            # Print debug characters if it's not part of the magic word
            if len(magic) == 1 and magic != b"F":
                if 32 <= c[0] <= 126 or c[0] in (10, 13):
                    print(chr(c[0]), end="", flush=True)
            
            if len(magic) > 6:
                magic = magic[1:]
        
        # 2. Read length (4 bytes)
        len_bytes = ser.read(4)
        if len(len_bytes) != 4:
            continue
        
        length = int.from_bytes(len_bytes, byteorder='little')
        
        if length == 0 or length > 500000: # sanity check
            continue
            
        # 3. Read JPEG data
        data = ser.read(length)
        if len(data) != length:
            print(f"Warning: expected {length} bytes, got {len(data)}")
            continue
            
        # 4. Decode and show image
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is not None:
            cv2.imshow("ESP32-S3 Camera USB Stream", img)
            if cv2.waitKey(1) == 27: # ESC key
                break
        else:
            print("Failed to decode frame")

    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
