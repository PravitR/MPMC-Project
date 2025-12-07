import serial
import time
from PIL import Image, ImageOps, UnidentifiedImageError
import sys

# Configuration
i = 45 # Index of sent image
SERIAL_PORT = 'COM11'  # Check your Device Manager!
BAUD_RATE = 115200
IMG_PATH = f'testSet\img_{i}.jpg' 
IMG_SIZE = 20          # The model expects 20x20 input

# Image Processing
try:
    print(f"Opening Image {i}...")
    img = Image.open(IMG_PATH)
    
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.Resampling.LANCZOS)
    
    # Prepare the byte array
    image_bytes = bytearray(IMG_SIZE * IMG_SIZE)
    pixels = list(img.getdata())
    for i in range(len(pixels)):
        # We send the raw pixel value (0-255).
        image_bytes[i] = int(pixels[i])
    
except FileNotFoundError:
    print(f"--- ERROR: File '{IMG_PATH}' not found. ---")
    sys.exit(1)
except Exception as e:
    print(f"--- ERROR during image processing: {e} ---")
    sys.exit(1)

# --- 3. Serial Communication (Slow Transmission) ---
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print(f"Opened serial port {ser.name}")
    time.sleep(2) # Give board time to initialize after connection

    print(f"Sending {len(image_bytes)} bytes slowly...")

    # Send the data one byte at a time with a small delay.
    # This prevents the hardware crash we solved earlier.
    for byte_to_send in image_bytes:
        ser.write(bytes([byte_to_send])) 
        time.sleep(0.002) # Wait for 2 milliseconds

    print("--------------------------")
    print("Data sent successfully!")
    print("--------------------------")

except serial.SerialException as e:
    print(f"--- SERIAL ERROR: {e} ---")
    sys.exit(1)
finally:
    if ser and ser.is_open:
        ser.close()
        print(f"Closed serial port {ser.name}")