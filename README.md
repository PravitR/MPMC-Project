# MPMC-Project
Course project for the course Microprocessor and Microcontrollers 


Python Transmitter + RISC-V Inference Engine

This project implements an end-to-end system where a Python Transmitter running on a PC sends processed data (e.g., images) to a RISC-V microcontroller, which performs lightweight ML inference and returns the result.
It demonstrates PC–MCU communication, UART protocols, and tinyML execution on embedded hardware.

⸻

Project Structure

/Python Transmitter/
    - Image preprocessing scripts
    - Data encoding & framing
    - UART transmission utilities

/VSD_Inference_Engine/
    - RISC-V firmware (C code)
    - UART receiver & packet parser
    - TinyML inference engine
    - Peripheral drivers & model files


⸻

How It Works
	1.	Python loads and preprocesses input data.
	2.	Data is encoded into byte frames and sent via UART.
	3.	The RISC-V board receives the stream, reconstructs inputs, and runs inference.
	4.	Output classification is printed or sent back to the PC.

⸻

Running the Project

PC (Python Transmitter)

python transmitter.py

Make sure to set the correct COM port in the script.

RISC-V Board
	•	Build the firmware in VSD_Inference_Engine/
	•	Flash to the board
	•	Open a serial monitor to view results

⸻

Features
	•	Real-time PC → RISC-V data transfer
	•	Lightweight embedded ML inference
	•	Demonstrates UART communication & MPMC concepts
	•	Clean separation of preprocessing (PC) and inference (MCU)

⸻

Purpose

Designed for academic use, especially MPMC projects and embedded ML demonstrations.

Project Demonstration : https://drive.google.com/drive/folders/1hvJBuOZEwjUZoKEcDB3VPNmnQRz9k7QE?usp=sharing
