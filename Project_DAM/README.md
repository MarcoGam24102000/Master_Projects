
# Cross-Platform Application Development Final Project

## Overview

This project showcases the development of a cross-platform application utilizing three ESP32 microcontrollers, multiple sensors, and actuators. The system is designed to handle external event activation and integrates with multiple HTML pages using JavaScript.

## Project Structure

- **first_esp32.ino**: Code for the first ESP32 microcontroller.
- **sec_esp32.ino**: Code for the second ESP32 microcontroller.
- **third_esp32.ino**: Code for the third ESP32 microcontroller.
- **write_controls.ino**: Controls and communication code.

## Features

1. **Multi-Device Integration**: Utilizes three ESP32 microcontrollers to manage various sensors and actuators.
2. **External Event Activation**: Capable of responding to external events to trigger specific actions.
3. **Web Integration**: Includes multiple HTML pages with JavaScript for real-time interaction and control.
4. **Sensor and Actuator Management**: Handles data from various sensors and actuators to perform desired functions.

## Setup Instructions

### Hardware Requirements

- 3 x ESP32 Microcontrollers
- Sensors:
  - DHT11 Temperature and Humidity Sensor
  - PIR Motion Sensor
  - MQ-2 Gas Sensor
- Actuators:
  - Relay Module
  - Servo Motor
  - Buzzer

### Software Requirements

- Arduino IDE
- Required libraries for ESP32 and connected sensors/actuators
- Web server setup (optional for serving HTML pages)

### Installation

1. **Clone the Repository**: 
   ```sh
   git clone [repository_link]
   ```

2. **Upload the Code**:
   - Open each `.ino` file in the Arduino IDE.
   - Select the correct board and port for your ESP32.
   - Upload the code to the respective ESP32 devices.

3. **Configure Web Server** (Optional):
   - Place the HTML files on your web server.
   - Ensure the ESP32 devices are connected to the same network as the web server.

### Usage

1. **Power On**: Power on the ESP32 devices.
2. **Access Web Interface**: Open your web browser and navigate to the configured web server to access the control interface.
3. **Monitor and Control**: Use the web interface to monitor sensor data and control actuators.

## Code Details

### `first_esp32.ino`

This file contains the code for the first ESP32 microcontroller. It is responsible for:
- Initializing the DHT11 Temperature and Humidity Sensor and the PIR Motion Sensor.
- Reading data from these sensors.
- Sending the sensor data to the main controller or web interface for monitoring and control.

### `sec_esp32.ino`

This file contains the code for the second ESP32 microcontroller. It is responsible for:
- Handling the MQ-2 Gas Sensor.
- Communicating with the first ESP32 and the third ESP32 for coordinated actions.
- Activating the relay module based on the sensor data and commands received.

### `third_esp32.ino`

This file contains the code for the third ESP32 microcontroller. It is responsible for:
- Managing the servo motor and the buzzer.
- Ensuring the overall system stability by acting as a backup controller.
- Coordinating with the other two ESP32 microcontrollers for seamless operations.

### `write_controls.ino`

This file contains the communication and control logic for the entire system. It includes:
- Functions to send and receive data between the ESP32 microcontrollers.
- Code to handle external event activation.
- Logic to integrate with the HTML pages and JavaScript for web-based control.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

