# Webots Leonard Aerospace Demonstrator

This repository contains a Webots simulation for a pedestrian detection system using a Crazyflie drone. The simulation is containerized using Docker for easy setup and execution.

The core of the simulation involves a Crazyflie drone navigating a world with pedestrians. The drone uses a YOLO-based model for pedestrian detection, which is running as a service.

The YOLO inference service is implemented in Python and is included as a submodule in this repository. The inference service is responsible for processing images captured by the drone's camera and providing pedestrian detection results to the drone's controller. It can be done locally using the onnxruntime library or remotely using a FPGA-based accelerator.

### Submodules

This repository includes the following submodule:

*   `services/tristan-yolo-py-inference`: The YOLO inference service for pedestrian detection.

## Getting Started

### Prerequisites

*   Git
*   Docker
*   Docker Compose

### Cloning the Repository

To clone this repository, including the necessary submodule, run the following command:

```bash
git clone --recurse-submodules https://github.com/luigifeola/webots-leonard-aerospace-demonstrator.git
cd webots-leonard-aerospace-demonstrator
```

If you have already cloned the repository without the `--recurse-submodules` flag, you can initialize and update the submodule with the following commands:

```bash
git submodule update --init --recursive
```

## Running the Simulation

There are two ways to run the simulation: using Docker (recommended) or by installing everything manually.

### Using Docker (Recommended)

The simulation is designed to be run within a Docker container.

1.  Navigate to the `docker` directory:
    ```bash
    cd docker
    ```
2.  An `.env` file may be required in the `docker` directory to configure environment variables for the Docker Compose setup. Ensure this file is present and correctly configured before proceeding.
3.  Launch the simulation using Docker Compose:
    ```bash
    docker compose up
    ```

This command will build the Docker image if it doesn't exist and then start the Webots simulation. The simulation world `pedestrian.wbt` will be loaded, and the Crazyflie controller will be activated.

To run the simulation using the FPGA-based accelerator for YOLO inference, set the `CF_RUN_ON_ONNX_RUNTIME` environment variable to `False` in the [`.env`](./docker/.env) file before running the Docker Compose command.

### Manual Installation and Execution

If you prefer to run the simulation without Docker, you will need to install Webots and the required Python dependencies manually.

#### Installing Webots (linux)

1. Download and install the `.deb` from the official website: [Webots-R2023b](https://github.com/cyberbotics/webots/releases/tag/R2023b)
2. Install the `webots` package:
    ```bash
    sudo apt install ./webots_*.deb
    ```

#### Installing Python Dependencies

1.  Create a Python virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install the Python dependencies for the controller:
    ```bash
    pip install -r requirements.txt
    ```
3.  Install the Python dependencies for the YOLO inference service:
    ```bash
    pip install -r services/tristan-yolo-py-inference/requirements.txt
    ```

#### Running the Simulation

1.  Set the correct python settings in Webots for the drone controller: open a terminal and type `Webots` in order to open the Webots interface. Select `Tools` > `Preferences` > `Python command` and set the field to use the virtual environment's Python interpreter (e.g., `$HOME/venv/bin/python`).
2.  Launch the simulation with the following command:
    ```bash
    webots worlds/pedestrian.wbt
    ```
4.  The YOLO inference service will be started automatically by the drone controller.
