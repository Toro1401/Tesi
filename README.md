# Manufacturing System Simulation

## 1. Project Overview

This project is a discrete-event simulation of a manufacturing system, developed as part of a Master's thesis. The simulation is built using Python with several key libraries:

*   **SimPy**: For the core discrete-event simulation framework.
*   **NumPy**: For numerical operations, particularly for handling arrays and mathematical functions.
*   **Pandas**: For data manipulation and analysis, especially for managing simulation results.

The primary goal of this simulation is to model and analyze the performance of a manufacturing environment under various conditions, including machine downtime and worker absenteeism. It simulates the flow of jobs through a system of machines, managed by workers, and subject to daily operational changes.

### Key Features:

*   **Job and Order Management**: Simulates the creation, release, and processing of manufacturing jobs and orders.
*   **Resource Modeling**: Models machines and workers as resources with specific capacities and states (e.g., available, busy, down).
*   **Dynamic Events**: Incorporates dynamic events such as machine breakdowns and worker absenteeism to reflect real-world manufacturing challenges.
*   **Data Collection and Analysis**: Gathers detailed statistics on system performance, such as throughput, cycle time, and resource utilization.
*   **Configurable Parameters**: Allows for easy configuration of simulation parameters to test different scenarios and strategies.

## 2. Project Setup

To run this simulation, you will need a Python environment with the necessary libraries installed.

### Prerequisites:

*   Python 3.x
*   `pip` (Python package installer)

### Installation:

1.  **Clone the repository (if applicable):**
    If this project is in a version control system like Git, clone it to your local machine.

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required libraries:**
    The simulation depends on `SimPy`, `NumPy`, and `Pandas`. You can install these using `pip`:

    ```bash
    pip install simpy numpy pandas
    ```

## 3. Running the Simulation

The main simulation logic is contained in the `Thesis Code_v4.py` file. To run the simulation, execute this file using the Python interpreter.

### How to Run:

1.  **Navigate to the project directory:**
    Open your terminal or command prompt and navigate to the directory where `Thesis Code_v4.py` is located.

2.  **Execute the script:**
    Run the following command:

    ```bash
    python "Thesis Code_v4.py"
    ```

### Simulation Output:

The simulation will run for a predefined number of replications and scenarios. As it runs, it will print status updates and debugging information to the console.

Upon completion, the simulation generates several output files in a `data_output` directory (which will be created if it doesn't exist). These files contain the detailed results of the simulation runs, typically in CSV format, for further analysis.

## 4. Code Structure

The `Thesis Code_v4.py` file is organized into several classes and functions that work together to create the simulation environment:

*   **`Job` and `Jobs_generator`**: Manage the creation and lifecycle of individual jobs.
*   **`Orders_release`**: Controls the release of jobs into the system based on workload.
*   **`Pool`, `Machine`, `Worker`**: Represent the core resources of the manufacturing system.
*   **`DailyPlanningAbsenteeismManager`, `MachineDowntimeManager`**: Handle the dynamic events of worker absenteeism and machine breakdowns.
*   **`System`**: The main class that orchestrates the entire simulation, bringing together all the components.
*   **Helper and Analysis Functions**: A set of functions for calculating workloads, resetting statistics, and analyzing simulation results.

For more detailed information on the implementation of each component, refer to the Google-style docstrings within the `Thesis Code_v4.py` file.