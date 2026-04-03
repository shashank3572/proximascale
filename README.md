# AI SysAdmin: Decision Engine & Actuator
This part of the project handles the scaling of Docker containers based on CPU predictions.
- **Decision Engine**: Logic for scaling thresholds.
- **Actuator**: Talks to Docker SDK to start/stop containers.
- **Receiver**: API to accept predictions from the LSTM model.