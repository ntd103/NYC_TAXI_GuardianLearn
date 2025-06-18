# NYC Taxi Duration Predictor
This project is a complete machine learning application that covers the entire data product lifecycle: from processing and analyzing raw data with Apache Spark, building and optimizing a linear regression model with PySpark MLlib, to deploying the model as an interactive web application using Streamlit. The entire development environment is containerized and managed by Docker and VS Code Dev Containers, ensuring consistency and reproducibility.

The primary goal is not only to create an accurate prediction model but also to build a complete product where end-users can directly interact with the model through a user-friendly graphical interface.

## Key Features
- Big Data Processing: Utilizes Apache Spark to efficiently process and clean millions of taxi trip records.
- Machine Learning Pipeline: Constructs an automated and consistent ML pipeline, including feature engineering, normalization, and model training steps.
- Model Optimization: Applies techniques like Cross-Validation and Regularization to find the best-performing model.
- Comprehensive Evaluation: Assesses model performance using RMSE, MAE, and R² metrics, along with residual analysis.
- Interactive User Interface: A web application built with Streamlit that allows users to input trip details and receive predictions instantly.
- Containerized Environment: The entire project is containerized using Docker, making setup and deployment simple across any system.

## Tech Stack
- Language: Python 3.9+
- Docker Images: pyspark-notebook:spark-3.5.0
- Machine Learning: PySpark MLlib
- Web Interface: Streamlit
- Containerization: Docker, Docker Compose
- Development Environment: VS Code, WSL2, Dev Containers

## Quick Start
### Step 1: Clone this repo

```
git clone https://github.com/ntd103/NYC_TAXI_GuardianLearn.git
cd NYC_TAXI_GuardianLearn
```

### Step 2: Data Setup
Due to GitHub file size limits, please download the dataset separately:

1. Download data from: [NYC Taxi Trip Duration Competition](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)
2. Place files in:
   - `data/raw/train.csv`
   - `data/raw/test.csv`

### Step 3: Open in container. Make sure you have installed Docker.
 
#### Option A: Using VS Code + Dev Containers (Recommended)
This method provides the most integrated development experience.
1. Open the project folder in Visual Studio Code.
2. A notification will appear in the bottom-right corner. Click the "Reopen in Container" button.
3. VS Code will automatically build and launch the Docker environment. This process might take a few minutes on the first run.

#### Option B: Using Docker Compose Directly (Universal Method)
This method works from any terminal, without needing the VS Code Dev Containers extension.
1. Open a terminal (PowerShell, Command Prompt, or a Linux terminal).
2. Navigate to the project's root directory.
3. Run the following command to build and start the services in the background:
```
docker-compose up -d
```

This will start the Spark/Jupyter container.


## How to Use
Once the environment is running, you can interact with it as follows.

### 1. Analyze and Train the Model (in JupyterLab)
If you used Option A (VS Code): The Jupyter server is already running. You can directly open and run the notebooks in the notebooks directory within VS Code.
If you used Option B (Docker Compose):
1. Check the logs to get the JupyterLab access URL with its token:
```
docker-compose logs
```
2. Look for a line similar to http://127.0.0.1:8888/lab?token=....
3. Copy this URL and paste it into your web browser.

Inside JupyterLab: Navigate to the work directory (which is mapped to your local notebooks folder). Open `01-Data-Exploration.ipynb` and `02-Feature-Engineering-and-Modeling.ipynb` to run the analysis and training steps. The best model will be saved to the `/models` directory. (`03-notebook` file is the version that is being further developed, you can try it)

### 2. Run the Prediction Application
The Streamlit application runs inside the same container. You need to execute the run command from within the container's terminal.
Open a terminal inside the container:
- If using VS Code: Simply open a new Terminal tab in VS Code. It will automatically connect to the container.
- If using Docker Compose directly: Open a new local terminal and run:
```
docker exec -it spark-lab-vs-code /bin/bash
```
(Note: spark-lab-vs-code is the container_name defined in docker-compose.yml)
Inside the container's terminal, launch the Streamlit app:
```
streamlit run /home/jovyan/app/app.py
```
Access the app: Open a web browser and navigate to `http://localhost:8501` to view and interact with the application.

## Project Structure
The project is organized with a clear structure to separate different components:
```
NYC_TAXI_GuardianLearn/
│
├── .devcontainer/              # Configuration for VS Code Dev Containers
│   └── devcontainer.json
│
├── app/                        # Source code for the Streamlit application
│   ├── app.py
│   └── requirements.txt
│
├── data/
│   ├── raw/                    # Raw data from Kaggle (.csv)
│   └── processed/              # (Optional) Processed data
│
├── models/                     # Trained and saved PipelineModels
│
├── notebooks/                  # Jupyter Notebooks for analysis and experimentation
│   ├── 01-Data-Exploration.ipynb
│   └── 02-Feature-Engineering-and-Modeling.ipynb
│
├── docker-compose.yml          # Defines and orchestrates the containers
└── README.md                   # This guide
```

## Future Improvements
...loading

---
Dzung Nguyen
