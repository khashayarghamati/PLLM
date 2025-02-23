# Personalised Large Language Model (PLLM) for HRI and HCI

## Overview
This repository provides the implementation of a **Personalised Large Language Model (PLLM)** framework for integrating and personalising large language model (LLM) agents in **Human-Robot Interaction (HRI)** and **Human-Computer Interaction (HCI)** domains. 

The project utilises:
- **OpenAI API** for LLM capabilities.
- **Streamlit** for an interactive chatbot interface.
- **NeuroSense EEG dataset** and the **Muse 2 device** for personalisation.
- **Python** for development and implementation.

This document guides you through setting up and running the project.

## Prerequisites
Ensure you have the following installed on your system:

- **Python (>=3.8)**: Download and install from [python.org](https://www.python.org/downloads/)
- **pip** (Python package manager): Ensure it is installed using:
  ```bash
  python -m ensurepip --default-pip
  ```
- **OpenAI API Key**: Get your API key by creating an account on [OpenAI's platform](https://platform.openai.com/)
- **Muse 2 Device** (if using EEG data collection)
- **Virtual Environment (Recommended)**:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate
  ```

## Installation
Clone the repository and install the required dependencies:

```bash
git clone <repository_url>
cd <repository_name>
pip install -r requirements.txt
```

Alternatively, you can install the dependencies manually:
```bash
pip install openai streamlit pandas numpy scipy matplotlib seaborn
pip install mne  # For EEG data processing
```

## Setting Up OpenAI API Key
To use OpenAI’s LLM, set up your API key:

1. **Method 1: Environment Variable** (Recommended)
   ```bash
   export OPENAI_API_KEY='your-api-key-here'  # For Linux/macOS
   set OPENAI_API_KEY='your-api-key-here'     # For Windows
   ```

2. **Method 2: Store in a `.env` file**
   - Create a `.env` file in the project directory:
     ```bash
     echo "OPENAI_API_KEY='your-api-key-here'" > .env
     ```

3. **Method 3: Enter via UI**
   - When running the chatbot, you can manually enter your API key.

## Running the Application

### 1. Start the Chatbot
Run the Streamlit chatbot interface:
```bash
streamlit run app.py
```
- This opens a web-based chatbot where you can input text, upload PDFs, CSV files, and interact with the LLM.

### 2. Personalising the LLM with EEG Data
To process EEG data from **NeuroSense dataset**:
1. Upload EEG files (`.edf`, `.json`, `.tsv`) using the Streamlit interface.
2. The LLM processes and personalises responses based on the EEG data.
3. The robot can then interpret and respond to emotions based on real-time EEG input.

### 3. Collecting EEG Data from Muse 2 Device
- Ensure your **Muse 2 device** is connected.
- Run the script to collect EEG data:
  ```bash
  python muse_data_collector.py
  ```
- The collected EEG data will be sent to the **PLLM** model for analysis.

## Directory Structure
```
|-- project_root/
    |-- app.py              # Streamlit Chatbot Interface
    |-- muse_data_collector.py  # Script for collecting EEG data
    |-- utils/
        |-- data_processing.py  # Functions for processing EEG data
        |-- openai_interface.py # OpenAI API interactions
    |-- data/
        |-- example_eeg.edf  # Example EEG dataset
    |-- requirements.txt  # Dependencies
    |-- README.md  # This file
```

## Troubleshooting
- **Issue: "Command not found" when running `streamlit run app.py`**
  - Ensure Streamlit is installed (`pip install streamlit`).
  - If using a virtual environment, activate it first (`source venv/bin/activate`).

- **Issue: "Invalid API key" error**
  - Ensure the OpenAI API key is correctly set in environment variables or entered in the UI.

- **Issue: "No EEG device detected"**
  - Ensure the **Muse 2 device** is properly connected via Bluetooth.
  - Restart the connection and retry.

## License
This project is licensed under the **Creative Commons Attribution (CC BY) License**. 

## Source Code Availability
The source code will be published once the paper is officially published.

## Authors
- **Khashayar Ghamati** (University of Hertfordshire)
- **Abolfazl Zaraki**
- **Maryam Banitalebi Dehkordi**

For questions or collaborations, please contact **k.ghamati@herts.ac.uk**.

---
**Enjoy building personalised AI applications with PLLM! 🚀**

