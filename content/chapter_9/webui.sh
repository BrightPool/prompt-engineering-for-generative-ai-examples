#!/bin/bash

# Ensure the virtual environment is present, or create one
if [ ! -d "test_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv test_env
fi

# Activate the virtual environment
source test_env/bin/activate

# Check for the existence of requirements.txt
if [ -f "requirements.txt" ]; then
    echo "requirements.txt found. Installing necessary libraries..."
    pip3 install -r requirements.txt
else
    echo "requirements.txt not found. Installing default libraries..."
    pip3 install gradio langchain python-dotenv
    echo "Exporting installed libraries to requirements.txt..."
    pip3 freeze > requirements.txt
fi

playwright install

# Run the Gradio app
echo "Launching interface"
python3 gradio_code_example.py

# Deactivate the virtual environment
deactivate

echo "Server closed."