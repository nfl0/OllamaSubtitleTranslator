# Step 1: Create the virtual environment
python3 -m venv env

# Step 2: Activate it
source env/bin/activate


pip install langchain langchain_community pydantic tqdm
pip install -U langchain-ollama


usage:
python translate.py sample_dutch.srt -i Dutch -o English -m "deepseek-r1:7b"

