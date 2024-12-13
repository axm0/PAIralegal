import os
import nltk

def setup_nltk():
    project_nltk = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(project_nltk, exist_ok=True)
    nltk.data.path.insert(0, project_nltk)
    try:
        nltk.data.find('tokenizers/punkt')
        print("'punkt' resource found.")
    except LookupError:
        print(f"Downloading 'punkt' to: {project_nltk}")
        nltk.download('punkt', download_dir=project_nltk)
    print(f"Attempting to download 'punkt_tab' to: {project_nltk}")
    nltk.download('punkt_tab', download_dir=project_nltk)
    print("\nNLTK Setup Complete!")
    print(f"Data directory: {project_nltk}")

if __name__ == "__main__":
    setup_nltk()
