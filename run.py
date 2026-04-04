
import os

replacements = {
    "ram": "ram",
    "RAM": "RAM",
    "Ram": "Ram"
}

def process_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        with open(filepath, "w") as f:
            f.write(new_content)
        print(f"Updated {filepath}")

for root, dirs, files in os.walk("./test"):
    if ".venv" in root or "__pycache__" in root:
        continue
    for file in files:
        if file.endswith((".py", ".jsx", ".md", ".json", ".yaml", ".txt")):
            process_file(os.path.join(root, file))
