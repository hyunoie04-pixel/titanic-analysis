import shutil
from pathlib import Path

import kagglehub

path = kagglehub.dataset_download("heptapod/titanic")
print("Path to dataset files:", path)

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

for file in Path(path).iterdir():
    if file.is_file():
        shutil.copy2(file, data_dir / file.name)
        print(f"Copied: {file.name} -> {data_dir / file.name}")
