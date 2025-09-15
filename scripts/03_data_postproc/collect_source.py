from pathlib import Path
import shutil

input_dir = Path("data/output")
output_dir = Path("data/output/source")
output_dir.mkdir(exist_ok=True)

for dir in input_dir.glob("*"):
    if dir.is_dir() and dir.name != "source":
        for file in dir.glob("*.xlsx"):
            new_name = output_dir / f"{dir.name}.xlsx"
            shutil.copy(file, new_name)