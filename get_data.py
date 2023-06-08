from pathlib import Path
import requests
import zipfile
import os

data_dir = Path("data")

image_path = data_dir / "pizza_steak_sushi"

if image_path.is_dir():
    print(f"Image directory {image_path} already exists.")

else:
    image_path.mkdir(parents=True,exist_ok=True)
    print(f"Created image directory at {image_path}.")

    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"

    response = requests.get(url)

    with open(data_dir / "pizza_steak_sushi.zip","wb") as f:
        print("Downloading pizza_steak_sushi.zip ...")
        f.write(response.content)

    with zipfile.ZipFile(data_dir / "pizza_steak_sushi.zip","r") as zip_ref:
        print("Unzipping pizza_steak_sushi.zip into image path ...")
        zip_ref.extractall(image_path)

    

    os.remove(data_dir / "pizza_steak_sushi.zip")


