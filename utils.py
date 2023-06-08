import torch
from pathlib import Path

def save_checkpoint(model: torch.nn.Module,
                    target_dir: str,
                    model_name: str):
    
    """
    Saves a PyTorch model checkpoint.

    Args:
        model: a PyTorch model.
        target_dir: the directory where the model will be saved.
        model_name: the name of the model.
    """

    target_dir_path = Path(target_dir)

    if not target_dir_path.is_dir():
        target_dir_path.mkdir(parents=True,exist_ok=True)

    else:
        print(f"Model directory {target_dir_path} already exists.")

    assert model_name.endswith(".pth") or model_name.endswith('.pt'), "Model name must end with .pth"

    model_path = target_dir_path / model_name

    print(f"Saving model checkpoint to {model_path} ...")

    torch.save(model.state_dict(),model_path)

    