import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
# torch.cuda.set_device(0)
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
python detect.py --weights weights/best.pt --img 640 --conf 0.25 --source data/video
