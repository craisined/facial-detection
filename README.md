# Pairwise Facial Recognition

Facial recognition model with 81% accuracy on LFW

## Running Instructions

### Dependencies
```sh
pip install scikit-learn torch torchvision tqdm
```

### Training
```sh
sh setup.sh
python3 model.py
```

### Inference
```python3
from model import FacialRecognitionModel
import torch
model = FacialRecognitionModel()
model.load_state_dict(torch.load("model.pth", weights_only=True))
print(model.predict(face1, face2)) # face1, face2 are PIL images
```

## Tech

### Architecture
- Resnet 18 esque backbone producing a 128 dimension embedding
- Cosine embedding loss

### Data
- Funneled LFW (with crop) for validation
- CASIA Webface for training