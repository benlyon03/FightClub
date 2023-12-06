
import sys
sys.path.append('mmAction/mmaction2/')
from mmaction.apis import init_recognizer, inference_recognizer
import os
import sklearn.metrics as metrics

# Load the newly trained model checkpoint
new_checkpoint = 'epoch_10.pth'
cfg = 'config.py'

# Initialize the recognizer with the new checkpoint
new_model = init_recognizer(cfg, new_checkpoint, device='cuda:0')

# Specify the label map
video = "nofight.mp4"
results = inference_recognizer(new_model, video)

print(results)