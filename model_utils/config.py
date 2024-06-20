import os
os.environ["KERAS_BACKEND"] = "torch" # "torch"
from scipy import signal
SEED = 42
stft_size = 512
window_size = 400
window_shift = 160
sampling_rate = 16000
windows = signal.windows.hann
max_audio_length = 40800
max_video_length = 64
video_frame_size = (88, 88)