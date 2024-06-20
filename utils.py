import numpy as np


def pad_audio(audio, length):
    if len(audio) < length:
        audio = np.pad(audio, (0, length - len(audio)))
    return audio


def pad_video(video, length):
    if len(video) < length:
        video = np.pad(video, ((0, length - len(video)), (0, 0), (0, 0), (0, 0)))
    return video


def get_enhanced(model, data):
    enhanced_audio = np.zeros(len(data["noisy_audio"]))
    for i in range(0, len(data["noisy_audio"]), 40800):
        video_idx = (i // 40800) * 64
        noisy_audio = data["noisy_audio"][i:i + 40800]
        inputs = dict(noisy_audio=pad_audio(noisy_audio, 40800)[np.newaxis, ...],
                      video_frames=pad_video(data["video_frames"][video_idx:video_idx + 64], 64)[np.newaxis, ...])
        estimated_audio = model.predict(inputs, verbose=0)[0, :]
        if len(enhanced_audio) < 40800:
            return estimated_audio[:len(enhanced_audio)]
        if len(noisy_audio) < 40800:
            enhanced_audio[i:i + len(noisy_audio)] = estimated_audio[:len(noisy_audio)]
        else:
            enhanced_audio[i:i + 40800] = estimated_audio
    return enhanced_audio
