from merlin_utility import *
from PIL import Image
from scipy.io import wavfile
import argparse
import numpy as np
import PIL.ImageOps

window_size = 512
hop_size = 128

#
# Parse command line arguments
#
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Merlin photo id")
    parser.add_argument("--model_path", help="Path to the .tflite model", required=True)
    parser.add_argument("--sound_path", help="Path to the sound file", required=True)
    parser.add_argument("--output_path", help="Path to save the spectrogram image", required=True)
    parser.add_argument("--invert", help="Invert the spectrogram image", action='store_true')
    parser.add_argument("--max_hops", help="Maximum number of hops to process", type=int, default=512)
    args = parser.parse_args()
    return args

#
# Get audio data from the sound file. It's expected to be 22050 Hz.
# To generate a spectrogram for sound id, max_hops should be set to 512.
#
def get_wav_samples(sound_path, max_hops):
    sample_rate, samples = wavfile.read(sound_path)
    samples = samples.astype(np.float32) / 32768.0

    print("Sample_rate:", sample_rate) # 22050
    print("Samples shape:", samples.shape) # array of 66150 samples for 3 second clip

    len = max_hops * hop_size + (window_size - hop_size)
    return samples[:len]

#
# Gets the next window of samples
#
def get_next_sample(samples):
    for start in range(0, len(samples) - window_size + 1, hop_size):
        yield samples[start:start + window_size]

#
# Generates the spectrogram for the supplied audio samples.
# The supplied model is used to generate the frequency data for each window.
#
def generate_spectrogram(model_path, samples):
    interpreter = create_tflite_interpreter(model_path)

    data = []
    for sample in get_next_sample(samples):
        fsample = np.array(sample).astype(np.float32)
        run_model(interpreter, [fsample])
        result = get_single_output(interpreter)
        data.append(result)

    spectrogram = np.hstack(data)  # shape (N, 128) where N == number of iterations
    return spectrogram

#
# Save the spectrogram image to the output path
#
def save_spectrogram_image(spectrogram, output_path, invert):
    img = Image.fromarray(spectrogram)
    if invert:
        img = PIL.ImageOps.invert(img)
    img.save(output_path)

args = parse_arguments()
samples = get_wav_samples(args.sound_path, args.max_hops)
result = generate_spectrogram(args.model_path, samples)
save_spectrogram_image(result, args.output_path, args.invert)
