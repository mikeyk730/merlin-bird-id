from PIL import Image
import csv
import tensorflow.lite
import numpy as np

#
# Create an interpreter for the .tflite model
#
def create_tflite_interpreter(model_path):
    interpreter = tensorflow.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

#
# Run the model with the supplied inputs
#
def run_model(interpreter, inputs):
    for i, input_data in enumerate(inputs):
        interpreter.set_tensor(i, input_data)
    interpreter.invoke()

#
# Get the output from a single-output model
#
def get_single_output(interpreter):
    output_details = interpreter.get_output_details()
    output_details = output_details[0]
    out = interpreter.get_tensor(output_details['index'])

    # dequantize if quantized (scale, zero_point) provided
    scale, zero_point = output_details.get('quantization', (0.0, 0))
    if scale:
        out = (out - zero_point) * scale
    return out

#
# Load labels for the .tflite model
#
def load_labels_for_model(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        labels_list = [line.strip() for line in f if line.strip()]

    return labels_list

#
# Load a mapping from species code to species name.
#
# e.g., 'amerob' -> 'American Robin'
#
# This information comes from the Clements Checklist, https://www.birds.cornell.edu/clementschecklist/
#
def load_species_id_map():
    path = r"metadata\Clements_v2025-October-2025.csv"
    bird_names = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            code = row[1].strip()
            name = row[6].strip() if len(row) > 1 else ""
            if code:
                bird_names[code] = name
    bird_names['bird1'] = 'Bird'

    return bird_names

#
# Load an image to pass into photo id and sound id models
#
def load_image_data(image_path, debug=False):

    with Image.open(image_path) as img:
        img_rgb = img.convert('RGB')

    # The 'rgb_array' is a 3D NumPy array of shape (height, width, 3)
    # where the last dimension represents the R, G, and B values for each pixel
    rgb_array = np.array(img_rgb).astype(np.float32)

    # Add the batch dimension, so the shape is (1, height, width, 3)
    model_input = np.expand_dims(rgb_array, axis=0)

    return model_input

#
# Print the results for the supplied indexes
#
def print_results_impl(out, idxs, labels, species):
    for rank, i in enumerate(idxs, start=1):
        val = out[i] * 100.0
        code = labels[i]
        name = species[code] if code in species else ''
        print(f"{rank}. {val:.2f}% [{i} {code}] - {name}")

#
# Print top N results
#
def print_top_results(out, labels, species, top_n):
    top_idxs = np.argsort(out)[::-1][:top_n]
    print_results_impl(out, top_idxs, labels, species)

#
# Print results above the supplied threshold
#
def print_matches(out, labels, species, threshold):
    idxs = np.where(out > threshold)[0]
    if idxs.size == 0:
        print(f"No results above threshold {threshold}")
    else:
        sorted_idxs = idxs[np.argsort(out[idxs])[::-1]]
        print_results_impl(out, sorted_idxs, labels, species)
