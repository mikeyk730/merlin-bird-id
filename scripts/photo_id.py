from bird_utility import *
import argparse

#
# Parse command line arguments
#
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run photo id model")
    parser.add_argument("--model", help="Path to the .tflite model", required=True)
    parser.add_argument("--labels", help="Path to the labels for the .tflite model", required=True)
    parser.add_argument("--image", help="Path to the image to classify", required=True)
    parser.add_argument("--threshold", help="Threshold for classification", type=float, default=0.1)
    args = parser.parse_args()
    return args

#
# Run the photo id model
#
def run_photo_id(model_path, image_path):
    interpreter = create_tflite_interpreter(model_path)

    input_data = load_image_data(image_path)
    run_model(interpreter, [input_data])

    return get_single_output(interpreter)[0]

#
# Print the results from the model
#
def print_results(results, label_path, threshold):
    labels = load_labels_for_model(label_path)
    bird_names = load_common_names()

    print()
    print(f"Results with confidence above {threshold*100.0}%:")
    print_matches(results, labels, bird_names, threshold)


args = parse_arguments()
results = run_photo_id(args.model, args.image)
print_results(results, args.labels, args.threshold)