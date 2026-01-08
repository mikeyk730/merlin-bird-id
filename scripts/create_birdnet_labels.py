from bird_utility import *
import argparse

#
# Parse command line arguments
#
def parse_arguments():
    parser = argparse.ArgumentParser(description="Create BirdNET labels")
    parser.add_argument("--labels", help="Path to the labels for the .tflite model", required=True)
    args = parser.parse_args()
    return args

args = parse_arguments()
labels = load_labels_for_model(args.labels)
common = load_common_names()
scientific = load_scientific_names()

for label in labels:
    print(f"{scientific.get(label, 'Unknown Species')}_{common.get(label, 'Unknown Species')}_{label}")