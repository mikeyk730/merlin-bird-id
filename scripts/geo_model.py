#
# Returns the likeliness of each bird species based on location and time of year
#
# Model expectations:
#
# The supplied geo model should take the following inputs:
#   0: longitude (float32)
#   1: week of year (float32)
#   2: latitude (float32)
#
# It should return a list of probabilities (float32), one per species. The supplied labels
# should map the output indices to species codes (e.g. 'amerob')
#
# Example:
#   geo_model.py --model geo_model.tflite --labels geo_model.txt --latitude 37.7 --longitude -122.4 --week 23 --top 25
#

from bird_utility import *
import argparse

#
# Parse command line arguments
#
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run geo model")
    parser.add_argument("--model", help="Path to the .tflite model", required=True)
    parser.add_argument("--labels", help="Path to the labels for the .tflite model", required=True)
    parser.add_argument("--latitude", help="Latitude for the location", required=True)
    parser.add_argument("--longitude", help="Longitude for the location", required=True)
    parser.add_argument("--week", help="Week of the year", required=True)
    parser.add_argument("--top", help="Number of top results to display", type=int, default=25)
    args = parser.parse_args()
    return args

#
# Run the geo model
#
def run_geo_model(model_path, lat, lon, week):
    interpreter = create_tflite_interpreter(model_path)

    lat = np.float32(lat)
    lon = np.float32(lon)
    week = np.float32(week)
    run_model(interpreter, [[lon], [week], [lat]])

    return get_single_output(interpreter)[0]

#
# Print the results from the model
#
def print_results(results, label_path, top_n):
    labels = load_labels_for_model(label_path)
    bird_names = load_common_names()

    print()
    print(f"Top results:")
    print_top_results(results, labels, bird_names, top_n)


args = parse_arguments()
results = run_geo_model(args.model, args.latitude, args.longitude, args.week)
print_results(results, args.labels, args.top)