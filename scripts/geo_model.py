from merlin_utility import *
import argparse

#
# Parse command line arguments
#
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Merlin photo id")
    parser.add_argument("--model_path", help="Path to the .tflite model", required=True)
    parser.add_argument("--label_path", help="Path to the labels for the .tflite model", required=True)
    parser.add_argument("--lat", help="Latitude for the location", required=True)
    parser.add_argument("--lon", help="Longitude for the location", required=True)
    parser.add_argument("--week", help="Week of the year", required=True)
    parser.add_argument("--top", help="Number of top results to display", type=int, default=25)
    args = parser.parse_args()
    return args

#
# Run the Merlin geo model
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
    bird_names = load_species_id_map()

    print()
    print(f"Top results:")
    print_top_results(results, labels, bird_names, top_n)


args = parse_arguments()
results = run_geo_model(args.model_path, args.lat, args.lon, args.week)
print_results(results, args.label_path, args.top)
