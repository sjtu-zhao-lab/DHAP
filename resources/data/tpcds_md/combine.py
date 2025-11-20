import json
import glob

# Initialize an empty dictionary to store the combined data
combined_data = {"tables": {}}

# Find all JSON files with the specified pattern
for file in glob.glob("*.metadata.json"):
    # Extract the table name from the file name
    table_name = file.split(".")[0]

    # Read the JSON file
    with open(file, "r") as f:
        data = json.load(f)

    # Add the data to the combined dictionary
    combined_data["tables"][table_name] = data

# Write the combined data to a new JSON file
with open("combined.json", "w") as f:
    json.dump(combined_data, f, indent=4)