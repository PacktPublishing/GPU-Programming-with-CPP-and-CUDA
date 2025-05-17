#!/bin/bash

# Get the current directory where the script is located
base_dir=$(pwd)

# List of subfolders (assuming there are exactly three subfolders)
subfolders=($(find "$base_dir" -maxdepth 1 -type d | tail -n +2 | head -n 3))

# Program to execute
program="test.py"

# Parameters to pass to the program
param=1000

for i in {1..5}; do
    echo "*** Processing parameter value: $param"
    # Loop over each subfolder
    for subfolder in "${subfolders[@]}"; do
    echo "## Processing folder: $subfolder"
    
    # Change to the subfolder
    cd "$subfolder" || { echo "Failed to enter $subfolder"; exit 1; }
    
    for i in {1..10}; do
        # Execute the program with parameters
        echo "$i"
        python3 "$program" "$param" || { echo "Failed to execute program in $subfolder"; exit 1; }
    done

    # Move back to the base directory
    cd "$base_dir"
    
    echo " "
    done

    # echo "Finished processing all subfolders for parameter $param"
    echo "#############"
    
    # Increment the parameter (multiply by 10 for the next iteration)
    param=$((param * 10))
done


echo "All folders processed."
