#!/bin/bash

git pull

# Base directory
base_directory="/workspace/data"

# Check if the last directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 last_directory"
  exit 1
fi

# Full directory path
directory="$base_directory/$1/input"

# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "The directory $directory does not exist."
  exit 1
fi

# Check if there are no files with the specified extensions
shopt -s nullglob
files=("$directory"/*.{ply,xyz,npy})
if [ ${#files[@]} -eq 0 ]; then
  echo "No .ply, .xyz, or .npy files found in the directory $directory."
  exit 1
fi

total_chamferl1=0
total_chamferl2=0
total_haus=0
file_count=0
# Loop through all files with the specified extensions in the directory
for file in "${files[@]}"; do
  # Extract the file name without the extension
  filename=$(basename "$file")
  name="${filename%.*}"

  # Remove ".xyz" extension if it exists in the dataname
  if [[ "$name" == *.xyz ]]; then
    dataname="${name%.*}"
  else
    dataname="$name"
  fi

  # Print the dataname
  echo "$dataname"

  # Run the Python script with the specified arguments
  output=$(python evaluation/"$1"/eval_mesh.py --conf confs/"$1".conf --dataname "$dataname" --dir "$name"_60000_mesh_new_centroid_surf0-05)
  
  chamferl1=$(echo "$output" | grep -oP 'Chamfer-L1: \s*\K[0-9.]+')
  chamferl2=$(echo "$output" | grep -oP 'Chamfer-L2: \s*\K[0-9.]+')
  haus=$(echo "$output" | grep -oP 'Hausdorff: \s*\K[0-9.]+')
  
  echo "$chamferl1"
  echo "$chamferl2"
  echo "$haus"

  total_chamferl1=$(echo "$total_chamferl1 + $chamferl1" | bc)
  total_chamferl2=$(echo "$total_chamferl2 + $chamferl2" | bc)
  total_haus=$(echo "$total_haus + $haus" | bc)
  file_count=$((file_count + 1))
done
if [ $file_count -gt 0 ]; then
  average_chamferl1=$(echo "scale=4; $total_chamferl1 / $file_count" | bc)
  average_chamferl2=$(echo "scale=4; $total_chamferl2 / $file_count" | bc)
  average_haus=$(echo "scale=4; $total_haus / $file_count" | bc)
  echo "Avg. Chamfer-L1: $average_chamferl1"
  echo "Avg. Chamfer-L2: $average_chamferl2"
  echo "Avg. Hausdorff: $average_haus"
else
  echo "No chamfer values found."
fi