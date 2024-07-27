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

# Loop through all files with the specified extensions in the directory
for file in "${files[@]}"; do
  # Extract the file name without the extension
  filename=$(basename "$file")
  name="${filename%.*}"
  
  # Print the file name without the extension
  if [[ "$name" == *.xyz ]]; then
    dirname="${name%.*}"
  else
    dirname="$name"
  fi

  echo "$dirname"

  python run.py --gpu 0 --conf confs/"$1".conf --dataname "$name" --dir "$dirname"
done