#!/bin/bash

# Check if SSH key is loaded
if [[ $(ssh-add -l) ]]; then
  echo "SSH key is loaded."
else
  echo "No SSH key loaded. Adding SSH key..."
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_rsa  # Replace with your key path if needed
fi

# # Check if SSH connection to GitHub is successful
# if ssh -qT git@github.com; then
#   echo "SSH connection to GitHub is successful."
# else
#   echo "SSH connection to GitHub failed. Please check your SSH key configuration."
#   exit 1
# fi

git remote remove origin
git init

# Get the repository URL from command-line argument
repo_url=git@github.com:Rachel1809/PPG.git

if [[ -z $repo_url ]]; then
  echo "Usage: $0 <repository_url>"
  exit 1
fi

git remote add origin $repo_url
git remote set-url origin $repo_url

git add .
git commit -m "first commit"

# Set the default upstream branch
git fetch

# Test Git push
if git push --set-upstream origin main; then
  echo "Git push successful."
else
  echo "Git push failed. Please check your Git configuration and permissions."
fi

git branch --set-upstream-to=origin/main main  # Replace with your branch name if needed
