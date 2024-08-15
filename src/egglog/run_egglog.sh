#!/bin/bash

# Check if the file name is provided as an argument
if [ $# -eq 0 ]; then
    echo "Please provide a file name as an argument."
    exit 1
fi

# Assign the file name to a variable
file_name=$1

# Check if the file exists
if [ ! -f "$file_name" ]; then
    echo "File does not exist."
    exit 1
fi

# Escape special characters in the file name
escaped_file_name=$(printf '%s\n' "$file_name" | sed -e 's/[]\/$*.^[]/\\&/g')

# Replace the string "INPUT" with the escaped file name in "main.tmpl.egg"
sed "s/INPUT/$escaped_file_name/g" main.tmpl.egg | egglog
