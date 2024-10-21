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

main_prog=$(sed "s/INPUT/$escaped_file_name/g" main.tmpl.egg)

# extract all the names after "(let ", concatenate all the names with a space, and 
# replace every occurrence of __INPUTS__ with the concatenated names
concat_names=$(sed -n 's/^(let \([a-zA-Z0-9_]*\).*/\"\1\"/p' $file_name | tr '\n' ' ')
schedule_prog=$(sed "s/__INPUTS__/$concat_names/g" schedule.egg)

# find all the names after "(let " and format them as "(extract name)" for each name using sed
extract_prog=$(sed -n 's/^(let \([a-zA-Z0-9_]*\).*/\(extract \1\)/p' $file_name)

prog="$main_prog

$schedule_prog

$extract_prog"

printf '%s\n' "$prog" | egglog-halide-sidecar
