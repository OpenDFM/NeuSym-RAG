#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 dataset filepath"
    exit 1
fi

if [ ! -f "$2" ]; then
    echo "Error: $1 is not a valid file"
    exit 1
fi

# Dataset should be chosen from airqa, m3sciqa, scidqa
dataset=$1
# Each line in $input_file should be a downloaded PDF file path, e.g., data/dataset/airqa/papers/acl2024/xxx-xxx-xxx.pdf
input_file=$2

dataset_dir=data/dataset
output_folder=processed_data
total=0
count=0

mkdir -p ${dataset_dir}/${dataset}/${output_folder}

while IFS= read -r file || [ -n "$file" ]; do
    if [ -z "$file" ]; then
        continue
    fi
    echo "Processing PDF ${file}"
    uid=$(basename "$file" | sed 's/\.pdf$//')
    total=$((total + 1))

    if [ -e "${dataset_dir}/${dataset}/${output_folder}/${uid}/auto/${uid}_middle.json" ] && [ -e "${dataset_dir}/${dataset}/${output_folder}/${uid}/auto/${uid}_content_list.json" ] ; then
        continue
    fi

    magic-pdf -p $file -o ${dataset_dir}/${dataset}/${output_folder} -m auto

    if [ $? -eq 0 ]; then
    ┆   echo "Finished processing PDF ${file}"
    else
    ┆   echo "Failed to process PDF ${file}"
    ┆   count=$((count + 1))
    fi
done < "$input_file"

echo "In total, processing $total PDFs, but failed $count PDFs."
