#!/bin/sh

# ! Change project directory to match
PROJECT_DIR="/home/daniel/dev/TUe-Thesis"

data_dir="$PROJECT_DIR/data/"

log_file="$PROJECT_DIR/script/download.log"
links_file="$PROJECT_DIR/download_links.txt"

# Enter data directory
cd $data_dir

# Start running wget in the background
wget --no-clobber --tries=45 -a $log_file -i $links_file &

# Show the output in the log file until wget ends
tail -f $log_file --pid $!

echo "\n++++ Unzipping files...\n"
# Recursively unzip all files in the directory and keep the original zipped file
# when prompted if we want to override a file automatically respond with 'no'
yes n | gunzip -rk ./

# Hard coded commands for the konect datasets in order to only extract the needed files
# Add .txt extension to file so the data loading functions recognize it
# TODO Find a better way to do this
if [ -f out.wikipedia_link_en.txt ]
then
    echo "++++ Skipping download.tsv.wikipedia_link_en.tar.bz2 because out.wikipedia_link_en.txt already exists."
else
    tar -xvjf download.tsv.wikipedia_link_en.tar.bz2 wikipedia_link_en/out.wikipedia_link_en --strip-components 1
    sed -i 's/%/#/g' out.wikipedia_link_en
    mv out.wikipedia_link_en out.wikipedia_link_en.txt
fi

if [ -f out.friendster.txt ]
then
    echo "++++ Skipping download.tsv.friendster.tar.bz2 because out.friendster.txt already exists."
else
    tar -xvjf download.tsv.friendster.tar.bz2 friendster/out.friendster --strip-components 1
    sed -i 's/%/#/g' out.friendster
    mv out.friendster out.friendster.txt
fi
