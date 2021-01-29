#!/bin/bash

#this script monitors the ../video/ directory for new files with the avi extension
inotifywait -m --exclude "[^a][^v][^i]$" ../video -e create -e moved_to |
    while read path action file; do
        echo "The file '$file' appeared in directory '$path' via '$action'"
    done