#!/bin/bash
filename=$1;
path=$2;
com=$(source first_script.sh $filename | head -n 10| awk '{print $3"_"$2}' | tr '\n' " " | cat);
cd $path;
touch $com;
