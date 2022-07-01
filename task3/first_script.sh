#!/bin/bash
filename=$1
sed  -e 's/[^[:alpha:]]/ /g' $filename | tr '\n' " " |  tr -s " " | tr " " '\n'| tr 'A-Z' 'a-z' | sort | uniq -c | sort -nr | nl | cat
