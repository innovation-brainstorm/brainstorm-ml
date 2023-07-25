#!/bin/bash

echo 'HELLO'

echo "input file: $1"
echo "output file: $2"
echo "count: $3"

echo "START GENERATE"

for var in $(seq 1 $3)
do
    echo "$var" >> $2
done
