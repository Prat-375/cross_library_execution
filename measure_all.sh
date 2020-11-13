#!/bin/bash
for (( i =0; i <= 3; i++ ))
do
        ./cross_library_execution $i
        printf ' \n '
done
