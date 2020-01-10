#!/usr/bin/env bash

NAME="sent140"

cd ../utils

python stats.py --name $NAME

cd ../$NAME