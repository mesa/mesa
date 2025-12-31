#!/bin/bash

cd examples/Flockers
python run.py &
PID=$!
sleep 3
curl localhost:8521 | grep Boids
kill $PID
