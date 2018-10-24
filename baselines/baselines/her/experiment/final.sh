#!/bin/bash
directory=/home/hainguyen/.local/lib/python3.5/site-packages/gym/envs/robotics/assets/fetch

# Remove trash file (cause error for CUDA)
rm -rf ~/.vn/
rm 1.txt 0.txt

# echo $directory
mv $directory/robot.xml $directory/temp.xml
mv $directory/robot_transparent.xml $directory/robot.xml

python3 door_detection.py

# Swap back
mv $directory/robot.xml $directory/robot_transparent.xml
mv $directory/temp.xml $directory/robot.xml


if [ -f "0.txt" ]
then
	echo "Door undetected."
else
	echo "Detect a door, running the control script!"
	python3 -m baselines.her.experiment.play policy_best.pkl
fi

# Remove trash file (cause error for CUDA)
rm -rf ~/.vn/
rm 1.txt 0.txt

