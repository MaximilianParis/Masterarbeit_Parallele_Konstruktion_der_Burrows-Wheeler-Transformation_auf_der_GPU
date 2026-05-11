#!/bin/bash
# Benchmark-Skript mit Nsight Systems Profiling für Linux

NSYS="/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/profilers/Nsight_Systems/target-linux-x64/nsys"   # Pfad zu nsys anpassen!
NCU="/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/profilers/Nsight_Compute/ncu"   # Pfad zu nsys anpassen!

for i in $(seq 1 29); do


cmp  "output_lyndon_$i.txt" "output_libcubwt_$i.txt"
#echo $?   # 0 = gleich, 1 = verschieden, 2 = Fehler


   


 


done



