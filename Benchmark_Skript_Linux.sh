#!/bin/bash
# Benchmark-Skript mit Nsight Systems Profiling für Linux

NSYS="/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/profilers/Nsight_Systems/target-linux-x64/nsys"   # Pfad zu nsys anpassen!
NCU="/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/profilers/Nsight_Compute/ncu"   # Pfad zu nsys anpassen!

for i in $(seq 1 29); do
    echo "========================================"
    echo "Starte Benchmark mit Datei $i ..."
    echo "========================================"

   

    # Wähle das richtige Binary je nach i
    case $i in
        2)
            EXEC=./Prefix_Free_Parsing_W=5_P=13_Linux
            ;;
        3)
            EXEC=./Prefix_Free_Parsing_W=5_P=23_small_alphabet_Linux
            ;;
        5)
            EXEC=./Prefix_Free_Parsing_W=5_P=23_Linux
            ;;
        6)
            EXEC=./Prefix_Free_Parsing_W=5_P=23_Linux
            ;;
        7)
            EXEC=./Prefix_Free_Parsing_W=5_P=23_Linux
            ;;
        8)
            EXEC=./Prefix_Free_Parsing_W=5_P=23_Linux
            ;;
        9)
            EXEC=./Prefix_Free_Parsing_W=5_P=23_small_alphabet_Linux
            ;;
        10)
            EXEC=./Prefix_Free_Parsing_W=5_P=13_Linux
            ;;
        12)
            EXEC=./Prefix_Free_Parsing_W=5_P=11_Linux
            ;;
        14)
            EXEC=./Prefix_Free_Parsing_W=5_P=23_Linux
            ;;
        15)
            EXEC=./Prefix_Free_Parsing_W=35_P=81_small_alphabet_Linux
            ;;
        17)
            EXEC=./Prefix_Free_Parsing_W=5_P=23_Linux
            ;;
        18)
            EXEC=./Prefix_Free_Parsing_W=5_P=16_Linux
            ;;
        19)
            EXEC=./Prefix_Free_Parsing_W=5_P=23_Linux
            ;;
        21)
            EXEC=./Prefix_Free_Parsing_W=5_P=14_small_alphabet_Linux
            ;;
        24)
            EXEC=./Prefix_Free_Parsing_W=35_P=81_small_alphabet_Linux
            ;;
        25)
            EXEC=./Prefix_Free_Parsing_W=5_P=13_Linux
            ;;
        27)
            EXEC=./Prefix_Free_Parsing_W=35_P=81_small_alphabet_Linux
            ;;
        *)
            EXEC=./Prefix_Free_Parsing_W=5_P=16_Linux
            ;;
    esac
    
     cp -f "$i.txt" input.txt
    
    # Prüfen, ob Binary existiert
    if [ ! -x "$EXEC" ]; then
        echo "❌ Fehler: Binary '$EXEC' wurde nicht gefunden oder ist nicht ausführbar!"
        continue
    fi
    #sudo $NCU --set=detailed -f -o report_nsight_compute_$i profile $EXEC
    # Nsight Systems Profiling + Output-Umleitung
    #$NSYS profile -o report_0 ./Prefix_Free_Parsing_W=5_P=16_Linux
    $NSYS profile --force-overwrite=true --trace=cuda --cuda-memory=true -o report_nsight_system_$i $EXEC
    
    # case $i in
    #    1)
    #       sudo $NCU --set=detailed -f -o report_nsight_compute_$i $EXEC
    #        ;;
    #    11)
    #       sudo $NCU --set=detailed -f -o report_nsight_compute_$i $EXEC
     #      ;;
     #   14)
     #      sudo $NCU --set=detailed -f -o report_nsight_compute_$i $EXEC
      #      ;;
      #  29)
      #     sudo $NCU --set=detailed -f -o report_nsight_compute_$i $EXEC
      #      ;;
       
    #esac
    
    cp -f out.txt output_$i.txt
    
    EXEC=./Lyndon_Grammar_BWT_Linux
    $NSYS profile --force-overwrite=true --trace=cuda --cuda-memory=true -o report_nsight_system_lyndon_grammar_$i $EXEC
    
    cp -f out.txt output_lyndon_$i.txt
   
    EXEC=./libcubwt_Linux
    $NSYS profile --force-overwrite=true --trace=cuda --cuda-memory=true -o report_nsight_system_libcubwt_$i $EXEC
    
     cp -f out.txt output_libcubwt_$i.txt
   
   # case $i in
    #    1)
    #       sudo $NCU --set=detailed -f -o report_nsight_compute_libcubwt_$i $EXEC
    #        ;;
     #   11)
     #      sudo $NCU --set=detailed -f -o report_nsight_compute_libcubwt_$i $EXEC
     #      ;;
     #   14)
     #      sudo $NCU --set=detailed -f -o report_nsight_compute_libcubwt_$i $EXEC
     #       ;;
      #  29)
      #     sudo $NCU --set=detailed -f -o report_nsight_compute_libcubwt_$i $EXEC
        #    ;;
      #
    #esac


 

    echo "Fertig mit Datei $i"
    echo "----------------------------"
done



echo "Alle Dateien wurden verarbeitet."
read -p "Drücke Enter zum Beenden..."
