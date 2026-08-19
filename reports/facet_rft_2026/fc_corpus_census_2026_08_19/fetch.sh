set -e
fetch(){ ds=$1; cfg=$2; off=$3; out=$4; 
  curl -s --max-time 120 "https://datasets-server.huggingface.co/rows?dataset=${ds}&config=${cfg}&split=train&offset=${off}&length=100" -o "$out"
  sz=$(wc -c < "$out"); echo "$out $sz"; }
for off in 0 20000 40000 60000 80000 100000; do fetch "glaiveai%2Fglaive-function-calling-v2" default $off "g2_$off.json"; done
for off in 0 400 800 1200 1700; do fetch "NousResearch%2Fhermes-function-calling-v1" func_calling $off "hfc_$off.json"; done
for off in 0 400 800 1200 1700; do fetch "NousResearch%2Fhermes-function-calling-v1" func_calling_singleturn $off "hst_$off.json"; done
for off in 0 1000 2000 3000 4000; do fetch "NousResearch%2Fhermes-function-calling-v1" glaive_func_calling $off "hgl_$off.json"; done
fetch "NousResearch%2Fhermes-function-calling-v1" json_mode_agentic 0 "hja_0.json"
fetch "NousResearch%2Fhermes-function-calling-v1" json_mode_singleturn 0 "hjs_0.json"
