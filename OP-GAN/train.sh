#!/bin/bash

GPU=""
DISTRIBUTED=0
NB_PROCS=0

# https://wiki.bash-hackers.org/howto/getopts_tutorial
while getopts ":g:p:d" opt; do
  case $opt in
    g) GPU="${OPTARG}"
    ;;
    d) DISTRIBUTED=1
    ;;
    p) NB_PROCS="${OPTARG}"
    ;;
    \?) echo "Invalid option -${OPTARG}" >&2
    ;;
  esac
done

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONUNBUFFERED=1

if [[ -z "${GPU}" ]]
then
      DEVICE="CPU"
else
      DEVICE="GPU ${GPU}"
      if [[ "${NB_PROCS}" -eq 0 ]]
      then
#           https://unix.stackexchange.com/questions/193039/how-to-count-the-length-of-an-array-defined-in-bash
#           https://stackoverflow.com/questions/10586153/split-string-into-an-array-in-bash
          IFS=',' read -ra GPU_ARRAY <<< "${GPU}"
          NB_PROCS="${#GPU_ARRAY[@]}"
      fi
fi
if [[ "${DISTRIBUTED}" -eq 1 && "${NB_PROCS}" -ne 1 ]]
then
    if [[ "${NB_PROCS}" -eq 0 ]]
    then
        echo "Set the number of processes first!"
        exit 1
    fi
    echo "Starting distributed (multi-process) training with ${NB_PROCS} processes on ${DEVICE}."
    python3 -m torch.distributed.launch --nproc_per_node=${NB_PROCS} \
                                        code/main.py --cfg code/cfg/cfg_file_train.yml --distributed
else
    echo "Starting local (single-process) training on ${DEVICE}."
    python3 -u code/main.py --cfg code/cfg/cfg_file_train.yml
fi
echo "Done."
