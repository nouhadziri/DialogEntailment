#!/usr/bin/env bash

BOLD='\033[1m'
RED='\033[0;31m'
BROWN='\033[0;33m'
NORMAL='\033[0m'

OVERWRITE=0

# Derived from https://stackoverflow.com/a/246128
CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CONFIG_FILE="${CURR_DIR}/configs/esim_elmo.jsonnet"

while [[ $# > 1 ]]
do
	key="$1"

	case $key in
		-c|--config)
		CONFIG_FILE="$2"
		shift # past argument
		;;
		-s|--serialization-output)
		OUT_DIR="$2"
		shift # past argument
		;;
		-w|--overwrite)
		OVERWRITE=1
		;;
    		*)
	        # unknown option
    		;;
	esac
	shift # past argument or value
done

if [ ! -f "$CONFIG_FILE" ];
then
    echo -e "${RED}Training config file cannot be found: '$CONFIG_FILE'${NORMAL}"
    exit 1
fi

if [ -z "$OUT_DIR" ];
then
    printf "%sOutput directory is not specified. Pass it using: -s OR --serialization-output%s\n" "${RED}" "${NORMAL}"
    exit 1
fi

if [ -d "$OUT_DIR" ];
then
    if [ "$OVERWRITE" -eq 1 ];
    then
        rm -fr "$OUT_DIR"
        echo -e "${BROWN}Output directory cleaned up!!${NORMAL}"
    fi
fi

echo -e "${BOLD}Training is about to start... config: ${CONFIG_FILE} - output will be saved to ${OUT_DIR}${NORMAL}"
allennlp train ${CONFIG_FILE} -s ${OUT_DIR}
