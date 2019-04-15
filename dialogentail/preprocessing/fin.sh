#!/usr/bin/env bash

MNLI_FILE=$1

tmp=$(mktemp)
tail -n +2 $MNLI_FILE | shuf -o $tmp

tmp2=$(mktemp)
head -1 $MNLI_FILE > $tmp2
cat $tmp >> $tmp2

mv $tmp2 $MNLI_FILE
rm -f $tmp $tmp2

