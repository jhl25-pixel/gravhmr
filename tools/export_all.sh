#!/bin/bash
# Re-export every clip on the project page. Outputs in ../data are the cache: delete a clip's files to redo it.
# Excluded wild clips: 41024 (climbing; GeoCalib gravity spread 20 deg) and 5149 (running; 2D failure) -- known failures.
cd "$(dirname "$0")"; P=/data/home/hangleijin/miniforge3/envs/josh/bin/python; H=/data/home/hangleijin; O=../data
run() { id=$1; shift; [ -f $O/$id.json ] && { echo "$id cached"; return; }; $P -I export.py "$@" --id $id --out $O 2>&1 | grep -vE "Warn|warn|^\s*$|INFO"; }
for n in 25021-720 4568-1080 48371-720 6667-720 7595-720 7869-720; do run $n wild --root $H/sports_wild --dir $H/sports_wild/htd_work --name $n --vh 360; done
for n in 14048-720 21477-720 24436-720; do run $n wild --root $H/cycling_wild --dir $H/cycling_wild/htd_work_paper --name $n --vh 360; done
for s in P8_64_outdoor_skateboard P5_40_indoor_walk_big_circle P9_79_outdoor_walk_rectangle; do run $s emdb --seq $s --vh 360; done
echo ALLDONE
