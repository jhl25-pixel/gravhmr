#!/bin/bash
# Re-export every clip on the project page. Outputs in ../data are the cache: delete a clip's files to redo it.
# Wild clips on the page: 6667 running, 25021 / 4568 skateboarding, 48371 skating, 21477 cycling.
# Excluded wild clips: 41024 (climbing; GeoCalib gravity spread 20 deg) and 5149 (running; 2D failure) -- known failures.
cd "$(dirname "$0")"; P=/data/home/hangleijin/miniforge3/envs/josh/bin/python; H=/data/home/hangleijin; O=../data
run() { id=$1; shift; [ -f $O/$id.json ] && { echo "$id cached"; return; }; $P -I export.py "$@" --id $id --out $O 2>&1 | grep -vE "Warn|warn|^\s*$|INFO"; }
for n in 6667-720 25021-720 4568-1080 48371-720; do run $n wild --root $H/sports_wild --dir $H/sports_wild/htd_work --name $n --vh 360; done
for n in 21477-720; do run $n wild --root $H/cycling_wild --dir $H/cycling_wild/htd_work_paper --name $n --vh 360; done
for s in P8_64_outdoor_skateboard P7_58_outdoor_parcours P7_56_outdoor_stairs_up_down P9_79_outdoor_walk_rectangle; do run $s emdb --seq $s --vh 360; done
echo ALLDONE
