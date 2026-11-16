#!/bin/zsh
# DFL / Sportec open data (Bassek et al. 2025, CC-BY 4.0): 7 matches, metadata + events + tracking
# with per-frame ball possession and ball status. Source: huggingface.co/datasets/pysport/idsse-data
mkdir -p "$(dirname "$0")/../external/dfl" && cd "$(dirname "$0")/../external/dfl"
BASE=https://huggingface.co/datasets/pysport/idsse-data/resolve/main
for m in J03WMX:000001 J03WN1:000001 J03WOH:000002 J03WOY:000002 J03WPY:000002 J03WQQ:000002 J03WR9:000002; do
  id=${m%%:*}; com=DFL-COM-${m##*:}
  for kind in 02_01_matchinformation 03_02_events_raw 04_03_positions_raw_observed; do
    f=DFL_${kind}_${com}_DFL-MAT-${id}.xml
    [[ -s $f ]] && continue
    curl -sSfL --retry 3 -o $f.part "$BASE/$f" && mv $f.part $f || { echo "failed $f"; exit 1; }
    echo "$(date +%T) $f $(du -h $f | cut -f1)"
  done
done
echo done
