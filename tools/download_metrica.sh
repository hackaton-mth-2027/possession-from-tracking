#!/bin/zsh
# Metrica Sports open sample data (3 anonymised matches, tracking + events, same provider as the
# challenge). Source: github.com/metrica-sports/sample-data ("please acknowledge the source").
mkdir -p "$(dirname "$0")/../external/metrica" && cd "$(dirname "$0")/../external/metrica"
BASE=https://raw.githubusercontent.com/metrica-sports/sample-data/master
for f in data/Sample_Game_1/Sample_Game_1_RawEventsData.csv \
         data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv \
         data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv \
         data/Sample_Game_2/Sample_Game_2_RawEventsData.csv \
         data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv \
         data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv \
         data/Sample_Game_3/Sample_Game_3_events.json \
         data/Sample_Game_3/Sample_Game_3_metadata.xml \
         data/Sample_Game_3/Sample_Game_3_tracking.txt \
         documentation/events-definitions.pdf; do
  out=$(basename $f)
  [[ -s $out ]] && continue
  curl -sSfL --retry 3 -o $out.part "$BASE/$f" && mv $out.part $out || { echo "failed $f"; exit 1; }
done
ls -la
