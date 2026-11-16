#!/bin/zsh
# SkillCorner open data (MIT): 20 A-League 2024/25 matches, broadcast tracking at 10 fps (players
# off camera extrapolated, a team-in-possession field per frame), dynamic events (player
# possessions) and phases of play. Source: github.com/SkillCorner/opendata. ~1.9 GB.
# Real automatic tracking with possession: a validation set whose noise we did not simulate
# (possession/skillcorner.py, scripts/skillcorner_val.py).
mkdir -p "$(dirname "$0")/../external/skillcorner" && cd "$(dirname "$0")/../external/skillcorner"
RAW=https://raw.githubusercontent.com/SkillCorner/opendata/master/data
LFS=https://media.githubusercontent.com/media/SkillCorner/opendata/master/data
get() {  # url, output: a transfer that stalls for 30 s is resumed where it stopped, up to 8 times
  [[ -s $2 ]] && return 0
  for try in {1..8}; do
    curl -sSfL --retry 3 -C - --speed-limit 20000 --speed-time 30 -o $2.part "$1" \
      && mv $2.part $2 && return 0
    sleep 2
  done
  echo "failed $1"; exit 1
}
get $RAW/matches.json matches.json
for id in 1874553 1886347 1899585 1925299 1927964 1953632 1959846 1986691 1996435 1996436 \
          2006229 2006363 2007448 2007721 2010085 2011166 2013725 2015213 2016236 2017461; do
  get $RAW/matches/$id/${id}_match.json ${id}_match.json
  get $RAW/matches/$id/${id}_dynamic_events.csv ${id}_dynamic_events.csv
  get $RAW/matches/$id/${id}_phases_of_play.csv ${id}_phases_of_play.csv
  get $LFS/matches/$id/${id}_tracking_extrapolated.jsonl ${id}_tracking_extrapolated.jsonl
done
ls | wc -l
du -sh .
