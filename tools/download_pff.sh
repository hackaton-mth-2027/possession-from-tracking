#!/bin/zsh
# PFF FC / Gradient Sports 2022 World Cup data: 64 matches, broadcast tracking (jsonl.bz2), events,
# metadata, rosters. Access link (a Google Drive folder) after signing up on
# blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset; pass the folder id as the argument.
#   tools/download_pff.sh <drive-folder-id>
[[ -n $1 ]] || { echo "usage: $0 <drive-folder-id>"; exit 1; }
mkdir -p "$(dirname "$0")/../external/pff" && cd "$(dirname "$0")/../external/pff"

entries() {   # "name<TAB>id<TAB>kind" for each entry of a shared folder
  curl -sSfL "https://drive.google.com/embeddedfolderview?id=$1" | python3 -c '
import html, re, sys
for m in re.finditer(r"<a href=\"([^\"]+)\"[^>]*>.*?<div class=\"flip-entry-title\">(.*?)</div>",
                     sys.stdin.read(), re.S):
    url, name = m.group(1), html.unescape(m.group(2))
    kind = "folder" if "/folders/" in url else "file" if "/file/d/" in url else "other"
    fid = re.search(r"/(?:folders|file/d)/([\w-]+)", url)
    print(name, fid.group(1) if fid else "", kind, sep="\t")'
}

fetch() {     # fetch <folder-id> <local dir>, files only (sub-folders: dated change logs, skipped)
  mkdir -p "$2"
  entries "$1" | while IFS=$'\t' read -r name fid kind; do
    [[ $kind == file ]] || continue
    f="$2/$name"
    [[ -s $f ]] && continue
    curl -sSfL --retry 3 -o "$f.part" \
      "https://drive.usercontent.google.com/download?id=$fid&export=download&confirm=t" \
      && mv "$f.part" "$f" || { echo "failed $f"; exit 1; }
    echo "$(date +%T) $f $(du -h "$f" | cut -f1)"
  done
}

entries "$1" | while IFS=$'\t' read -r name fid kind; do
  case $kind in
    folder) fetch "$fid" "$name" ;;
    file) [[ -s $name ]] || curl -sSfL --retry 3 -o "$name" \
            "https://drive.usercontent.google.com/download?id=$fid&export=download&confirm=t" ;;
  esac
done
echo done
