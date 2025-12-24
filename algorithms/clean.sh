#!/bin/sh

set -eu

rm_file() {
  p="$1"
  if [ -f "$p" ]; then
    rm -f -- "$p"
    echo "deleted:$p"
  else
    echo "can't fine $p, skip"
  fi
}

rm_dir() {
  p="$1"
  if [ -d "$p" ]; then
    rm -rf -- "$p"
    echo "deleted:$p/"
  else
    echo "can't find $p/, skip"
  fi
}

rm_file "./debug_log.txt"
rm_dir "./output"
