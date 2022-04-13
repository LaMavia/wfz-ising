#!/bin/bash

if "$@" -lt 1; then
  printf 'usage: %s base_dir\n' "$(basename "$0")"
  exit 1
fi

mkdir -p "${1}/data" "${1}/plots"