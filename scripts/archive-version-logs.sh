#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 VERSION [DESTINATION]" >&2
    exit 2
fi

version=$1
case "$version" in
    *[!0-9A-Za-z.-]*|'')
        echo "invalid version: $version" >&2
        exit 2
        ;;
esac

destination=${2:-"logs/archive/v$version"}
mkdir -p "$destination"

for host in orangepi p550 meles; do
    node_destination="$destination/$host"
    mkdir -p "$node_destination"
    scp -q -r "$host:edgeforge/logs/v$version/worker" "$node_destination/"
done

echo "archived node logs under $destination"

