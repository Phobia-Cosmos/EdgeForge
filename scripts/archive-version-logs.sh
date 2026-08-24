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

log_root=${EDGEFORGE_LOG_DIR:-logs}
source="$log_root/v$version"
if [ -d "$source" ]; then
    mkdir -p "$destination/local"
    # cp -a merges existing files and preserves prior records; it never removes
    # an older run when the archive command is repeated.
    cp -a "$source/." "$destination/local/"
else
    echo "local version log directory is missing: $source" >&2
    exit 1
fi

# Remote boards are optional. Set ARCHIVE_REMOTE_HOSTS to a whitespace-separated
# list when a board is reachable. A board being offline must not lose local logs
# or prevent the remaining nodes from being archived.
remote_hosts=${ARCHIVE_REMOTE_HOSTS:-}
for host in $remote_hosts; do
    node_destination="$destination/$host"
    mkdir -p "$node_destination"
    if scp -q -o ConnectTimeout=5 -r "$host:edgeforge/logs/v$version/worker" "$node_destination/"; then
        printf '%s\n' "archived remote worker logs: $host"
    else
        printf '%s\n' "remote host unavailable (preserved as a gap): $host" >> "$destination/REMOTE_GAPS"
    fi
done

manifest="$destination/SHA256SUMS"
tmp_manifest="$manifest.tmp.$$"
(
    cd "$destination"
    find . -type f ! -name 'SHA256SUMS' ! -name 'SHA256SUMS.tmp.*' -printf '%P\0' \
        | sort -z \
        | xargs -0 sha256sum > "$(basename "$tmp_manifest")"
)
mv "$tmp_manifest" "$manifest"

echo "archived version logs under $destination"
