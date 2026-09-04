#!/bin/sh
set -eu

archive_url='https://www.itu.int/wftp3/public/t/testsignal/SpeImage/T834/v2014_10/ITU-T_T.834(2014-10)_ConformanceSuite.zip'
archive_sha256='c066c5e24a212f3bb09eaf235cf21359754ffbb747d9fedc620e09629ca2a55d'

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
workspace=$(CDPATH= cd -- "$script_dir/../.." && pwd)
suite_root="$workspace/target/t834-conformance"
download_dir="$suite_root/downloads"
archive="$download_dir/ITU-T_T.834(2014-10)_ConformanceSuite.zip"
extracted="$suite_root/suite-2014"

mkdir -p "$download_dir"

if [ ! -f "$archive" ]; then
    temporary_archive=$(mktemp "$download_dir/T834-ConformanceSuite.zip.XXXXXX")
    curl -L --fail --retry 5 --retry-delay 2 --retry-all-errors \
        --silent --show-error "$archive_url" -o "$temporary_archive"
    mv "$temporary_archive" "$archive"
fi

if command -v sha256sum >/dev/null 2>&1; then
    actual_sha256=$(sha256sum "$archive" | awk '{print $1}')
else
    actual_sha256=$(shasum -a 256 "$archive" | awk '{print $1}')
fi

if [ "$actual_sha256" != "$archive_sha256" ]; then
    echo "T.834 archive checksum mismatch: expected $archive_sha256, found $actual_sha256" >&2
    echo "Move the invalid archive to trash and rerun: $archive" >&2
    exit 1
fi

if [ ! -d "$extracted" ]; then
    extraction=$(mktemp -d "$suite_root/extract.XXXXXX")
    unzip -q "$archive" -d "$extraction"
    mv "$extraction/JXR_ConformanceSuite_2014" "$extracted"
fi

printf '%s\n' "$extracted"
