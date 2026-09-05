#!/bin/sh
set -eu

archive_url='https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-T.835-201201-S!!SOFT-ZST-E&type=items'
archive_sha256='22526f45c09d5f7c77793aba68b3fbe480f0e1d58315868fc8fa2d60db6db79b'

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
workspace=$(CDPATH= cd -- "$script_dir/../.." && pwd)
oracle_root="$workspace/target/t835-oracle"
download_dir="$oracle_root/downloads"
source_root="$oracle_root/t835-201201"
source_dir="$source_root/Software"
archive="$download_dir/T-REC-T.835-201201-S.zip"

mkdir -p "$download_dir" "$source_root"

if [ ! -f "$archive" ]; then
    temporary_archive=$(mktemp "$download_dir/T-REC-T.835-201201-S.zip.XXXXXX")
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
    echo "T.835 archive checksum mismatch: expected $archive_sha256, found $actual_sha256" >&2
    echo "Move the invalid archive to trash and rerun: $archive" >&2
    exit 1
fi

if [ ! -d "$source_dir" ]; then
    extraction=$(mktemp -d "$oracle_root/extract.XXXXXX")
    if command -v unzip >/dev/null 2>&1; then
        unzip -q "$archive" -d "$extraction"
    else
        python3 -m zipfile -e "$archive" "$extraction"
    fi
    mv "$extraction/Software" "$source_dir"
fi

make -C "$source_dir" final CFLAGS="${CFLAGS:--Wall -O2}"
printf '%s\n' "$source_dir/jpegxr"
