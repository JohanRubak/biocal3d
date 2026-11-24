#!/bin/sh
set -eu
STUDIO_SOURCE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec /bin/sh "$STUDIO_SOURCE/setup/install_unix.sh" "$STUDIO_SOURCE" "$@"
