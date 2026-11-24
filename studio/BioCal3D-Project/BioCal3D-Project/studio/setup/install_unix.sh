#!/bin/sh
set -eu
STUDIO_SOURCE=$1
shift
STUDIO_PYTHON=${BIOCAL3D_PYTHON:-python3}
if ! command -v "$STUDIO_PYTHON" >/dev/null 2>&1; then
    echo 'Install Python 3.11 or 3.12 first. Then run this launcher again.' >&2
    exit 1
fi
"$STUDIO_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10, 3.11 or 3.12 (BIOCAL3D_PYTHON can select it)."'
case $(uname -s) in
    Darwin) STUDIO_INSTALL="$HOME/Library/Application Support/BioCal3DStudio" ;;
    *) STUDIO_INSTALL="${XDG_DATA_HOME:-$HOME/.local/share}/BioCal3DStudio" ;;
esac
mkdir -p "$STUDIO_INSTALL/logs"
if [ ! -x "$STUDIO_INSTALL/runtime/bin/python" ]; then
    "$STUDIO_PYTHON" -m venv "$STUDIO_INSTALL/runtime"
fi
STUDIO_RUN_PYTHON="$STUDIO_INSTALL/runtime/bin/python"
# Reconcile dependencies on each launch; pip skips matching installed packages.
if ! "$STUDIO_RUN_PYTHON" -m pip install --only-binary=:all: -r "$STUDIO_SOURCE/requirements.txt" >"$STUDIO_INSTALL/logs/setup.log" 2>&1; then
    cat "$STUDIO_INSTALL/logs/setup.log" >&2
    echo 'Setup failed. See the log above. Linux may also need python3-venv and desktop OpenGL libraries.' >&2
    exit 1
fi
exec "$STUDIO_RUN_PYTHON" "$STUDIO_SOURCE/launch_studio.py" "$@"
