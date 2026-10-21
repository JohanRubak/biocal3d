"""Probe Qt in a child process, preserve native crash diagnostics, then launch."""
from pathlib import Path
import os
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    log = here / 'qt_startup.log'
    try: log.write_text('Approved ROI browser startup\nPython: ' + sys.executable + '\n', encoding='utf-8')
    except OSError:
        log = Path.home() / 'BioCal3D_qt_startup.log'
        log.write_text('Approved ROI browser startup\nPython: ' + sys.executable + '\n', encoding='utf-8')
    print('Python:', sys.executable, flush=True)
    print('Startup log:', log, flush=True)
    diagnose = '--diagnose-qt' in sys.argv[1:]
    env = os.environ.copy()
    if diagnose: env['QT_DEBUG_PLUGINS'] = '1'
    probe = subprocess.run([sys.executable, str(here / 'qt_bootstrap.py')], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with log.open('ab') as stream: stream.write(probe.stdout)
    if probe.returncode or diagnose:
        if probe.returncode and not diagnose:
            env['QT_DEBUG_PLUGINS'] = '1'
            detailed = subprocess.run([sys.executable, str(here / 'qt_bootstrap.py')], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            with log.open('ab') as stream: stream.write(b'\nDetailed retry:\n' + detailed.stdout)
        print(probe.stdout.decode('utf-8', errors='replace'), flush=True)
        print('Qt check ' + ('failed.' if probe.returncode else 'passed.') + ' Full details: ' + str(log), flush=True)
        return 1 if probe.returncode else 0
    print('Qt startup check passed. Opening browser…', flush=True)
    process = subprocess.Popen([sys.executable, '-u', str(here / 'browse_approved_rois.py'), *sys.argv[1:]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with log.open('ab') as stream:
        for line in iter(process.stdout.readline, b''):
            stream.write(line); stream.flush(); print(line.decode('utf-8', errors='replace'), end='', flush=True)
    status = process.wait()
    if status: print('Browser exited with code', status, '— full details:', log, flush=True)
    return 1 if status else 0


if __name__ == '__main__': sys.exit(main())
