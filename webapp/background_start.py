"""Start the local learning site without keeping a console window open."""
import ctypes
import json
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = 'http://127.0.0.1:8124/'


def site_ready():
    try:
        with urllib.request.urlopen(URL + 'api/catalog', timeout=1) as response:
            return isinstance(json.load(response).get('lessons'), dict)
    except (OSError, ValueError):
        return False


def main():
    if not site_ready():
        child = subprocess.Popen(
            [sys.executable, str(ROOT / 'webapp' / 'pilot_server.py')],
            cwd=str(ROOT), creationflags=subprocess.CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(40):
            if site_ready() or child.poll() is not None:
                break
            time.sleep(0.25)
        if not site_ready():
            ctypes.windll.user32.MessageBoxW(
                None, '网站启动失败。请检查 8124 端口是否被占用，并在项目目录运行 python webapp\\pilot_server.py 查看错误。',
                '学习网站', 0x10,
            )
            return
    webbrowser.open(URL)


if __name__ == '__main__':
    main()
