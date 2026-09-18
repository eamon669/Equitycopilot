"""可双击的标准库启动器：环境自检、按需安装、日志、端口检测。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
SUPPORTED_MINORS = (10, 11, 12, 13)
REQUIRED_FILES = (
    'app.py', 'financial_engine.py', 'agent_copilot.py', 'mock_data.py',
    'requirements.txt', 'assets/fonts/EquityCJK-Regular.ttf',
)


def supported_python(version=None, bits=None):
    version = version or sys.version_info
    bits = bits if bits is not None else struct.calcsize('P') * 8
    return version[0] == 3 and version[1] in SUPPORTED_MINORS and bits == 64


def child_environment():
    """避免用户的全局 PYTHONPATH / PYTHONHOME 污染隔离环境。"""
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    return env


def missing_resources(root=ROOT):
    return [name for name in REQUIRED_FILES if not (root / name).is_file()]


def choose_port(preferred=8501):
    if not 1024 <= preferred <= 65525:
        raise ValueError('Port must be in 1024..65525.')
    for port in range(preferred, preferred + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError('No free local port. Close an old Copilot window or use --port 8601.')


class Diagnostics:
    def __init__(self, root=ROOT):
        try:
            folder = root / 'logs'
            folder.mkdir(exist_ok=True)
            self.path = folder / 'startup.log'
            self.stream = self.path.open('w', encoding='utf-8')
        except OSError:
            folder = Path(tempfile.gettempdir()) / 'equity-copilot-logs'
            folder.mkdir(exist_ok=True)
            self.path = folder / 'startup.log'
            self.stream = self.path.open('w', encoding='utf-8')
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            # 不记录环境变量、API Key、用户文档或提问。
            self.stream.write(str(text) + '\n')
            self.stream.flush()
            try:
                print(text, flush=True)
            except UnicodeEncodeError:
                print(str(text).encode('ascii', 'replace').decode(), flush=True)

    def close(self):
        self.stream.close()


def run_logged(command, log, cwd=ROOT):
    """参数数组执行，支持中文/空格路径；不拼接 shell 命令。"""
    process = subprocess.Popen(
        [str(x) for x in command], cwd=str(cwd), env=child_environment(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace', bufsize=1,
    )
    try:
        for line in process.stdout:
            log.write(line.rstrip('\r\n'))
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return 130
    finally:
        process.stdout.close()


def environment_probe(python):
    """检查实际已安装版本和关键接口，而不是只看 venv 目录存在。"""
    script = r'''
import importlib.metadata as md
import pathlib, re, sys
from packaging.requirements import Requirement
for line in pathlib.Path('requirements.txt').read_text().splitlines():
    line=line.strip()
    if not line or line.startswith('#'):continue
    r=Requirement(line)
    if not r.specifier.contains(md.version(r.name)):raise RuntimeError('Version mismatch: '+r.name)
import streamlit, pandas, numpy, plotly, pdfplumber, reportlab, pypdfium2
assert hasattr(streamlit,'fragment')
assert sys.version_info[:2] >= (3,10) and sys.version_info[:2] <= (3,13)
print('Dependency probe OK')
'''
    try:
        result = subprocess.run(
            [str(python), '-c', script], cwd=str(ROOT), env=child_environment(),
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=45,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def prepare_environment(log, no_install=False):
    # 使用新目录；保留用户可能已有的 .venv，不在修复时删除环境。
    directory = ROOT / f'.venv-copilot-py{sys.version_info.major}{sys.version_info.minor}'
    python = directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        if no_install:
            raise RuntimeError('No prepared environment. Run START_WINDOWS.bat once without --no-install.')
        log.write('[2/4] Creating an isolated Python environment...')
        if run_logged([sys.executable, '-m', 'venv', directory], log) != 0:
            raise RuntimeError('Could not create the environment. Check folder write access and Python venv support.')
    log.write('[2/4] Checking installed dependencies...')
    if not environment_probe(python):
        if no_install:
            raise RuntimeError('Dependencies are missing or incompatible. Run without --no-install to repair.')
        log.write('Installing the tested dependency set; first run needs Internet access.')
        code = run_logged([
            python, '-m', 'pip', 'install', '--only-binary=:all:',
            '--timeout', '20', '--retries', '1', '-r', ROOT / 'requirements.txt',
        ], log)
        if code != 0:
            raise RuntimeError('Dependency installation failed. The pip error above is saved in startup.log. No app was started.')
        if not environment_probe(python):
            raise RuntimeError('Installed packages failed the dependency probe. See startup.log.')
    return python


def self_check(python, log):
    """启动前检查计算和真正的 PDF 渲染；不生成用户报告，不使用网络模型。"""
    script = r'''
from mock_data import demo_financials, demo_peers
from financial_engine import calculate_metrics, peer_matrix
from agent_copilot import memo_to_pdf, pdf_compatibility_bundle
m=calculate_metrics(demo_financials())
assert m.iloc[-1].balance_gap == -90
assert len(peer_matrix(demo_peers())) == 8
b=pdf_compatibility_bundle(memo_to_pdf('# Export self-check\n\nChinese PDF: 中文内容可见。'))
assert b['page_count'] == 1 and len(b['compatible_pdf']) > 1000
print('Financial engine and PDF rendering self-check OK')
'''
    if run_logged([python, '-c', script], log) != 0:
        raise RuntimeError('Self-check failed before app launch. See startup.log for the original error.')


def watch_server(port, process, log, open_browser=True):
    # 此逻辑运行在使用者电脑；健康检查成功后才打开浏览器。
    url = f'http://127.0.0.1:{port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(100):
        if process.poll() is not None:
            return
        try:
            with opener.open(url + '/_stcore/health', timeout=.5) as response:
                if response.status == 200:
                    log.write('[4/4] READY: ' + url)
                    log.write('Keep this terminal open while using Copilot. Press Ctrl+C to stop.')
                    if open_browser:
                        try:
                            webbrowser.open(url)
                        except Exception:
                            log.write('Could not open a browser automatically. Copy the URL above.')
                    return
        except Exception:
            time.sleep(.3)
    log.write('Server is taking longer to become ready. Check the errors above; URL: ' + url)


def serve(python, port, log, no_browser=False):
    command = [str(python), '-m', 'streamlit', 'run', str(ROOT / 'app.py'),
               '--server.address', '127.0.0.1', '--server.port', str(port),
               '--server.headless', 'true', '--browser.gatherUsageStats', 'false']
    process = subprocess.Popen(
        command, cwd=str(ROOT), env=child_environment(), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1,
    )
    watcher = threading.Thread(target=watch_server, args=(port, process, log, not no_browser), daemon=True)
    watcher.start()
    try:
        for line in process.stdout:
            log.write(line.rstrip('\r\n'))
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            code = process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
        log.write('Server stopped by user.')
        code = 0
    finally:
        process.stdout.close()
        watcher.join(timeout=2)
    if code:
        raise RuntimeError(f'Streamlit exited with code {code}. The original error is saved in startup.log.')
    return 0


def entrypoint(argv=None):
    parser = argparse.ArgumentParser(description='Equity Copilot launcher and self-check')
    parser.add_argument('--no-pause', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--no-install', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args(argv)
    log = Diagnostics()
    result = 1
    try:
        log.write('EQUITY COPILOT 1.1 / STARTUP DIAGNOSTICS')
        log.write('Log: ' + str(log.path))
        log.write(f'[1/4] Python {platform.python_version()}, {struct.calcsize("P") * 8}-bit, {platform.system()}')
        if not supported_python():
            raise RuntimeError('Use 64-bit Python 3.10, 3.11, 3.12 or 3.13. Recommended: Python 3.12. Then run START_WINDOWS.bat.')
        missing = missing_resources()
        if missing:
            raise RuntimeError('Incomplete project. Extract ALL files from the ZIP first. Missing: ' + ', '.join(missing))
        python = prepare_environment(log, args.no_install)
        log.write('[3/4] Checking financial calculations and PDF output...')
        self_check(python, log)
        if args.check_only:
            log.write('ALL CHECKS PASSED. You can now run START_WINDOWS.bat.')
            result = 0
        else:
            result = serve(python, choose_port(args.port), log, args.no_browser)
    except KeyboardInterrupt:
        log.write('Startup cancelled by user.')
        result = 130
    except Exception as exc:
        log.write('STARTUP FAILED: ' + str(exc))
        log.write(traceback.format_exc())
        log.write('Send the visible error or logs/startup.log for diagnosis. Do not include API keys.')
    finally:
        log.close()
        if os.name == 'nt' and not args.no_pause:
            try:
                input('\nPress Enter to close this window...')
            except (EOFError, KeyboardInterrupt):
                pass
    return result


if __name__ == '__main__':
    raise SystemExit(entrypoint())
