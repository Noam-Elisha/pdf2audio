# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for pdf2audio."""

import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Project paths
PROJECT_DIR = os.path.abspath('.')
SRC_DIR = os.path.join(PROJECT_DIR, 'src')

# Collect data files upfront
misaki_datas = collect_data_files('misaki')

all_datas = [
    # Flask templates
    (os.path.join(SRC_DIR, 'pdf2audio', 'web', 'templates'), os.path.join('pdf2audio', 'web', 'templates')),
] + misaki_datas

a = Analysis(
    [os.path.join(SRC_DIR, 'pdf2audio', 'launcher.py')],
    pathex=[SRC_DIR],
    binaries=[],
    datas=all_datas,
    hiddenimports=[
        # Core app
        'pdf2audio',
        'pdf2audio.cli',
        'pdf2audio.web',
        'pdf2audio.web.app',
        'pdf2audio.pdf_extract',
        'pdf2audio.tts_engine',
        'pdf2audio.audio_formats',
        'pdf2audio.model_manager',
        'pdf2audio.manifest',
        # TTS engine
        'kokoro',
        'misaki',
        'misaki.en',
        'misaki.ja',
        'misaki.zh',
        'misaki.ko',
        'misaki.vi',
        'misaki.he',
        'misaki.espeak',
        'misaki.cutlet',
        'misaki.token',
        'misaki.transcription',
        # Audio
        'soundfile',
        '_soundfile_data',
        'lameenc',
        # PDF
        'fitz',
        'pymupdf',
        # ML
        'torch',
        # Web
        'flask',
        'flask.json',
        'jinja2',
        'werkzeug',
        # Model download
        'huggingface_hub',
        # Misc
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'PIL',
        'tkinter',
        'pytest',
        'IPython',
        'notebook',
        'sphinx',
        'docutils',
        'pygments',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pdf2audio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Show console so user can see server status
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='pdf2audio',
)
