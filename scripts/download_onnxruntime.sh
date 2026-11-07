#!/usr/bin/env bash
# Download a prebuilt ONNX Runtime release (CPU) into third_party/onnxruntime.
#   scripts/download_onnxruntime.sh [platform]   platform: linux-x64 | linux-aarch64 | osx-arm64 | osx-x86_64 | osx-universal2 | win-x64
# (default: detected). Override the version with ORT_VERSION=1.20.1.
set -euo pipefail
VER="${ORT_VERSION:-1.20.1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/onnxruntime"
plat="${1:-}"
if [ -z "$plat" ]; then
  os="$(uname -s)"; arch="$(uname -m)"
  case "$os" in
    Linux)  case "$arch" in x86_64) plat=linux-x64;; aarch64|arm64) plat=linux-aarch64;; *) echo "unsupported arch $arch"; exit 1;; esac;;
    Darwin) case "$arch" in arm64) plat=osx-arm64;; x86_64) plat=osx-x86_64;; *) plat=osx-universal2;; esac;;
    MINGW*|MSYS*|CYGWIN*) plat=win-x64;;
    *) echo "unsupported OS $os"; exit 1;;
  esac
fi
case "$plat" in win-*) ext=zip;; *) ext=tgz;; esac
name="onnxruntime-${plat}-${VER}"
url="https://github.com/microsoft/onnxruntime/releases/download/v${VER}/${name}.${ext}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "downloading $url"
curl -fL --retry 3 -o "$tmp/ort.$ext" "$url"
if [ "$ext" = zip ]; then (cd "$tmp" && unzip -q "ort.$ext"); else tar -xzf "$tmp/ort.$ext" -C "$tmp"; fi
rm -rf "$DEST"; mkdir -p "$(dirname "$DEST")"
mv "$tmp/$name" "$DEST"
echo "ONNX Runtime $VER ($plat) -> $DEST"
ls "$DEST/lib"
