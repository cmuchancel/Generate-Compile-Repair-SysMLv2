#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="$SCRIPT_DIR/third_party"
SYSML_RELEASE_DIR="$THIRD_PARTY_DIR/SysML-v2-Release"
HAMR_PARSER_DIR="$THIRD_PARTY_DIR/hamr-sysml-parser"
PINNED_COMMIT="b48c37f3bc5702bc4dfce9ce2b7e454720c7c2fb"
HAMR_PINNED_COMMIT="d7c87942ca9f84de611415c8cca0c5916cc9ccae"
ANTLR_VERSION="4.13.2"
ANTLR_JAR="$SCRIPT_DIR/tools/antlr-${ANTLR_VERSION}-complete.jar"

mkdir -p "$THIRD_PARTY_DIR"
if [[ ! -d "$SYSML_RELEASE_DIR/.git" ]]; then
  git clone https://github.com/Systems-Modeling/SysML-v2-Release "$SYSML_RELEASE_DIR"
fi
if [[ ! -d "$HAMR_PARSER_DIR/.git" ]]; then
  git clone https://github.com/sireum/hamr-sysml-parser "$HAMR_PARSER_DIR"
fi

pushd "$SYSML_RELEASE_DIR" >/dev/null
git fetch --all --tags --prune
git checkout "$PINNED_COMMIT"
popd >/dev/null

pushd "$HAMR_PARSER_DIR" >/dev/null
git fetch --all --tags --prune
git checkout "$HAMR_PINNED_COMMIT"
popd >/dev/null

if [[ ! -f "$ANTLR_JAR" ]]; then
  mkdir -p "$SCRIPT_DIR/tools"
  curl -L "https://www.antlr.org/download/antlr-${ANTLR_VERSION}-complete.jar" -o "$ANTLR_JAR"
fi

if ! command -v java >/dev/null 2>&1; then
  echo "ERROR: Java is required to run ANTLR parser artifacts." >&2
  exit 2
fi
if ! command -v javac >/dev/null 2>&1; then
  echo "ERROR: javac is required to compile the HAMR parser runner." >&2
  exit 2
fi

mkdir -p "$SCRIPT_DIR/generated/hamr_java_classes"

# Compile HAMR Java parser + tiny runner
javac -cp "$ANTLR_JAR" \
  -d "$SCRIPT_DIR/generated/hamr_java_classes" \
  "$HAMR_PARSER_DIR/src/org/sireum/hamr/sysml/parser/"*.java \
  "$SCRIPT_DIR/java/ParseSysML.java"

echo "[ok] SysML-v2-Release pinned at: $PINNED_COMMIT"
echo "[ok] HAMR parser pinned at: $HAMR_PINNED_COMMIT"
echo "[ok] Full HAMR Java parser classes at: $SCRIPT_DIR/generated/hamr_java_classes"
