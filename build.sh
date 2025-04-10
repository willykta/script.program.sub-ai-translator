#!/bin/bash
set -e

# Katalog, w którym jest skrypt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADDON_ID="plugin.subtitle.translator"
ADDON_DIR="${SCRIPT_DIR}"
ADDON_XML="${ADDON_DIR}/addon.xml"
DIST_DIR="${SCRIPT_DIR}/dist"

# Wydobądź wersję
version=$(xmllint --xpath 'string(/addon/@version)' "$ADDON_XML")

if [[ -z "$version" ]]; then
  echo "❌ Nie udało się znaleźć wersji w $ADDON_XML"
  exit 1
fi

IFS='.' read -r major minor patch <<< "$version"
new_patch=$((patch + 1))
new_version="${major}.${minor}.${new_patch}"

# Podmień wersję w pliku (niezależnie od struktury)
# Tworzymy nowy plik tymczasowy i zamieniamy, bo sed nie zadziała z xpath
tmp_file=$(mktemp)
xmllint --xpath "/*" "$ADDON_XML" | sed "s/version=\"$version\"/version=\"$new_version\"/" > "$tmp_file"
mv "$tmp_file" "$ADDON_XML"


# Utwórz dist/
mkdir -p "$DIST_DIR"

# Stwórz ZIP z katalogiem głównym (wymagane przez Kodi!)
zip_name="${ADDON_ID}-${new_version}.zip"
zip_path="${DIST_DIR}/${zip_name}"

# Zrób ZIP z katalogu, nie jego zawartości
cd "$SCRIPT_DIR/.."
zip -r "$zip_path" "$(basename "$ADDON_DIR")" \
  -x "*.pyc" "*__pycache__/*" "*.DS_Store" "*.git*" "*wheels/*" "*/dist/*" "*/build.sh"

echo "✅ Zbudowano: $zip_path (wersja: $new_version)"
echo "🔄 Zaktualizowano addon.xml do wersji: $new_version"