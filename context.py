import sys
import os
import xbmc
import xbmcgui
import xbmcaddon

addon_path = xbmcaddon.Addon().getAddonInfo('path')
lib_path = os.path.join(addon_path, "resources", "lib")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from core import subtitle_sources
from core.language_labels import get_language_label

addon = xbmcaddon.Addon()
_ = addon.getLocalizedString

kodi_lang = xbmc.getLanguage(xbmc.ISO_639_1, True)

def resolve_path():
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return sys.argv[1]
    
    path = xbmc.getInfoLabel("ListItem.Path")
    filename = xbmc.getInfoLabel("ListItem.FilenameAndPath") or xbmc.getInfoLabel("ListItem.Filename")
    
    if path and filename:
        return os.path.join(path, filename) if not filename.startswith(path) else filename
    return None

def notify_error(message_id):
    xbmcgui.Dialog().notification(_(30006), _(message_id), xbmcgui.NOTIFICATION_ERROR, 3000)

def run_translation(srt_path):
    xbmc.executebuiltin(f'RunScript(script.program.sub-ai-translator, "{srt_path}")')

def handle_external_subtitle(path):
    run_translation(path)

def handle_embedded_subtitle(video_path, track_index):
    progress = xbmcgui.DialogProgress()
    progress.create("Subtitle AI Translator", _(30011)) 

    def on_progress(pct):
        if progress.iscanceled():
            raise Exception("Subtitle extraction cancelled by user")
        percent = int(pct * 100)
        progress.update(percent, f"{_(30011)}: {percent}%")

    try:
        srt_path = subtitle_sources.extract_to_temp_srt(video_path, track_index, on_progress)
        progress.close()
        run_translation(srt_path)
    except Exception as e:
        progress.close()
        xbmc.log(f"[SubAI] Failed to extract subtitles: {e}", xbmc.LOGERROR)
        notify_error(30012) 
        sys.exit(1)

def choose_subtitle_entry(entries):
    labels = [entry["label"] for entry in entries]
    choice = xbmcgui.Dialog().select(_(30010), labels)
    return entries[choice] if choice >= 0 else None

def with_labels(entries):
    def label(entry):
        if entry["type"] == "external":
            return {**entry, "label": os.path.basename(entry["path"])}
        lang = entry["language"]
        name = get_language_label(lang, kodi_lang)
        return {**entry, "label": f"[MKV] {name} ({lang})"}
    return list(map(label, entries))


def main():
    selected_path = resolve_path()

    if not selected_path:
        notify_error(30008) 
        sys.exit(1)

    if selected_path.lower().endswith(".srt"):
        run_translation(selected_path)
        return

    entries = with_labels(subtitle_sources.list_available_subtitles(selected_path))

    if not entries:
        xbmcgui.Dialog().notification(_(30006), _(30009), xbmcgui.NOTIFICATION_INFO, 3000)
        return

    selected = choose_subtitle_entry(entries)
    if not selected:
        return

    if selected["type"] == "external":
        handle_external_subtitle(selected["path"])
    else:
        handle_embedded_subtitle(selected["video_path"], selected["index"])

if __name__ == "__main__":
    main()
