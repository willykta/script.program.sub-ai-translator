import sys
import os
import glob
import xbmc
import xbmcgui
import xbmcaddon

addon = xbmcaddon.Addon()
addon_path = addon.getAddonInfo('path')
_ = addon.getLocalizedString

def resolve_path():
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return sys.argv[1]
    
    path = xbmc.getInfoLabel("ListItem.Path")
    filename = xbmc.getInfoLabel("ListItem.FilenameAndPath") or xbmc.getInfoLabel("ListItem.Filename")
    
    if path and filename:
        full_path = os.path.join(path, filename) if not filename.startswith(path) else filename
        if os.path.isfile(full_path):
            return full_path

    return None

selected_path = resolve_path()

if not selected_path:
    xbmcgui.Dialog().notification(_(30006), _(30008), xbmcgui.NOTIFICATION_ERROR, 3000)
    sys.exit(1)

def run_translation(srt_path):
    xbmc.executebuiltin(f'RunScript(script.program.sub-ai-translator, "{srt_path}")')


if selected_path.lower().endswith(".srt"):
    run_translation(selected_path)
else:
    folder = os.path.dirname(selected_path)
    srt_files = sorted(glob.glob(os.path.join(folder, "*.srt")))
    if not srt_files:
        xbmcgui.Dialog().notification(_(30006), _(30009), xbmcgui.NOTIFICATION_INFO, 3000)
        sys.exit(0)

    labels = [os.path.basename(p) for p in srt_files]
    choice = xbmcgui.Dialog().select(_(30010), labels)
    if choice >= 0:
        run_translation(srt_files[choice])
