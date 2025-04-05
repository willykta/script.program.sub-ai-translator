import xbmcaddon
import xbmcgui
import os
import sys

addon_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(addon_dir, "lib"))
sys.path.insert(0, os.path.join(addon_dir, "translator"))
sys.path.insert(0, os.path.join(addon_dir, "api"))

import translator.translator as translator
import api

addon = xbmcaddon.Addon()
lang = addon.getSetting("target_lang")
use_mock = addon.getSettingBool("use_mock")
api_key = addon.getSetting("api_key")
model = "gpt-3.5-turbo"

srt_path = xbmcgui.Dialog().browse(1, "Wybierz plik SRT", "files", ".srt")
if not srt_path:
    xbmcgui.Dialog().notification("Tłumaczenie napisów", "Anulowano", xbmcgui.NOTIFICATION_INFO, 3000)
    exit()

call_fn = api.mock if use_mock else api.openai

try:
    out_path = translator.translate_subtitles(srt_path, api_key, lang, model, call_fn)
    xbmcgui.Dialog().notification("Gotowe", f"Zapisano: {os.path.basename(out_path)}", xbmcgui.NOTIFICATION_INFO, 5000)
except Exception as e:
    xbmcgui.Dialog().notification("Błąd", str(e), xbmcgui.NOTIFICATION_ERROR, 5000)
