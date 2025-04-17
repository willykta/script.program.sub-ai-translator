import xbmcaddon
import xbmcgui
import os
import sys

addon_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(addon_dir, "core"))
sys.path.insert(0, os.path.join(addon_dir, "api"))

from core.translation import translate_subtitles, estimate_cost
import api.mock as mock
import api.openai as openai

addon = xbmcaddon.Addon("script.program.sub-ai-translator")
lang = addon.getSetting("target_lang")
use_mock = addon.getSettingBool("use_mock")
api_key = addon.getSetting("api_key")
model = "gpt-3.5-turbo"
_ = addon.getLocalizedString

if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    srt_path = sys.argv[1]
else:
    srt_path = xbmcgui.Dialog().browse(1, _(30000), "files", ".srt")
    if not srt_path:
        xbmcgui.Dialog().notification(_(30000), _(30001), xbmcgui.NOTIFICATION_INFO, 3000)
        exit()

est = estimate_cost(srt_path, lang)

if not xbmcgui.Dialog().yesno(
    _(30002),
    _(30003).format(tokens=est["tokens"], usd=est["usd"])
):
    xbmcgui.Dialog().notification(_(30000), _(30001), xbmcgui.NOTIFICATION_INFO, 3000)
    exit()

call_fn = mock if use_mock else openai

progress = xbmcgui.DialogProgress()
progress.create(_(30000), "…")

def report_progress(idx, total):
    percent = int(100 * idx / total)
    progress.update(percent, f"{_(30000)}: {idx} / {total}")

def check_cancelled():
    return progress.iscanceled()

try:
    out_path = translate_subtitles(
        srt_path,
        api_key,
        lang,
        model,
        call_fn,
        report_progress=report_progress,
        check_cancelled=check_cancelled
    )
    progress.close()
    xbmcgui.Dialog().notification(_(30004), _(30005).format(filename=os.path.basename(out_path)), xbmcgui.NOTIFICATION_INFO, 5000)
except Exception as e:
    progress.close()
    xbmcgui.Dialog().notification(_(30006), str(e), xbmcgui.NOTIFICATION_ERROR, 5000)
