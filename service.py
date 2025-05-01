import xbmcaddon
import xbmcgui
import os
import sys

addon_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(addon_dir, "core"))
sys.path.insert(0, os.path.join(addon_dir, "api"))

from core.translation import translate_subtitles
from core.estimation import estimate_cost
from core import settings

addon = xbmcaddon.Addon("script.program.sub-ai-translator")
_ = addon.getLocalizedString
cfg = settings.get()
call_fn = settings.get_call_fn()

if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    srt_path = sys.argv[1]
else:
    srt_path = xbmcgui.Dialog().browse(1, _(30000), "files", ".srt")
    if not srt_path:
        xbmcgui.Dialog().notification(_(30000), _(30001), xbmcgui.NOTIFICATION_INFO, 3000)
        exit()

est = estimate_cost(srt_path, cfg["lang"], cfg["price_per_1000_tokens"])

if not xbmcgui.Dialog().yesno(
    _(30002),
    _(30003).format(tokens=est["tokens"], usd=est["usd"]) + f"\n({cfg['provider']})"
):
    xbmcgui.Dialog().notification(_(30000), _(30001), xbmcgui.NOTIFICATION_INFO, 3000)
    exit()

progress = xbmcgui.DialogProgress()
progress.create(f"{_(30000)} ({cfg['provider']})", "…")

def report_progress(idx, total):
    percent = int(100 * idx / total)
    progress.update(percent, f"{_(30000)}: {percent}%")

def check_cancelled():
    return progress.iscanceled()

try:
    out_path = translate_subtitles(
        srt_path,
        cfg["api_key"],
        cfg["lang"],
        cfg["model"],
        call_fn,
        report_progress=report_progress,
        check_cancelled=check_cancelled,
        parallel=cfg["parallel"]
    )
    progress.close()
    xbmcgui.Dialog().notification(
        _(30004),
        _(30005).format(filename=os.path.basename(out_path)),
        xbmcgui.NOTIFICATION_INFO,
        5000
    )
except Exception as e:
    progress.close()
    import traceback
    xbmc.log(f"[Sub-AI Translator] Exception: {e}", level=xbmc.LOGERROR)
    xbmc.log(traceback.format_exc(), level=xbmc.LOGERROR)
    xbmcgui.Dialog().notification(_(30006), str(e), xbmcgui.NOTIFICATION_ERROR, 5000)
