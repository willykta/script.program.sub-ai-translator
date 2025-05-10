from babel.core import Locale
from babel import localedata

def get_language_label(lang_code: str, display_locale: str = "en") -> str:
    try:
        if not localedata.exists(display_locale):
            display_locale = "en"
        loc = Locale.parse(display_locale)
        parsed = Locale.parse(lang_code)
        label = parsed.get_display_name(loc)
        return label[:1].upper() + label[1:]
    except Exception:
        return "Unknown language"
