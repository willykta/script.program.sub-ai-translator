# 🧠 Subtitle AI Translator (Kodi Add-on)

**Subtitle AI Translator** is a smart, user-friendly **Kodi add-on** that enables you to **translate subtitle files using Large Language Models (LLMs)** — currently supporting **OpenAI's GPT models**.

It’s especially useful for users who want to enjoy movies and shows with subtitles in their preferred language, with preserved formatting and natural, fluent translations.

> ⚠️ This add-on is **experimental** and provided **as-is**. Use at your own risk.

---

## 🎯 Features

- 🔤 Translate `.srt` subtitle files (other formats planned)
- 🤖 Uses **OpenAI ChatGPT models** (e.g., `gpt-3.5-turbo`)
- 🧪 Mock backend for **offline testing** (no token usage)
- 📂 Context menu support: translate subtitles from `.srt` files or folders containing video files
- 🔧 Configurable:
  - Target language (predefined or custom)
  - OpenAI model and API key
  - Token price estimation
  - Parallel request control (advanced setting)
- 📈 Live token cost estimation before translation
- 📊 Progress bar with cancel option

---

## 🧰 Requirements

- ✅ Kodi 20+
- ✅ OpenAI account and **API key**
  - Get yours here: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
- ⚠️ You are responsible for your **own API usage and associated costs**.

---

## 📸 Screenshots

### Configuration – Language Selection
![Language Configuration](resources/screenshots/configuration_langugage.png)

### Configuration – Model and OpenAI Key
![Model Configuration](resources/screenshots/configuration_model.png)

### Translate from File Selector
![File Selector](resources/screenshots/translate_file_selector.png)

### Translate from Context Menu
![Context Menu](resources/screenshots/translate_context_menu.png)

### Estimated Cost Dialog
![Cost Estimation](resources/screenshots/cost_estimation.png)

---

## 🚀 Installation

1. [Download the latest `.zip` release](https://github.com/re999/script.program.sub-ai-translator/releases)
2. In Kodi:
   - Go to **Add-ons → Install from zip file**
   - Select the downloaded `.zip`
3. Open the add-on via:
   - **Program Add-ons → Subtitle AI Translator**
   - Or by right-clicking a video or subtitle file → **Translate subtitles**

---

## ⚙️ Configuration

Accessible via **Add-on Settings**:

| Setting | Description |
|--------|-------------|
| **Target Language** | Choose a predefined language or enter a custom one |
| **Model** | Select from supported OpenAI models |
| **API Key** | Paste your OpenAI API key here |
| **Price per 1000 tokens** | Used for cost estimation |
| **Parallel Requests** | Advanced setting to control performance |
| **Mock Backend** | Use fake responses for testing (no real API calls) |

---

## 💡 Roadmap

- [ ] Pause/resume translations
- [ ] Support for subtitle extraction from video files (e.g., `.mkv`)
- [ ] Additional LLM backends (Mistral, Claude, local models, etc.)
- [ ] GUI improvements and setup wizard
- [ ] Format support beyond `.srt` (e.g., `.ass`, embedded subtitles)

---

## 🤝 Support This Project

If you find this add-on useful:

- 🌟 Star it on GitHub
- 🧡 [Sponsor via BuyMeACoffee](https://buymeacoffee.com/re999)
- 🐛 Report bugs or suggest features

---

## 📜 License

This project is licensed under the **MIT License**, see the `LICENSE` file for details, with the following addition:

> 🧾 **Disclaimer**: You are fully responsible for any costs incurred by using this add-on. The author is not liable for API charges or misuse. Use at your own risk.

---

**© 2025 by re999**
