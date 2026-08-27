## کوچینگ خصوصی و مصاحبهٔ آزمایشی AIMLInterviews

من برای مهندسان AI/ML، دانشمندان کاربردی AI، مهندسان پژوهش، دانشمندان پژوهشی، استراتژیست‌های AI، مدیران مهندسی و رهبران ارشد AI، **جلسه‌های خصوصی کوچینگ و مصاحبهٔ آزمایشی AI/ML** برگزار می‌کنم.

موضوع جلسه‌ها می‌تواند شامل طراحی سیستم AI/ML، مبانی هوش مصنوعی مولد و هوش مصنوعی عامل‌محور، مبانی ML، کدنویسی AI، مصاحبهٔ رفتاری و مصاحبهٔ رهبری باشد.

اطلاعات بیشتر: [https://aimlinterviews.io](https://aimlinterviews.io)

---

<p align="center">
<img width="720" src="src/imgs/cover.png">
</p>

[English](README.md) | [简体中文](README-CN.md) | فارسی

[![مجوز: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![سبک کد: black](https://img.shields.io/badge/code%20style-black-000.svg)](https://github.com/psf/black) [![GitHub stars](https://img.shields.io/github/stars/alirezadir/AIMLInterviews?style=social)](https://github.com/alirezadir/AIMLInterviews/stargazers) [![GitHub forks](https://img.shields.io/github/forks/alirezadir/AIMLInterviews?style=social)](https://github.com/alirezadir/AIMLInterviews/network) [![آخرین commit](https://img.shields.io/github/last-commit/alirezadir/AIMLInterviews)](https://github.com/alirezadir/AIMLInterviews/commits/main) [![GitHub issues](https://img.shields.io/github/issues/alirezadir/AIMLInterviews)](https://github.com/alirezadir/AIMLInterviews/issues) [![مشارکت‌کنندگان](https://img.shields.io/github/contributors/alirezadir/AIMLInterviews)](https://github.com/alirezadir/AIMLInterviews/graphs/contributors)

# مصاحبه‌های هوش مصنوعی و یادگیری ماشین :robot:

> **دربارهٔ ترجمه:** این نسخه برای جامعهٔ فنی فارسی‌زبان بومی‌سازی شده است. اصطلاحات رایج مهندسی، نام مدل‌ها، APIها، کد، فرمول‌ها و URLها در صورت لزوم به انگلیسی باقی می‌مانند. [راهنمای ترجمه و واژه‌نامه](fa/TRANSLATION-GUIDE.md) مبنای یکدستی این نسخه است.

این مخزن راهنمایی برای آمادگی **مصاحبه‌های فنی هوش مصنوعی و یادگیری ماشین** در شرکت‌های بزرگ فناوری، به‌ویژه FAANG، است. مطالب آن از تجربهٔ شخصی نویسنده و یادداشت‌های دوران آمادگی برای مصاحبه جمع‌بندی شده‌اند. نویسنده در سال ۲۰۲۰ هم‌زمان از Meta برای نقش ML Specialist، از Google برای ML Engineer، از Amazon و Apple برای Applied Scientist و از Roku برای ML Engineer پیشنهاد همکاری دریافت کرد. او در سال ۲۰۲۵ نیز برای نقش AI Tech Lead از Amazon و Apple پیشنهاد گرفت.

> **به یاد داشته باشید:** مصاحبه‌دادن یک مهارت است؛ هرچه این مهارت را بیشتر تمرین کنید، احتمال رسیدن به نتیجهٔ بهتر بیشتر می‌شود.

مراحل زیر رایج‌ترین بخش‌های مصاحبه برای نقش‌های فنی ML در شرکت‌های مختلف هستند. در هر فصل، روش آماده‌شدن برای یکی از این مراحل را بررسی می‌کنیم.

| فصل | محتوا |
| --- | --- |
| فصل ۱ | [کدنویسی عمومی ـ DSA (ساختمان داده و الگوریتم)](fa/src/lc-coding.md) |
| فصل ۲ | [کدنویسی ML و داده](fa/src/MLC/ml-coding.md) |
| فصل ۳ | [مبانی و گسترهٔ ML؛ از ML کلاسیک تا LLM و هوش مصنوعی چندوجهی](fa/src/ml-fundamental.md) |
| فصل ۴ | [طراحی سیستم ML / هوش مصنوعی مولد / LLM](fa/src/MLSD/ml-system-design.md) |
| فصل ۵ | [سیستم‌های هوش مصنوعی عامل‌محور](https://github.com/alirezadir/Agentic-AI-Systems.git) |
| فصل ۶ | [مصاحبه‌های رفتاری](fa/src/behavioral/behavior.md) · [فایل تمرین در Google Sheets](https://docs.google.com/spreadsheets/d/1W8H2DMzetOt2BxCTmENOgdfBS84Kf_mbXIHLF2-sP-M/edit?gid=244760119#gid=244760119) · [دانلود Excel](src/behavioral/Behavioral%20%26%20Leadership%20Interview%20Prep%20Template.xlsx) |
| منابع | [منابع یادگیری هوش مصنوعی مولد](fa/src/genai-resources.md) |
| مربی AI | [سرور MCP پروژهٔ AIMLInterviews](MCP/README.md) |

## مربی مصاحبه با MCP

با `aimlinterviews-mcp` می‌توانید هر دستیار AI سازگار با MCP را به مربی مصاحبهٔ AI/ML تبدیل کنید. این ابزار مسئله‌های موجود در curriculum را پیدا می‌کند، راهنمایی مرحله‌ای می‌دهد، برنامهٔ یادگیری و آمادگی برای شرکت‌ها می‌سازد و پاسخ‌ها را بدون لو دادن زودهنگام راه‌حل بررسی می‌کند.

```bash
claude mcp add aimlinterviews -- npx -y aimlinterviews-mcp
# or
codex mcp add aimlinterviews -- npx -y aimlinterviews-mcp
```

ابتدا این مخزن را clone کنید. سپس دستور را از پوشهٔ clone اجرا کنید یا متغیر `AIMLINTERVIEWS_ROOT` را تنظیم کنید. جزئیات پیکربندی، ابزارها و توسعه در [راهنمای سرور MCP](MCP/README.md) آمده است.

## تازه‌ها

:newspaper: نام این مخزن اکنون **AIMLInterviews** است. نسخهٔ ۲۰۲۶ با مطالب گسترده‌تر دربارهٔ LLM، هوش مصنوعی چندوجهی، post-training و طراحی سیستم هوش مصنوعی مولد به‌روزرسانی شده است.

**نکته‌ها:**

- ساختار مصاحبه‌های AI و ML در همهٔ شرکت‌ها یکسان نیست، اما مراحل اصلی در شرکت‌های FAANG شباهت زیادی دارند. مصاحبهٔ استارتاپ‌ها معمولاً به کاربردها و مسئله‌های همان شرکت نزدیک‌تر است، در حالی که شرکت‌های بزرگ ساختار باثبات‌تری دارند.
- تمرکز اصلی این راهنما نقش‌های AI/ML Engineering، Applied Science و Tech Lead در شرکت‌های بزرگ است. ساختار مصاحبه برای نقش‌هایی مانند Data Science یا دانشمند پژوهشی متفاوت است، اما بخشی از فصل‌های این راهنما همچنان برای آن‌ها مفید خواهد بود.

# مشارکت

از بازخورد و مشارکت شما استقبال می‌کنیم :blush:

اگر پیشنهادی برای بهترشدن محتوا یا ترجمه دارید، لطفاً pull request باز کنید. برای تغییر ترجمه، ابتدا [راهنمای سبک و واژه‌نامهٔ فارسی](fa/TRANSLATION-GUIDE.md) را بخوانید.
