"""Take screenshots of the running app for the README (desktop and phone widths).

Start the app first:  streamlit run app/streamlit_app.py --server.port 8611
Then:                 python scripts/screenshots.py [--url http://localhost:8611] [--out docs/screenshots]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def wait_ready(page, text=None, timeout=60):
    page.wait_for_selector('[data-testid="stApp"]', timeout=timeout * 1000)
    t0 = time.time()
    while time.time() - t0 < timeout:
        running = page.locator('[data-testid="stStatusWidget"]').count()
        if not running and (text is None or page.get_by_text(text).count()):
            break
        time.sleep(0.5)
    time.sleep(1.5)


def click_tab(page, name):
    page.get_by_role("tab", name=name).click()
    time.sleep(2.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8611")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "screenshots")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        desk = browser.new_page(viewport={"width": 1320, "height": 900}, device_scale_factor=1.5)
        desk.goto(args.url)
        wait_ready(desk, "Other possibilities")
        desk.screenshot(path=str(args.out / "classify_desktop.png"), full_page=True)

        # A questionable image, to show the warning state
        desk.get_by_text("Images the app should question").click()
        time.sleep(2)
        wait_ready(desk)
        desk.screenshot(path=str(args.out / "unfamiliar_desktop.png"), full_page=True)

        for tab, name in [("Differential count", "differential"), ("Model performance", "performance"),
                          ("About and limits", "about"), ("Session history", "history")]:
            click_tab(desk, tab)
            wait_ready(desk)
            desk.screenshot(path=str(args.out / f"{name}_desktop.png"), full_page=True)

        phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True)
        phone.goto(args.url)
        wait_ready(phone, "Other possibilities")
        phone.screenshot(path=str(args.out / "classify_phone.png"), full_page=True)
        click_tab(phone, "Differential count")
        wait_ready(phone)
        phone.screenshot(path=str(args.out / "differential_phone.png"), full_page=True)
        browser.close()
    print("Saved screenshots to", args.out)


if __name__ == "__main__":
    main()
