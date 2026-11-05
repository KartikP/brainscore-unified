"""Exercise the exported replay in a remote headless browser and retain screenshots."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def main():
    from playwright.sync_api import sync_playwright
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    errors, network = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1100, 'height': 1100}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: network.append(request.url) if request.url.startswith('http') else None)
        page.goto(args.html.resolve().as_uri())
        page.wait_for_selector('body[data-ready="true"]')
        payload = json.loads(page.locator('#replay-data').text_content())
        assert page.locator('#frame').input_value() == '0'
        assert 'Bin 1 / 8' in page.locator('#bin').text_content()
        page.locator('#next').click()
        assert page.locator('#frame').input_value() == '1'
        assert 'Bin 1 / 8' in page.locator('#bin').text_content()
        page.locator('#next').click()
        assert 'Bin 2 / 8' in page.locator('#bin').text_content()
        page.locator('#previous').click()
        assert page.locator('#frame').input_value() == '1'
        page.locator('#frame').evaluate("(slider) => {slider.value=8; slider.dispatchEvent(new Event('input'));}")
        assert 'Bin 5 / 8' in page.locator('#bin').text_content()
        page.locator('#heat').hover(position={'x': 20, 'y': 20})
        assert 'raw' in page.locator('#hover').text_content()
        page.screenshot(path=str(args.output / 'desktop.png'), full_page=True)
        page.locator('#restart').click()
        page.locator('#speed').select_option('.25')
        page.locator('#play').click()
        page.wait_for_function("Number(document.getElementById('frame').value) >= 2")
        page.locator('#play').click()
        assert page.locator('#play').text_content() == 'Play'
        page.locator('#frame').evaluate("(slider) => {slider.value=15; slider.dispatchEvent(new Event('input'));}")
        assert 'Bin 8 / 8' in page.locator('#bin').text_content()
        page.locator('summary').click()
        assert page.locator('#raw').text_content() == payload['raw_answer']
        page.set_viewport_size({'width': 600, 'height': 1100})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(args.output / 'narrow.png'), full_page=True)
        page.set_viewport_size({'width': 1100, 'height': 1100})
        document = args.html.read_text()
        page.set_content('<iframe title="Recorded activity replay" sandbox="allow-scripts" '
                         'style="width:100%;height:1040px;border:0" srcdoc="'
                         + html.escape(document, quote=True) + '"></iframe>')
        embedded = page.frame_locator('iframe')
        embedded.locator('body[data-ready="true"]').wait_for()
        embedded.locator('#next').click()
        embedded.locator('#next').click()
        assert embedded.locator('#frame').input_value() == '2'
        assert 'Bin 2 / 8' in embedded.locator('#bin').text_content()
        page.screenshot(path=str(args.output / 'notebook-embed.png'), full_page=True)
        assert not errors and not network
        report = {'passed': True, 'frame_count': payload['frame_count'],
                  'html_sha256': hashlib.sha256(args.html.read_bytes()).hexdigest(),
                  'browser_version': browser.version, 'sandboxed_iframe': True,
                  'time_bins': len(payload['start_ms']), 'frame_step_alignment': True,
                  'seek_alignment': True, 'play_pause': True, 'speed_control': True,
                  'raw_hover': True, 'answer_preserved': True, 'narrow_layout': True,
                  'page_errors': errors, 'network_requests': network}
        (args.output / 'browser-validation.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        browser.close()


if __name__ == '__main__':
    main()
