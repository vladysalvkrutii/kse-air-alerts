"""Download the pinned dataset snapshot from GitHub (no token needed)."""
import urllib.request

from . import config


def download(force: bool = False) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.RAW_DIR / config.DATA_FILE
    sha_file = config.RAW_DIR / "SHA"
    if dest.exists() and sha_file.exists() and sha_file.read_text().strip() == config.DATA_SHA and not force:
        print(f"[download] {dest.name} already at {config.DATA_SHA[:10]}, skipping")
        return
    url = (f"https://raw.githubusercontent.com/{config.DATA_REPO}/"
           f"{config.DATA_SHA}/datasets/{config.DATA_FILE}")
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, dest)
    sha_file.write_text(config.DATA_SHA)
    print(f"[download] saved {dest} ({dest.stat().st_size:,} bytes)")
