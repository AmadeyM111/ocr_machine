import requests

url = "url_for_downloading_big_files"
with requests.get(url, stream=True, verify=False)
    r.rise_for_status()
    with open("file.bin", "wb") as f:
        for chunk in r.iter_content(64 * 1024):
            f.write(chunk)
