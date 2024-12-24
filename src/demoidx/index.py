import requests
from lxml import etree
import os
from ete3 import Tree, TreeStyle

proxies = {
    'http': 'http://127.0.0.1:33210',
    'https': 'http://127.0.0.1:33210',
    'all_proxy': 'socks5://127.0.0.1:33210'
}
headers = {
    "if-modified-since":"Wed, 04 Dec 2024 11:03:27 GMT",
    "priority":"u=0, i",
    "cookie":"user_uuid=cff7fa10-0bd8-4d9f-9aa5-30cb81a9b8d5; _ga=GA1.1.1603870572.1710306902;",
    "cache-control":"max-age=0",
    "accept-language":"zh-CN,zh;q=0.9",
    "accept-encoding":"gzip, deflate, br, zstd",
    "Accept": "text/html,application/xhtml+xml,application/xml",
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 "
}


def _fetch(url, xpath, extra=None):
    res = requests.get(url, headers=headers, proxies=proxies)
    # print(url, res)
    if res.status_code <= 400:
        res.encoding = 'utf-8'
        html = res.content
        selector = etree.HTML(html)  # etree.HTML(源码) 识别为可被xpath解析的对象
        nodes = selector.xpath(xpath)

        return nodes[0] if nodes and len(nodes) else ""


def get_title(code):
    url = f"https://missav.com/dm35/cn/{code}"
    xpath = "/html/head/meta[@property='og:title']"
    node = _fetch(url, xpath)
    title = node.attrib.get("content", "") if node is not None and hasattr(node,"attrib")  else ""
    if title:
        return title
    else:
        url = f"https://jable.tv/videos/{code}/?lang=zh"
        xpath = "/html/head/title/text()"
        title = _fetch(url, xpath)
        if title:
            title = title.encode('iso-8859-1').decode("utf-8")
            return title
        url = f"https://missav.com/cn/search/{code}"
        xpath = "/html/body/div[1]/div[3]/div[2]/a[@class='text-secondary group-hover:text-primary']/text()"
        title = _fetch(url, xpath)
        return title


def vindex(fpath="/Volumes/Extreme SSD/name/白石桃"):
    for root, dirs, files in os.walk(fpath):
        findex = os.path.join(root, "index.txt")
        titles, caches = [], {}
        # if os.path.exists(findex):
        #     with open(findex) as fp:
        #         for line in fp:
        #             values = line.split()
        #             if len(values)>1:
        #                 key, value = line.split(maxsplit=1)
        #                 caches[key] = value
        for f in sorted(files):
            fname = os.path.join(root,f)
            if f.endswith(".mp4") and not f.startswith(".") and not os.path.islink(fname):
                name = f.split(".")[0]
                name = name.upper().strip()
                title = get_title(name) if name not in caches else caches[name]
                print(title, name)
                if title:
                    titles.append(title)
        print(root)
        with open(findex, "w") as wp:
            for title in sorted(titles):
                wp.write(title+"\n")


def find(key, fpath="/Users/john/Downloads"):
    # wp = open("./index.txt","w")
    for root, dirs, files in os.walk(fpath):
        findex = os.path.join(root, "index.txt")
        if not os.path.exists(findex):
            continue

        with open(findex) as fp:
            for line in fp:

                if key in line:
                    # print(findex)
                    print(line.strip())
                    # wp.write(line.strip()+"\n")


def symlink(fpath="/Volumes/Extreme SSD/name", target="/Volumes/Extreme SSD/nuit/"):
    fnames = {}
    for root, dirs, files in os.walk(target):
        for f in files:
            kname = f.split(".")[0].strip().upper()
            src = os.path.join(root, f)
            if not f.startswith("."):
                fnames[kname] = src
    for root, dirs, files in os.walk(fpath):
        for f in files:
            if f == "index.txt":
                findex = os.path.join(root,f)
                with open(findex) as fp:
                    for line in fp:
                        if not line.strip():
                            continue
                        kname = line.strip().split()[0].upper()
                        link = os.path.join(root, f"{kname}.mp4")
                        if kname in fnames:
                            src = fnames[kname]
                            if not os.path.exists(link):
                                print(src, link)
                                os.symlink(src, link)


if __name__ == '__main__':
    print(get_title("JUQ-614"))
    # vindex(fpath=f"/Volumes/Extreme SSD/nuit")
    # for name in ["fsdss"]:
    #     vindex(fpath=f"/Volumes/Extreme SSD/nuit/{name}")
    # find(key="凪ひかる", fpath="/Volumes/Extreme SSD/nuit") #
    # symlink(fpath="/Volumes/PortableSSD/name",target="/Volumes/PortableSSD/nuit/")
    # symlink()








