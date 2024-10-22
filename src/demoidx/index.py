import requests
from lxml import etree
import os
from ete3 import Tree, TreeStyle

proxies = {
    'http': 'http://127.0.0.1:33210',
    'https': 'http://127.0.0.1:33210'
}
headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.66 Safari/537.36"
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
    url = f"https://missav.com/dm46/cn/{code}"
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
        if os.path.exists(findex):
            with open(findex) as fp:
                for line in fp:
                    values = line.split()
                    if len(values)>1:
                        key, value = line.split(maxsplit=1)
                        caches[key] = value
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
    # print(get_title("020"))
    # vindex(fpath=f"/Volumes/Extreme SSD/cache")
    # for name in ["fsdss"]:
    #     vindex(fpath=f"/Volumes/Extreme SSD/nuit/{name}")
    # find(key="凪ひかる", fpath="/Volumes/Extreme SSD/nuit") #
    # symlink(fpath="/Volumes/PortableSSD/name",target="/Volumes/PortableSSD/nuit/")
    symlink()








