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
    if res.status_code <= 400:
        res.encoding = 'utf-8'
        html = res.content
        selector = etree.HTML(html)  # etree.HTML(源码) 识别为可被xpath解析的对象
        nodes = selector.xpath(xpath)
        return nodes[0]


def get_title(code):
    url = f"https://jable.tv/videos/{code}/?lang=zh"
    xpath = "/html/head/title/text()"
    title = _fetch(url, xpath)
    if title:
        title = title.encode('iso-8859-1').decode("utf-8")
        title = title.split(" - ")[0]
        return title
    else:
        url = f"https://missav.com/dm46/cn/{code}"
        title = _fetch(url, xpath)
        if title:
            return title
        url = f"https://missav.com/cn/search/{code}"
        xpath = "/html/body/div[1]/div[3]/div[2]//a[@class='text-secondary group-hover:text-primary']"
        title = _fetch(url, xpath)
        return title


def vindex(fpath="/Users/john/Downloads"):
    for root, dirs, files in os.walk(fpath):
        findex = os.path.join(root, "index.txt")
        titles = []
        for f in sorted(files):
            fname = os.path.join(root,f)
            if f.endswith(".mp4") and not f.startswith(".") and not os.path.islink(fname):
                name = f.split(".")[0]
                name = name.upper().strip()
                title = get_title(name)
                print(f, title, name)
                if title:
                    titles.append(title)
        print(root)
        with open(findex, "w") as wp:
            for title in sorted(titles):
                wp.write(title+"\n")


def find(key, fpath="/Users/john/Downloads"):
    wp = open("./index.txt","w")
    for root, dirs, files in os.walk(fpath):
        findex = os.path.join(root, "index.txt")
        if not os.path.exists(findex) or "name" in root:
            continue
        with open(findex) as fp:
            for line in fp:
                if key in line:
                    wp.write(line.strip()+"\n")


def symlink(fpath="/Volumes/Extreme SSD/name/枫ふうあ", target="/Volumes/Extreme SSD/nuit/"):
    fnames = {}
    for root, dirs, files in os.walk(fpath):
        for f in files:
            if f == "index.txt":
                findex = os.path.join(root,f)
                with open(findex) as fp:
                    for line in fp:
                        if line.strip():
                            fname = line.strip().split()[0].upper()
                            fnames[fname]=root
    for root, dirs, files in os.walk(target):
        for f in files:
            kname = f.split(".")[0].strip().upper()
            src = os.path.join(root, f)
            if kname in fnames and not f.startswith(".") and not os.path.islink(src):
                link = os.path.join(fnames[kname], f)
                if not os.path.exists(link):
                    os.symlink(src, link)






if __name__ == '__main__':
    vindex(fpath="/Volumes/Extreme SSD/nuit/miaa")
    # find(key="枫",fpath="/Volumes/Extreme SSD/nuit")
    # symlink()
    # print(get_title("MIAA-019"))








