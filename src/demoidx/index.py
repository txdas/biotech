import requests
from lxml import etree
import os
import re
from ete3 import Tree, TreeStyle

proxies = {'https': 'http://127.0.0.1:33210'
}
cookies = '''_ga=GA1.1.527904775.1735779639; user_uuid=e02ab1fe-6232-4e3e-9058-28cc19ec273e; search_history=[%22JUQ-337%22%2C%22JUQ-471%22%2C%22JUQ-400%22%2C%22JUL-740%22%2C%22JUL-676%22%2C%22JUL-395%22]; cf_clearance=oM_scD8aGK458L1uiWV_iBLllzzO4VrywIcO5DQMjNU-1735919462-1.2.1.1-VGhYv10TlnkzX7NFe56DulwHen_joQ5SnzeYWc4ALM5zy2nQsSYv4GGgIGINXNQiHjLdflBmd1zP4RD4Xd09r0gDpQ_ld7LUP_aithGx1nvLRfpYygjwuLL7YjlzB5Jwu4PQH.rIMVyNSpNXzwuV0Lo3VAS8hURRtft.SAx57eQKbI5WjNaSP7FdvJ6hkNt5AyobaVFDLGQgHOia3G3Satu4vIzVrACpjPG_J5NamgOxqiCn8IKVSCIinB.paJflPQjsajuO0FqHHNuW62I7kxnq6ap0K9FCbDGZvFgubVMbYtpXYFyvcZr2e6HAMN8d9aZlCPpsg.w4049UNQDReKW6LcMA9y_z_MsaobfDEUqOTw0AmwVgaERjBz2ShbuvJ5Sako9RJgkSQK80CGOeoQ; _ga_Z3V6T9VBM6=GS1.1.1735919063.5.1.1735919466.0.0.0; XSRF-TOKEN=eyJpdiI6ImF2WjFYVit4VGoyNHNNRDRjZkxvZXc9PSIsInZhbHVlIjoiWTJiWkdKOGllN2p0dVcrK2dnQXM3K1ZPMXM5ZHFSZ0paVWVSMHV1YnpKL09xQURYZE1yRWRmbUlqL2x0WmZiOEJLaFpHRjdyY1FlaVBsSkl4WDZ4UGhmSjdodUxJRkQ5UGxFUGpFejRvQ09EL3VSNTVHZUFBUUJHM05kQjNkdmUiLCJtYWMiOiI2OWZlZjIyMTQzZTc0NTAwYjI5MTcyYzhmMDEwMDJiNGM0MmNiM2E2ZDM5MWY1ZTY4MzQ3NjBjY2E4NzRhMGNhIiwidGFnIjoiIn0%3D; missav_session=eyJpdiI6Imd1OStQQ01HUFBRTmVLUFh2L0drMVE9PSIsInZhbHVlIjoiM2d0aE5iS1d5UTVybzhHZVl4WmlVaTEwRWorVk02QmMzNzVBVGUxZ2pENTk1L2QrV2JTK256RUNWa2NJeFdIeERkK2krVjd2Y1dFb1FFM2NSUDVZbWFxR25YSnNPd2pIdW9Yby9PZ2gvcmJiNnl6TnI2VDRsdWc0SjE0NmtYS3QiLCJtYWMiOiJjZWY1MWFjMDg3NWYyMzc2OGM1N2VkMjBmNmZhMzJhZjdjNWMwZWFiODAyNDk0Y2JlYWNiZWVhMDFhNzc2NzhlIiwidGFnIjoiIn0%3D; yKLuOLTOX6GdB6LWWDal81opMKB61o6UVFW4i8Sz=eyJpdiI6ImE5S3dwcERSQUdzUFU4cHpTaDA1TWc9PSIsInZhbHVlIjoiQlVsTGFDbTJCWDVkNHgyZVI3T0N2TE9obEI0S08vd201NGl5cUxMMmFqUWFXU21HNXdnZjIxeVY5c2lNMEZBUDgzQ2xKL29Cc081Nlh1a1NSNGNtZmEyMldpbktHTjhnbU9nWlBOQTdIS1lzYy9vci9RcWVDMVdLQ1gyRlErVWhPYkVidDM3QXl3VkcxSkdzQ2FNT3VTdzZja1FCbHNEbXZ6RnBHaVhaRlZRWjhQMGF4NFdqRmxueTNNRGZsSmVtZE1hc201THgvRVZNVHVZNG41Mmg0RGkvWVFoUGk5Vnc2bDdLUlVYSUFmRGk0Y0owR1o3ZVBhUWZxZkZDVy9PUkpUME8zRWNUOUQ0T2JRbmdFUXJ0KzB2cnFMNHdibVRuY1hBaEg1WGdFb1Z4RUdLNEtSUkNIQjRLMDJkNnh2b3hTS01kVXk3M2MreFcvNy9vdnBlSnJIU3p5Q29BN1RjUzdxZXdsR0JxbXJOcmdnQXBvNHJpY0lDRWc1K0dxZklTSTlIaCt5dG81aFg1UitIaXRDb2hRdz09IiwibWFjIjoiZDBiMDcyNWZhYzYwMzUyNzBlM2ZkMmQ1MzNjZWM4NWIxMjY5YjE1Y2NiNDhhY2MxYzc5N2U2MDIzODVjZDZlNyIsInRhZyI6IiJ9'''
headers = {
    "cookie":cookies,
    "cache-control":"max-age=0",
    "accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language":"zh-CN,zh;q=0.9",
    "accept-encoding":"gzip, deflate, br, zstd",
    "Accept": "text/html,application/xhtml+xml,application/xml",
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36"
}


def _fetch(url, xpath, extra=None):
    res = requests.get(url, headers=headers,proxies=proxies)
    print(url, res)
    if res.status_code <= 400:
        res.encoding = 'utf-8'
        html = res.content
        selector = etree.HTML(html)  # etree.HTML(源码) 识别为可被xpath解析的对象
        nodes = selector.xpath(xpath)
        return nodes[0] if nodes and len(nodes) else ""


def get_title(code):
    url = f"https://123av.com/zh/v/{code}"
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


def vindex(fpath="/Volumes/Extreme SSD/nuit/jul"):
    for root, dirs, files in os.walk(fpath):
        findex = os.path.join(root, "index.txt")
        titles, caches = [], {}
        if os.path.exists(findex):
            with open(findex) as fp:
                for line in fp:
                    values = line.split()
                    if len(values)>1:
                        key, value = line.split(maxsplit=1)
                        caches[key] = line.strip()
        for f in sorted(files):
            fname = os.path.join(root,f)
            if f.endswith(".mp4") and not f.startswith(".") and not os.path.islink(fname):
                name = f.split(".")[0]
                name = name.upper().strip()
                title = get_title(name) if name not in caches else caches[name]
                if title:
                    title = re.sub("在线观看,|- 123AV", "", title).strip()
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
    # print(get_title("JUQ-614"))
    # vindex(fpath=f"/Volumes/Extreme SSD/nuit/stars")
    # for name in ["fsdss"]:
    #     vindex(fpath=f"/Volumes/Extreme SSD/nuit/{name}")
    # find(key="车", fpath="/Volumes/Extreme SSD/nuit/") #
    # symlink(fpath="/Volumes/PortableSSD/name",target="/Volumes/PortableSSD/nuit/")
    symlink()








