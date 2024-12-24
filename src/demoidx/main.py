if __name__ == '__main__':
    import pyhttpx

    proxies = {
        'http': 'http://127.0.0.1:33210',
        'https': 'http://127.0.0.1:33210',
        'all_proxy': 'socks5://127.0.0.1:33210'
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
    }
    session = pyhttpx.HttpSession()
    res = session.get(url='https://missav.com/dm35/cn/JUQ-614', headers=headers, proxies=proxies)
    print(res.text)