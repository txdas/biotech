
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
#
from selenium.webdriver.common.keys import Keys
import json
import img2pdf


def png_to_pdf(png_files, pdf_file):
    pdf_data = img2pdf.convert(png_files)
    with open(pdf_file, "wb") as fp:
        fp.write(pdf_data)


chrome_options = webdriver.ChromeOptions()
settings = {
    "recentDestinations": [{
        "id": "Save as PDF",
        "origin": "local",
        "account": ""
    }],
    "selectedDestinationId": "Save as PDF",
    "version": 2,
    "isHeaderFooterEnabled": False,
    "isCssBackgroundEnabled": True,
    "mediaSize": {
        # "height_microns": 297000,
        # "name": "ISO_A4",
        # "width_microns": 210000,
        # "custom_display_name": "A4"
    },
}
chrome_options.add_argument('--enable-print-browser')
# chrome_options.add_argument("--headless")
prefs = {
    'printing.print_preview_sticky_settings.appState': json.dumps(settings),
    'savefile.default_directory': 'C:\\Users\\jinya\\Downloads\\ebook'  # 此处填写你希望文件保存的路径,可填写your file path默认下载地址
}
chrome_options.add_argument('--kiosk-printing')
chrome_options.add_experimental_option('prefs', prefs)

driver = webdriver.Chrome()
driver.get("https://ebooks.jd.com/reader/?ebookId=30833796&return_url=%2Flogin&index=0&from=3")

# 等待元素 "id=myDynamicElement" 可见
total = 100
current = -1
pngs = []
while current<=total:
    current += 1
    element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "app"))
    )
    sleep(1)
    if not current:
        element.click()
    else:
        ActionChains(driver).send_keys(Keys.ARROW_RIGHT).perform()
    sleep(1)
    png = f'C:\\Users\\jinya\\Downloads\\ebook\\{current}.png'
    driver.get_screenshot_as_file(png)
    pngs.append(png)
    # driver.execute_script(f'document.title="{current}.pdf";window.print();')
    sleep(1)
png_to_pdf(pngs,"C:\\Users\\jinya\\Downloads\\ebook\\nlp.pdf")
sleep(1)
# 关闭浏览器
driver.quit()

# driver.quit()