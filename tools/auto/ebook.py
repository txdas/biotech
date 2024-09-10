from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import img2pdf
import os
import io
import pickle
import PyPDF2


def set_cookies(driver):
    cookies = '''jdu=334161470; shshshfpa=03d0bb82-56a7-4a07-7632-9bdc357b7a99-1694698504; shshshfpx=03d0bb82-56a7-4a07-7632-9bdc357b7a99-1694698504; pinId=jrDO9pkPsws; mba_muid=334161470; shshshfpb=BApXcEXY6YPVAJhWSpOCks8T78quOvctDB9Yogit49xJ1MgV-voO2; 
    __jdv=181111935%7Cbaidu%7C-%7Corganic%7Cnot%20set%7C1725867509640; __utma=122270672.562794796.1725867570.1725867570.1725867570.1; 
    __utmc=122270672; __utmz=122270672.1725867570.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); areaId=19; ipLoc-djd=19-1601-0-0; __jdc=95931165; 3AB9D23F7A4B3C9B=HUX3O3UWLZIHQP2UG6AR4LD4V7ALXAAYGXE3Q6TPZIXSWODQ4WMM3N6OEZAW5AYLXMI57GKGVTYID5VHRSS6CAI5MA; wlfstk_smdl=g2cd2awpvdrr85ccl2w96rfzk0wbrc98; TrackID=1KFYIu9PeOBBtPzDpYlHiBFEHvjSEVvjAmiS3JMAfs6bolwlvNAVrBCVYzC6R8UzfLAjIq78suDK8NJXnVtGPyXMuRcj_eR89JN1ZGVexikI; thor=1520ACE5994ED7BDD186D621346A66F37542A4895AD1539A5EC65B44AFD27A4EF003E19F1589E62FC6A798AFF40D55D6B7EEE20D60FFA9540A33A1385234CCE982868D0E3003AFAA3AF4DA064360CF5B37A35ABDD8076F380716CB96389ABAC8F48F8B2B1A1E2942C55F6CF46F471F3AA150E16F6170CFE202769C0DA37E7916; flash=3_bxA_ZG7N2vEfKMBPy4IS9OpJqasup5_MO1n3nMGEgR_YblAcPm8B7fw3mpYY3MI1C1tiSC0jWOCoGbiNQb7hZVzV0_98p6vf8tXU9nY_YA010Z43f7FUeJIS8WHAglrbL8fcmIyjvtWiFBxAcEp3CEDuihUFYz1BE6-h3_Xi; light_key=AASBKE7rOxgWQziEhC_QY6yawEF4NuTT8ni11EGnToDpgPGJhUBcmZfv1RDu1qR3ArIU-CmG; pin=txdlut; unick=jd_txdl870; ceshi3.com=201; _tp=2eENXDcvByVP984T6qOOZQ%3D%3D; _pst=txdlut; __jda=95931165.334161470.1694698503.1725874680.1725881075.16; __jd_ref_cls=Pc-Web-MyBooks-BookClick; 3AB9D23F7A4B3CSS=jdd03HUX3O3UWLZIHQP2UG6AR4LD4V7ALXAAYGXE3Q6TPZIXSWODQ4WMM3N6OEZAW5AYLXMI57GKGVTYID5VHRSS6CAI5MAAAAAMR22TNY5YAAAAADKZAHOBW6HFYQQX; _giad=1
    '''
    for e in cookies.split(";"):
        k,v = e.split("=",maxsplit=1)
        driver.add_cookie({"name":k.strip(),"value":v.strip()})

def png_to_pdf(png_files, pdf_file):
    pdf_data = img2pdf.convert(png_files)
    with open(pdf_file, "wb") as fp:
        fp.write(pdf_data)


def get_driver(url, waits=15):
    service = webdriver.ChromeService(executable_path="C:\Program Files\Google\Chrome\Application\driver\chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    driver.get(url)
    driver.maximize_window()
    windows = driver.get_window_size()
    driver.set_window_position(windows["width"]+10, 0)
    driver.set_window_size(windows["width"],windows["height"]+300)
    print(driver.get_window_size())

    # WAIT FOR LOGIN
    sleep(waits)
    return driver


def fetch(driver,trg,total=389):
    current = -1
    pngs = []
    chapter_id, chapter_name, pages = 0, "",[]
    chapters = []
    while current<=total:
        current += 1
        app = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "app"))
        )
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "chapter-name"))
        )
        sleep(1)
        if not current:
            app.click()
        else:
            ActionChains(driver).send_keys(Keys.ARROW_RIGHT).perform()
        element=WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "chapter-name"))
        )
        if element.text != chapter_name:
            if chapters:
                print(chapters[-1])
            chapters.append((chapter_id,chapter_name,pages))
            chapter_name = element.text
            chapter_id +=1
            pages = [current]
        else:
            pages.append(current)
        sleep(1)
        png = os.path.join(trg, f'{current}.png')
        driver.get_screenshot_as_file(png)
        pngs.append(png)
        cfn = os.path.join(os.path.dirname(trg),"chapters.pkl")
        with open(cfn, 'wb') as cfp:
            pickle.dump({"chapters":chapters}, cfp)

    return pngs, cfn


def mk_pdf(pngs,chapters,trg, total=389):
    menus = pickle.load(open(chapters,"rb"))
    print(menus)
    # 创建一个新的 PDF 写入器
    pdf_writer = PyPDF2.PdfWriter()
    for i,pdf_file in enumerate(pngs):
        pdf_data = img2pdf.convert(pdf_file)
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
        # 将原 PDF 的页面添加到新的 PDF
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_writer.add_page(page)
    # 为目录添加书签
    for i,cname, pages in menus["chapters"]:
        if pages:
            if pages[0]>=total:
                break
            pdf_writer.add_outline_item(cname, pages[0])
    # 保存新的 PDF 文件
    with open(trg, 'wb') as output_file:
        pdf_writer.write(output_file)


if __name__ == '__main__':
    url = "https://ebooks.jd.com/reader/?ebookId=30833796&return_url=%2Flogin&index=0&from=3"
    ourl = "https://passport.jd.com/new/login.aspx?ReturnUrl=https%3A%2F%2Febooks.jd.com%2Freader%2F%3FebookId%3D30833796%26return_url%3D%252Flogin%26index%3D0%26from%3D3"
    wdriver, total = get_driver(ourl,waits=35), 450
    pngs, chapters = fetch(wdriver,trg="C:\\Users\\jinya\\Downloads\\ebook\\png1",total=total) # 389
    mk_pdf(pngs,chapters,'C:\\Users\\jinya\\Downloads\\ebook\\nlp1.pdf',total=total)
    # 关闭浏览器
    wdriver.quit()
