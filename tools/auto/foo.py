import shutil
import img2pdf
import PyPDF2
import os
import pickle
import io

src = "C:\\Users\\jinya\\Downloads\\ebook\\pngs1\\{}.png"
trg = "C:\\Users\\jinya\\Downloads\\ebook\\pngs\\{}.png"


def move(src=src, trg=trg):
    for i in range(671,690):
        fn = src.format(i)
        tfn = trg.format(i+1)
        shutil.copy(fn,tfn)


def png_to_pdf(pdf_file):
    png_files = []
    for i in range(0,161):
        png_files.append(trg.format(i))
    pdf_data = img2pdf.convert(png_files)
    with open(pdf_file, "wb") as fp:
        fp.write(pdf_data)

def mk_pdf(pngs,chapters,total=389):

    menus = pickle.load(open(chapters,"rb"))
    print(menus)
    # 创建一个新的 PDF 写入器
    pdf_writer = PyPDF2.PdfWriter()
    for i in range(total):
        pdf_file = os.path.join(pngs,f"{i}.png")
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
    with open('C:\\Users\\jinya\\Downloads\\ebook\\nlp1.pdf', 'wb') as output_file:
        pdf_writer.write(output_file)



if __name__ == '__main__':
    # png_to_pdf("C:\\Users\\jinya\\Downloads\\ebook\\nlp1.pdf")
    mk_pdf("C:\\Users\\jinya\\Downloads\\ebook\png1",
           "C:\\Users\\jinya\\Downloads\\ebook\\chapters.pkl",total=68)