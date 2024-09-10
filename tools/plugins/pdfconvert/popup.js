document.addEventListener('DOMContentLoaded', () => {
  const savePdfBtn = document.getElementById('savePdfBtn');

  if (savePdfBtn) {
    savePdfBtn.addEventListener('click', () => {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.scripting.executeScript({
          target: { tabId: tabs[0].id },
          function: savePageAsPDF
        });
      });
    });
  } else {
    console.error("Button with ID 'savePdfBtn' not found.");
  }
});

function savePageAsPDF() {
  const { jsPDF } = window;
  html2canvas(document.body,{
      allowTaint: true, // 允许跨域的图片被渲染到 canvas 上
      useCORS: true     // 开启 CORS 支持，避免跨域问题
    }).then(canvas => {
//    const pdf = new jsPDF();
    const pdf = new jsPDF('l', 'mm','a4',true);
//    const pdf = new jsPDF('p', 'mm',[595.28, 641.89],true);
//    const pdf = new jsPDF({
//      orientation: "portrait",
//      unit: "mm", // 单位: 毫米 (mm),
//      format:"a4" //[210, 297] // 自定义 A4 尺寸
//    });
    const imgData = canvas.toDataURL('image/png');
    const imgWidth = 297; // A4纸的宽度为210mm
    const pageHeight = 210; // A4纸的高度为297mm
    const imgHeight = canvas.height * imgWidth / canvas.width
//    const imgHeight = 200;
    let position = 0;
    console.log(imgHeight);
    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight,true);
    pdf.save('page.pdf');
  });
}