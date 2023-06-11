# ! Python 3.8
# V_fetchFedPDF.py --- Aim to download all the pdf files from U.S. Federal reserved website.
import os
from pathlib import Path
import requests, bs4
 # Create directory to stored files downloaded from Federal reserved website.
fedDirName = "fedPDFs"
crtDir = Path(Path(os.getcwd()).parent, fedDirName)
os.makedirs(crtDir, exist_ok=True) # This will not overwrite the content of an existed directory anyway.

# request from the fed calender website.
fedRes = requests.get('https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm')
fedResSoup = bs4.BeautifulSoup(fedRes.text, 'html.parser')

# seperate PDF file names from the website content.
fedFileNames = []
fedResElems = fedResSoup.select('a')
for i_fedResElems in range(len(fedResElems)):
    pdfPath = fedResElems[i_fedResElems].get('href')
    if pdfPath != None and ".pdf" in pdfPath:
        fedFileNames.append("https://www.federalreserve.gov"+ pdfPath)

# Download all the PDF files located from the website.
for dl_fileName in fedFileNames:
    res = requests.get(dl_fileName)
    with open(os.path.join(crtDir, dl_fileName.split('/')[-1]), "wb") as saving:
        saving.write(res.content)


