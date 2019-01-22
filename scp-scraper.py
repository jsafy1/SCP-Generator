import requests
import csv
from bs4 import BeautifulSoup as bs

url = "http://www.scp-wiki.wikidot.com/scp-"
pages = []

for scp in range(2, 5000):
    print(scp) if scp % 100 == 0 else 0
    if scp < 100:
        scp = "0" * (3 - len(str(scp))) + str(scp)
    else:
        scp = str(scp)  
    response = requests.get(url + str(scp))
    soup = bs(response.text, "html.parser")
    try:
        text = ""
        for p in soup.find_all('p')[8: -1]:
            text += p.text + " "
        text = text[text.find("Item #:"):]
        pages.append([
            scp,
            text[:-1]
        ])
    except BaseException as e:
        print(e)
        pass

with open("scp.csv", "w", encoding="utf8") as output:
    fieldnames = ["SCP Number", "Text"]
    writecsv = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writecsv.writeheader()
    for i in pages:
        writecsv.writerow({
            "SCP Number": i[0],
            "Text": i[1]
        })
