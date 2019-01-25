import requests
import csv
import re
import string
import unicodedata as ud
from bs4 import BeautifulSoup as bs

url = "http://www.scp-wiki.wikidot.com/scp-"
pages = []
ascii = set(string.printable) 
ascii.add("█")  


def remove_non_ascii(s):  # This function will help us reduce the number of characters the model needs to learn.
    return filter(lambda x: x in ascii, s)


for scp in range(2, 5000):
    print(scp) if scp % 100 == 0 else 0
    if scp < 100:
        scp = "0" * (3 - len(str(scp))) + str(scp)  # SCPs are left-padded e.g. 001, 099, etc. until 100
    else:
        scp = str(scp)

    response = requests.get(url + str(scp))
    soup = bs(response.text, "html.parser")
    try:
        text = ""
        for p in soup.find_all('p')[8: -1]:  # Splice out generic elems to get body text only. 
            text += p.text.replace("\n", " ").lower() + " "  # We use lower to reduce chars needed to learn. 
        text = text[text.find("item #:"):]
        if text:
            text_filter = remove_non_ascii(text)
            filtered_text = ""
            for char in text_filter:
                filtered_text += char
            pages.append(filtered_text[:-1])
    except BaseException as e:
        print(e)
        pass

print(len(pages))

with open("scp.txt", "w", encoding="utf8") as output:
    for page in pages:
        output.write(page + "\n\n")

# Output to CSV if needed
# with open("scp.csv", "w", encoding="utf8") as output:
#     fieldnames = ["SCP Number", "Text"]
#     writecsv = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
#     writecsv.writeheader()
#     for i in pages:
#         writecsv.writerow({
#             "SCP Number": i[0],
#             "Text": i[1]
#         })
