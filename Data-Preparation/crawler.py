from urllib.request import urlopen
from urllib.error import HTTPError 
from urllib.error import URLError
from bs4 import BeautifulSoup
 
url = "https://www.wongnai.com/restaurants/233127qX-%E0%B9%80%E0%B8%88%E0%B9%8A%E0%B9%80%E0%B8%88%E0%B8%B4%E0%B8%99%E0%B8%99%E0%B8%A1%E0%B8%AA%E0%B8%94-%E0%B8%A8%E0%B8%B2%E0%B8%A5%E0%B8%B2%E0%B8%A2%E0%B8%B2"

try:
  html = urlopen(url)
 
except HTTPError as e:
  print(e)
 
except URLError:
  print("Server down or incorrect domain")
 
else:
  res = BeautifulSoup(html.read(),"html5lib")
  paragraph = res.findAll("p")
  name = res.findAll("span", {"class": "favouritesContainer"})
  if res.title is None:
    print("Tag not found")
  else:
    #print(res.title)
    
    for tag in paragraph:
      print(tag.getText())

    for x in name:
      print(x.getText())