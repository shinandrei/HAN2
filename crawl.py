import requests
import re
from urllib.request import HTTPError
from bs4 import BeautifulSoup
import pandas as pd

def fetchUrl(url):
    '''
    功能：访问 url 的网页，获取网页内容并返回
    参数：目标网页的 url
    返回：目标网页的 html 内容
    '''

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return r.text

def get_content(url):
    try:
        html = fetchUrl(url)
    except HTTPError as e:
        return None
    try:
        obj = BeautifulSoup(html, "html.parser")
    except HTTPError as f:
        return None
    titlelist = obj.find("div", {"class": "datelist"}).ul.find_all('a')
    urllist = []
    data = []
    for title in titlelist:
        urllist.append(title['href'])
        data.append(title.text)
    date = re.findall('(\d{4}-\d{2}-\d{2})', obj.find("div", {"class": "datelist"}).ul.text)
    return urllist, data, date

news_total = pd.DataFrame(columns = ['date','text'])
url = "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol=sh600028&Page=1"
urllist, data, date = get_content(url)
data = pd.Series(data)
date = pd.Series(date)
news = pd.concat([date, data], axis=1)
for i in range(2, 27):
    url = "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol=sh600028&Page=%d"%i
    urllist, data, date = get_content(url)
    data = pd.Series(data)
    date = pd.Series(date)
    news_1 = pd.concat([date, data], axis=1)
    news = pd.concat([news, news_1])
news.columns = ['date', 'text']
#news.to_csv(r'E:\data\news_train.csv', encoding='utf_8_sig', index=False)