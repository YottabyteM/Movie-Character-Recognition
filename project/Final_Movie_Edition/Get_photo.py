import requests
import time
import random
from urllib.request import urlretrieve
from xpinyin import Pinyin
import os


class catcher(object):
    def __init__(self, name):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'
        }
        self.name = name

    def download_pic(self, href):
        # 判断当前目录下是否存在该文件夹，如果没有就创建
        path = "./photo/"
        p = Pinyin()
        pinyin = p.get_pinyin(self.name)
        if not os.path.exists(path + pinyin):
            os.mkdir(path + pinyin)

        name = os.path.split(href)[1]
        urlretrieve(href, './' + path + pinyin + '/{}'.format(name))
        print("=================={}下载完成===================".format(name))
        time.sleep(random.random())

    def request(self, url):
        response = requests.get(url, headers=self.headers)
        time.sleep(random.uniform(0, 1))
        lists = response.json()['data']['object_list']
        for list in lists:
            pic_url = list['photo']['path']
            self.download_pic(pic_url)  # pic_url即为图片的网址

    def run(self):
        for i in range(0, 200, 24):
            url = 'https://www.duitang.com/napi/blog/list/by_search/?kw=' + self.name + '&type=feed&include_fields=top_comments%2Cis_root%2Csource_link%2Citem%2Cbuyable%2Croot_id%2Cstatus%2Clike_count%2Clike_id%2Csender%2Calbum%2Creply_count%2Cfavorite_blog_id&_type=&start={}'.format(
                i)
            self.request(url)