from scrapy import Spider, Request
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import re
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import urllib.request
import os
# CONSUMER_KEY = "PZDC6OYeOJGVvNhdk5OW6xlie"
# CONSUMER_SECRET = "PiErxlNup3AqYR2Bzw2GRdZwiLrRI8E5VfFdpVMybXduufqDZc"
# ACCESS_TOKEN = "947247576474177536-h7LnVv4EAZXGTx5kGCIXge1HVsuONbb"
# ACCESS_SECRET = "evTE1WJn5xQBX307BhIzhToedJtzHWNrfBQtHtVpLraoQ"
# ACCESS_LEVEL = "R/W"
# auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
# auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
# api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

read = []
driver = None

def clean(tweet) -> str:
    tweet = tweet.lower()
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet) # remove hyperlinks
    tweet = re.sub(r'\&\w*;', '', tweet)
    tweet.replace(',', ';')
    #tweet = re.sub(r'[' + '$%&()*+-/:<=>[\]^_`{|}~' + ']+', ' ', tweet)
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    tweet = tweet.lstrip(' ')
    tweet = ''.join(c for c in tweet if c <= '\uFFFF')
    tweet = re.sub(r'\s\s+', ' ', tweet)
    return tweet


def driver_setup():
    global driver
    desired_capabilities = DesiredCapabilities.CHROME.copy()
    # desired_capabilities['acceptInsecureCerts'] = True
    option = webdriver.ChromeOptions()
    #option.add_argument('--ignore-certificate-errors')
    #option.add_argument('--ignore-ssl-errors')
    # option.add_argument('headless')
    driver = webdriver.Chrome('C:\chromedriver\chromedriver.exe',
                              desired_capabilities=desired_capabilities, options=option)

def driver_close():
    driver.close()
driver_setup()


class ReplySpider(Spider):
    name = "pinterest"

    def scroll_down(self):
        driver.implicitly_wait(4)
        page = driver.find_element_by_tag_name('body')
        count = 0
        i = 0
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            # print(count)
            page.send_keys(Keys.PAGE_DOWN)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height and count > 70:
                break
            elif new_height == last_height:
                count += 1
            if new_height != last_height:
                count = 0
            last_height = new_height
            i += 1

    def parse(self, response):
        self.item = []
        print("Parse Reply URL:", response.url)
        driver.implicitly_wait(8)
        driver.get(response.url)
        driver.implicitly_wait(8)
        self.scroll_down()
        urls = os.listdir('./images/')
        board = BeautifulSoup(driver.page_source, "html.parser")
        # board = board.find('div', {"class":"gridCentered"})
        imgs = board.find_all('img', {"class": "GrowthUnauthPinImage__Image"})
        urls_temp = [img['src'].replace('236x', '564x') for img in imgs if img['src'].split('/')[-1] not in urls]
        urls += [url.split('/')[-1] for url in urls_temp]
        [urllib.request.urlretrieve(url, './images/' + url.split('/')[-1]) for url in urls_temp]
        print(urls_temp)
        return self.item

    def scrape_id(self) -> int:
        # css-4rbku5 css-18t94o4 css-901oao r-1re7ezh r-1loqt21 r-1q142lx r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-3s2u2q r-qvutc0
        urls = driver.find_elements_by_xpath(".//*[@class='css-4rbku5' and @class='css-18t94o4' and @class='css-901oao' and @class='r-1re7ezh' and @class='r-1loqt21' and @class='r-1q142lx' and @class='r-1qd0xha']")
        print(urls)
        result = []
        for url in urls:
            url = url.get_attribute('href')
            result.append(url.split('/'[-1]))
        return result


url = 'https://www.pinterest.ca/154757929sherry/digital-painting-and-drawing/'
response = Request(url)
spider = ReplySpider()
spider.parse(response)