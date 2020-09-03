import requests
import selenium.webdriver as webdriver
from selenium.webdriver.firefox.options import Options

class Scraper:
    def __init__(self, driver='/opt/anaconda3/bin/geckodriver', timeout=0):
        options = Options()
        options.set_preference("browser.download.folderList", 1)
        options.set_preference("browser.download.manager.showWhenStarting", False)
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
        options.set_preference("pdfjs.disabled", True)
        driver = webdriver.Firefox(executable_path=driver, options=options)
        driver.minimize_window()
        if timeout != 0:
            driver.set_page_load_timeout(timeout)
        self.driver = driver

    def get_all_links(self, url):
        # Get all link elements at a given url
        try:
            self.driver.get(url)
            self.driver.implicitly_wait(1)
            elements = self.driver.find_elements_by_tag_name('a')
            links = []
            for element in elements:
                try:
                    link = element.get_attribute('href')
                    links.append(link)
                except: pass
            for link in links:
                if (link == None) or (link == 'None'):
                    links.remove(link)
            return links
        except:
            return None

    def download_files(self, url, path, format='.pdf'):
        # Get all files of a given format at a given url
        links = self.get_all_links(url)
        filenames = []
        for link in links:
            if format in link:
                directory = path.split('/')[0]
                local_filename = path + '_' + url.split('/')[-1]
                filenames.append(local_filename)
                with requests.get(link, stream=True) as r:
                    r.raise_for_status()
                    with open(local_filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
        return filenames

    def close_session(self):
        self.driver.close()

