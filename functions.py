## principal functions and objects file

# clear sections are shown in comments
# go to docstrings for function purpose and arguments

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import requests as req
load_dotenv()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
import string

from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
tokenizer = RegexpTokenizer(r'\b[A-Za-z0-9\-]{1,}\b')
default_tk = tokenizer
gen_stop_words = list(set(stopwords.words("english")))
gen_stop_words += list(string.punctuation)
import bs4
from bs4 import BeautifulSoup
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
lemmy = WordNetLemmatizer()

import wordcloud

from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.stem import WordNetLemmatizer as wlemmy
from nltk import word_tokenize       


class LemmaTokenizer(object):
    def __init__(self, tokenizer = default_tk, stopwords = gen_stop_words):
        self.wnl = WordNetLemmatizer()
        self.tokenizer = tokenizer
        self.stopwords = stopwords
    def __call__(self, articles):
        return [self.wnl.lemmatize(token, ) for token in self.tokenizer.tokenize(articles) if token not in self.stopwords]
    
    def tokenize(self, articles):
        return [self.wnl.lemmatize(token) for token in self.tokenizer.tokenize(articles) if token not in self.stopwords]
    
    
    
    
import selenium as sl
from selenium.common.exceptions import ElementClickInterceptedException, NoSuchElementException
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.action_chains import ActionChains
import time




#################################CLEANING#####################################
def preprocess_data(string):
    """Function that takes in any single continous string;
    Returns 1 continuous string
    A precautionary measure to try to remove any emails or websites that BS4 missed"""
    new_str = re.sub(r"\S+@\S+", '', string)
    new_str = re.sub(r"\S+.co\S+", '', new_str)
    new_str = re.sub(r"\S+.ed\S+", '', new_str)
    new_str_tok = tokenizer.tokenize(new_str)
    new_str_lemm = [lemmy.lemmatize(token) for token in new_str_tok]
    new_str_cont = ''
    for tok in new_str_lemm:
        new_str_cont += tok + ' '
    return new_str_cont

def gen_stopwords(additional_sw = ['data', 'experience', 'learning', 
                                'science', 'machine', 'work', 'company', 
                                'role', 'the', 'skills', ' data', '000', 
                                "data", "the", 'join', 'you'], general_sw = gen_stop_words):
    """Function that calls upon nltk's built-in list of stop words and appends the word included in the param additional_sw.
    Param:
    additional_sw - (list) expects a list of string which will be added to the stopwords
    Returns:
    new_stop_words - (list) stopwords to be used"""
    new_stop_words = general_sw
    for sw in additional_sw:
        new_stop_words.append(sw)
    new_stop_words = list(set(new_stop_words))
    return new_stop_words



#################################SCRAPING#####################################


class JobPostScraper:
    def __init__(self, root_url, search_term_job, location, num_jobs):
        """Initialise the job scraper object with 
        - root_url - (str) of the website you're visiting in our case 'indeed.co.uk', 
        - search)_term_job - (str) the job you're looking for (e.g. 'data scientist'), 
        - location - (str) your location (e.g. 'London')
        - num_jobs - (int) how many job postings you'd like to look at
        """
        self.root_url = root_url
        self.search_term_job = search_term_job
        self.location = location
        self.num_jobs = num_jobs
        self.job_descr_lst_ = []
        self.job_titles_lst_ =[]
        self.companies_lst_ = []
        self.job_post_dom_ = []
        self.job_post_urls_ = []
        self.scrape_asctimes = []
        
        return
    


    # def get_job_link_urls(self, headless=False):
    #     """Instance method that start a Selenium Chrome driver that scrapes a website and searches
    #     for job URLs, paginates and then stores the num_jobs amount of URLs in a pandas dataframe
    #     for use later down the pipeline.
    #     headless - (bool) whether to have the chrome window showing or not as it's scraping
    #     """
    #     start = time.time()
    #     # empty list to store urls from within the main job posting website
    #     sub_urls = []
    #     #init the selenium driver
    #     chrome_options = Options()
    #     if headless:
    #         chrome_options.add_argument("--headless")
    #     driver = webdriver.Chrome('/Users/ipreoteasa/Desktop/Io/chromedriver_2', 
    #                              options=chrome_options)
    
    #     # accessing main page
    #     driver.get(self.root_url)
    #     time.sleep(2)


    def get_job_link_urls(self, headless=False):
        """Instance method that start a Selenium Chrome driver that scrapes a website and searches
        for job URLs, paginates and then stores the num_jobs amount of URLs in a pandas dataframe
        for use later down the pipeline.
        headless - (bool) whether to have the chrome window showing or not as it's scraping
        """
        start = time.time()
        # empty list to store urls from within the main job posting website
        sub_urls = []
        #init the selenium driver
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        driver = webdriver.Chrome('/Users/ipreoteasa/Desktop/Io/chromedriver_2', 
                                 options=chrome_options)
    
        # accessing main page
        driver.get(self.root_url)
        time.sleep(2)
        
        
#         DEBUGG: website introduced a new popup re consent to data collection practices, that is 
#         messing up the scraping. For now will click on manually. Will return to this at some point
#         try:
#             time.sleep(3)
#             pop_up_close = driver.find_element_by_class_name('icl_LegalConsentBanner-action')
#             pop_up_close.click()
#         except:
#             print("CANT FIND BUTTON")
        
        #enter our job search terms
        elem = driver.find_element_by_name('q')
        elem.clear()
        elem.send_keys(self.search_term_job)

        time.sleep(2)
        #enter our location search term
        elem = driver.find_element_by_name('l')
        elem.clear()
        elem.send_keys(self.location)
        elem.click()
        time.sleep(1)
        elem.send_keys(Keys.RETURN)

        time.sleep(4)
        
        time_index = 0
        
        while len(sub_urls)<self.num_jobs:
            try:
                time.sleep(3)
                pop_up_close = driver.find_element_by_class_name('popover-x')
                pop_up_close.click()
            except:
                pass
            
            

            # using BS4 on the page source to get all the urls
            DOM = driver.page_source
            soup = BeautifulSoup(DOM, 'lxml')
            
            jobtitle_soup = soup.find_all(name='a', 
                                               attrs= {'class': 'jobtitle turnstileLink', 
                                                       'data-tn-element':'jobTitle'})
            
            # getting href attributes and storing them
            list_hrefs = [jobtitle_elem['href'] for jobtitle_elem in jobtitle_soup]
            for href in list_hrefs:
                sub_urls.append(href)
                sub_urls = list(set(list(dict.fromkeys(sub_urls))))
                if len(sub_urls)>= self.num_jobs:
                    break

            #the following try deals with the cookies popup    
            try:
                cookie_popup_elem=WebDriverWait(driver, 2).until(ec.presence_of_element_located((By.ID, 'onetrust-accept-btn-handler')))
                ActionChains(driver).move_to_element(cookie_popup_elem).click().perform()
            except:
                pass

            # driver waits for the next page button to be viewable before moving and clicking
            WebDriverWait(driver, 2).until(ec.element_to_be_clickable((By.CLASS_NAME, 'np')))
            next_page_buttons = driver.find_elements_by_class_name('np')
            time.sleep(4)
            ActionChains(driver).move_to_element(next_page_buttons[-1]).click().perform()
            
            # following lines deal with the sign-up popup 
            try:
                popup_elem=WebDriverWait(driver, 2).until(ec.presence_of_element_located((By.ID, 'popover-x')))
                ActionChains(driver).move_to_element(popup_elem).click().perform()
            except:
                pass
            
            
            #printout
            time_elapsed = time.time() - start
            
            printout = f'Step {time_index} --- Time elapsed so far {time_elapsed}; URLs stored : {len(sub_urls)}'
            print(printout)
            time_index+=1
            
        # Now we take our list of urls, preppend the root url to them and store them in a dataframe
        job_urls_full = list(map(lambda x: str(self.root_url)+x , sub_urls))
        # job_urls_full = list(dict.fromkeys(job_urls_full))
        job_url_df = pd.DataFrame(job_urls_full, columns=['job_url'])
        
        self.job_post_urls_ = job_urls_full
        print('URL column successfully stored as pandas obj')

        return job_url_df
    
    
    def get_job_text_html(self,url_df, url_column = 'job_url', headless=True):
        """Retrieve the body of the job posting text using Selenium for browser interaction and
        Beautiful Soup for parsing and HTML tag removal
        url_df - (pandas dataframe/series) that contains our URLs
        url_column - (str) name of the dataframe column that contains URLs, by default = 'job_url'
        headless - (bool) whether to have the chrome window showing or not as it's scraping
        """
        start_job_descr = time.time()
        job_descr_lst = []
        # empty list to store urls from within the main job posting website
        if str(type(url_df)) == 'pandas.core.frame.DataFrame':
            url_list = list(url_df[url_column].values)
        elif str(type(url_df)) == 'pandas.core.series.Series':
            url_list = list(url_df)
        else:
            url_list = list(url_df[url_column].values)
            

        #init the selenium driver
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        driver = webdriver.Chrome('/Users/ipreoteasa/Desktop/Io/chromedriver_2', 
                                 options=chrome_options)
        
        job_descr_list = []
        
        for url in url_list:
            driver.get(url)
            time.sleep(5)
            dom =  driver.page_source
            job_soup = BeautifulSoup(dom, 'lxml')
            job_soup_title = job_soup.find(name='div', 
                                           attrs= {'class': 'jobsearch-JobInfoHeader-title-container'})
            
            job_soup_descr = job_soup.find(name='div', 
                                           attrs= {'class': 'jobsearch-jobDescriptionText', 
                                                   'id':'jobDescriptionText'})
            
            job_soup_company = job_soup.find(name='div', 
                                           attrs= {'class': 'jobsearch-InlineCompanyRating icl-u-xs-mt--xs jobsearch-DesktopStickyContainer-companyrating'})
            try:
                job_soup_title_txt = job_soup_title.get_text()
                self.job_titles_lst_.append(job_soup_title_txt)
            except:
                pass
            
            try:
                job_soup_descr_txt = job_soup_descr.get_text()
                self.job_descr_lst_.append(job_soup_descr_txt)
            except:
                pass
                
            try:
                job_soup_comp_txt = job_soup_company.get_text()
                self.companies_lst_.append(job_soup_comp_txt)
            except:
                pass
            
            self.job_post_dom_.append(job_soup)
            self.scrape_asctimes.append(time.asctime())
            
            if (len(self.job_descr_lst_)%50 ==0):
                time_elapsed_get_jobs = time.time() - start_job_descr
                printout = f'--- Time elapsed so far {time_elapsed_get_jobs}; jobs stored so far:  {len(self.job_descr_lst_)}'
                print(printout)
                
        return            
    
    def get_jobs_df(self):
        """Functions assembles a Pandas dataframe with 5 columns:
        URLs of job posts; the company; the job titles; the job
        description text and also the entire html of the job post
        page, in case the user would like to to any more data
        extraction from that data
        """
        data = pd.DataFrame({
                            'company': self.companies_lst_,
                            'job_title' : self.job_titles_lst_,
                            'job_descr' : self.job_descr_lst_,
                            'job_post_html' : self.job_post_dom_,
                            'time_of_scrape' : self.scrape_asctimes})
        
        data['job_search_term'] = str(self.search_term_job)
        data['job_location'] = str(self.location)
        
        return data       




################################TEXT MINING#####################################


def get_salary(dom, parser = 'html', regex_pattern_salary = ['£[0-9]*[,]*[0-9]+[ ]+[a]+'], 
               regex_pattern_interval=['£[0-9]*[,]*[0-9]+[ ]+[a]*[ ]+year', 
                                       '£[0-9]*[,]*[0-9]+[ ]+[a]*[ ]+month', 
                                       '£[0-9]*[,]*[0-9]+[ ]+[a]*[ ]+week',
                                       '£[0-9]*[,]*[0-9]+[ ]+[a]*[ ]+day',
                                       '£[0-9]*[,]*[0-9]+[ ]+[an]*[ ]+hour'],
              ):
    """Function that should be applied across the elements of a column in a pandas dataframe. 
    Parses a webpages html code using BeautifulSoup and a specified parser, after which it tries to identify
    salary mentions using regex. If multiple pattern occurrences are found, the function will return the 
    mean of the first two, based on the assumption that the web page would feature a string similar to :
     - Salary range: £25,000 - £32,000
     Args:
     dom - block of html code
     parser - (str) parser that BeautifulSoup should use, e.g. 'html' (default) or 'lxml'
     regex_pattern_salary - (list of str) lis of patterns for regex to use for retrieving the numerical data
     regex_pattern_interval - (list of str) list of patterns for regex to use for determining if the salary is
                            per year, per week, per month.
     
     Returns a TUPLE:
     salaries_final - (int / float) a single value of salary
     salaries_final_adjusted - (float) - the hourly salary, calculated from whichever value was in the html
     salary_period - (str) for what time period the salary is declared for: Y - per year; 
                         M - per month, H - per hour, W - per week, D - per day
     """
    dom = str(dom)
    soup = BeautifulSoup(dom, parser)
    # BS4 used to parse the dom and just retrieve the text, eliminating any html tags
    soup_str = soup.get_text()
    for regex_pattern in regex_pattern_salary:
        # iterating over the text using the numerical patterns to find salary mentions
        salary_tokenizer = RegexpTokenizer(regex_pattern)
        salaries = salary_tokenizer.tokenize(soup_str)
        # if we get several occurrences, we take the first two mentions only
        if len(salaries)>1:
            salaries_lst = [salaries[0], salaries[1]]
            # removing pound sympol, letters and any commas
            salaries_clean = [re.sub('£', '', salary) for salary in salaries_lst]
            salaries_clean = [re.sub('[a-zA-Z]', '', salary) for salary in salaries_clean]
            salaries_clean = [int(re.sub(',', '', salary)) for salary in salaries_clean]
            # final salary is the mean of the first two occurrences
            salaries_final = np.mean([salaries_clean[0], salaries_clean[1]])
        # if there's only 1 occurence we just take that one
        elif len(salaries)==1:
            salary_clean = re.sub('£', '', salaries[0])
            salary_clean = re.sub('[a-zA-Z]', '', salary_clean)           
            salary_clean = int(re.sub(',', '', salary_clean))
            salary_clean = int(salary_clean)
            salaries_final = salary_clean
        else:
            # for ease of dataframe manipulation down the line
            # posts with no detectable job are left with a NaN
            salaries_final = np.NaN
    
    for regex_pattern in regex_pattern_interval:
        # now iterating over the regex patterns in the interval list
        # trying to suss out if the mentioned salary is per year/month/week
        # break is added after each if partially for efficiency but also 
        # because of not wanting a latter interval str occurence to 
        # overwrite the original, e.g. if the post gives a per month salary first then
        # later repeats in weeks, but the function's stored salary numeric value is the 
        # one given in months, we want to avoid treating it as if in months
        interval_tokenizer = RegexpTokenizer(regex_pattern)
        intervals = interval_tokenizer.tokenize(soup_str)
        if (type(intervals)==list) and (len(intervals)>1):
            salary_interval = intervals[0]
#             print('INTERVAL', salary_interval)
            break
        elif type(intervals)==str:
            salary_interval = intervals
#             print('INTERVAL', salary_interval)
            break
        else:
            salary_interval = ''
            continue
        
    if (salary_interval!=''):    
        if 'month' in salary_interval:
            salaries_final_adjusted = salaries_final/159
#             print('SALARY PER MONTH is', salaries_final)
            salary_period = 'M'
        elif 'week' in salary_interval:
            salaries_final_adjusted =  salaries_final/36.5
#             print('SALARY PER MONTH is', salaries_final)
            salary_period = 'W'
        elif 'year' in salary_interval:
            salaries_final_adjusted = salaries_final/1898
#             print('SALARY PER YEAR is', salaries_final)
            salary_period = 'Y'
        elif 'day' in salary_interval:
            salaries_final_adjusted = salaries_final/7.3
#             print('SALARY PER DAY')
            salary_period = 'D'
        elif 'hour' in salary_interval:
            salaries_final_adjusted = salaries_final
#             print('SALARY PER HOUR')
            salary_period = 'H'
        else:
#             print('salary stated for interval that is not comprehended - not reliable long-term data')
            salaries_final = np.NaN
            salart_period = np.NaN
            salaries_final_adjusted = np.NaN
    else:
        salary_period = np.NaN
        salaries_final_adjusted = np.NaN
    
    return salaries_final, salaries_final_adjusted, salary_period





#################################DATA TRANSFORMATION#####################################
def get_num_reviews(text, regex_pattern = '[0-9]*[,]*[0-9]* review[s]*'):
    """Function that should be applied across the elements of a column in a pandas dataframe. 
    Goes through the text trying to find a pattern for the number of reviews left using regex. 
    An occurrence of 0 will just return a nan
     Args:
     text - (str) input text
     regex_pattern - (str) pattern for regex to use for retrieving data
     
     Returns :
     num_review - a single
     """
    text = str(text)
    
    reviews = re.findall(regex_pattern, text)
    if reviews==[]:
        num_review_int= np.NaN
    else:
        num_review = reviews[-1]
        num_review_clean = re.sub(',', '', num_review)
        num_review_int = int(re.sub('review[s]*', '', num_review_clean))
    return num_review_int

def get_pattern_count(text, lang_string):
    
    if (lang_string.lower()=='c#' or lang_string.lower()=='c++'):
        text = re.sub('[.,!?:;\']', ' ', text)
        text_lst = text.split(' ')
        found_lst = [text for text in text_lst if text==lang_string]   
        matches = found_lst
    else:

        regexp = re.compile("(?=\w)(?<![-a-zA-Z+#])"+lang_string+"(?<=\w)(?![#+a-zA-Z])") #"(?=[ ,.!?])")  # (?=[ ,.!?])
    
        matches = regexp.findall(text)
    return len(matches)

def get_pattern_count_and_store(data, lang_string_lst, columns = None):
    df = data.copy()
    if columns == None:
        columns ='job_descr'
    
    if type(columns)!=list:
        col = columns
        for lang in lang_string_lst:

            if type(lang)==list:
                prime_lang = lang[0]
                df['num_mention_{}_{}'.format(col, prime_lang)] = df[col].apply(lambda x : get_pattern_count(x, prime_lang))  
                for lang_variant in lang[1:]:
                    df['num_mention_{}_{}'.format(col, lang_variant)] = df[col].apply(lambda x : get_pattern_count(x, prime_lang))
                    df['num_mention_{}_{}'.format(col, prime_lang)] = df['num_mention_{}_{}'.format(col, prime_lang)] + df['num_mention_{}_{}'.format(col, lang_variant)]
                    df.drop(columns=['num_mention_{}_{}'.format(col, lang_variant)], inplace=True)

            else:        
                df['num_mention_{}_{}'.format(col, lang)] = df[col].apply(lambda x : get_pattern_count(x, lang))  
        
    else:
        for col in columns:
            
            for lang in lang_string_lst:

                if type(lang)==list:
                    prime_lang = lang[0]
                    df['num_mention_{}_{}'.format(col, prime_lang)] = df[col].apply(lambda x : get_pattern_count(x, prime_lang))  
                    for lang_variant in lang[1:]:
                        df['num_mention_{}_{}'.format(col, lang_variant)] = df[col].apply(lambda x : get_pattern_count(x, prime_lang))
                        df['num_mention_{}_{}'.format(col, prime_lang)] = df['num_mention_{}_{}'.format(col, prime_lang)] + df['num_mention_{}_{}'.format(col, lang_variant)]
                        df.drop(columns=['num_mention_{}_{}'.format(col, lang_variant)], inplace=True)

                else:        
                    df['num_mention_{}_{}'.format(col, lang)] = df[col].apply(lambda x : get_pattern_count(x, lang))  
        for lang in lang_string_lst:
            if type(lang)==list:
                cur_lang = lang[0]
            else:
                cur_lang = lang
            
            n_langs = len(lang_string_lst)
            n_cols = len(columns)
            df['num_mention_TOTAL_{}'.format(cur_lang)] = 0
            for col in columns:
                df['num_mention_TOTAL_{}'.format(cur_lang)] = df['num_mention_TOTAL_{}'.format(cur_lang)] + df['num_mention_{}_{}'.format(col, cur_lang)] 
#                 df.drop(columns = ['num_mention_{}_{}'.format(col, cur_lang)], axis=1, inplace=True)
                
#           
            df.drop(columns=['num_mention_{}_{}'.format(x, cur_lang) for x in columns], axis=1, inplace=True)

    return df

    

def get_location(text, regex_pattern = '\w[^-]*$'):
    """Function that should be applied across the elements of a column in a pandas dataframe. 
    Goes through the text (title of post typically) trying to find a pattern for the location left using regex. 
    An occurrence of 0 will just return a nan
     Args:
     text - (str) input text
     regex_pattern - (str) pattern for regex to use for retrieving data
     
     Returns :
     location - str
     """
    text = str(text)
    postcode_regex=  '\w[A-Z0-9]{2,}'
    
    location = re.findall(regex_pattern, text)
    if location==[]:
        retrieved_loc= np.NaN
    else:
        retrieved_loc = location[0]
        #removing any potential postcodes
        try:
            postcode = re.findall(postcode_regex, 
                          location[0])[0]
            retrieved_loc = re.sub(postcode,'', retrieved_loc)
        except:
            retrieved_loc = retrieved_loc
            
    
    if retrieved_loc[-1] == ' ':
        retrieved_loc = retrieved_loc[:-1]
    return retrieved_loc

def remove_reviews(text, regex_pattern = '[0-9]*[,]*[0-9]* review[s]*'):
    return re.sub(regex_pattern, '', text)

def remove_location(text, sep='-'):
    text_lst = text.split('-')
    return text_lst[0]

def full_clean_and_store(file_path, new_file_name):
    df = pd.read_csv(file_path, index_col=0)
#     df['job_descr'].drop_duplicates(inplace=True)
    df['job_descr'] = df['job_descr'].apply(preprocess_data) #removing emails, websites, identifiers
    df['salary_data'] = df['job_post_html'].apply(get_salary) #retrieve and store salary dataa from html dom
    df['salary_from_page_source_as_stated'] = [round(x[0],2) for x in df['salary_data'].values]
    df['salary_from_page_source_conv_hourly'] = [round(x[1],2) for x in df['salary_data'].values]
    df['salary_from_page_source_time_period'] = [x[2] for x in df['salary_data'].values]
    df.drop(columns=['salary_data'], inplace=True)
    df['Num_reviews'] = df.company.apply(get_num_reviews)
    df.company = df.company.apply(remove_reviews)
    df['Loc_from_title'] = df.company.apply(get_location)
    df.company = df.company.apply(remove_location)
    df.drop(columns=['job_post_html'], axis=1, inplace=True)
    df['date'] = pd.to_datetime(df.time_of_scrape).dt.date
    df.to_csv(new_file_name+'_CLEAN.csv')
    return



def get_experience(exp_pattern, text):
    results = re.findall(exp_pattern, text.lower())
    if results==[]:
        return np.NaN
    else:
        return results[-1]


#################################EDA#####################################

def plot_freqdist_from_series(pd_series, tokenizer_obj = default_tk, stop_words_list = gen_stop_words, 
                              title = 'Term Frequency distribution', num_terms=20, 
                              figsize = (10,10), ngram_number=1, lower_case=True):
    """Function that takes in a Pandas Series or column of a DataFrame and plots the Frequency Distribution
    of termns within that list of documents.
    Args:
    pd_series - either a standalone Pandas Series object or a dataframe column, e.g. df.job_description
    tokenizer_obj - (obj) a tokenizer object, normally of the NLTK variety
    num_terms - (int) how many of the top terms to plot on the Freq Dist, default 20
    stop_words - (list of str) list of stop words to exclude from final corpus
    figsize - (tuple of 2 integers) size of matplotlib plot, default is (10,10)
    ngram_numer - (int) what size ngrams to use, expects 1, 2 or 3. Default is 1.
                Values outside that list will just return the default. 
    lower_case - (bool) whether to return all words lowercased or not
    
    
    Plot of the Frequency Distribution of the words in the corpus, using NLTK's built in FreqDist function.

    Returns:
    f_dist_dict - (dict) ngrams as keys; frequency as value
    """
    all_text_lst = []
    for string in pd_series.tolist():
        output_txt = ''
        tokenized_str = tokenizer_obj.tokenize(string)
        for word in tokenized_str:
            if ((word.lower() not in stop_words_list) and (word not in stop_words_list)):
                if lower_case:
                    output_txt += word.lower() + ' '
                else:
                    output_txt += word + ' '
            else:
                continue
        ngram_list = list(nltk.ngrams(output_txt.split(' ')[:-1], n=ngram_number))
        for ngram in ngram_list:
            all_text_lst.append(ngram)
    
    f_dist = FreqDist(all_text_lst)

    f_dist_dict = dict(f_dist)
    
    plt.figure(figsize=figsize)
    plt.title(title)
    f_dist.plot(num_terms)
    plt.show();

    return f_dist_dict

def gen_cloud(data, max_words_num : int, stop_words = None,
              background_color = 'black', height = 300, 
              randomstate=42, fig_size=(12,12),
             cloud_title='Word Cloud', 
             ):
    cloud = wordcloud.WordCloud(max_words=max_words_num, background_color=background_color, height=height, random_state=randomstate)
    plt.figure(figsize=fig_size)
    text_data = ' '.join(data)
    if stop_words:
        text_data_clean = []
        for word in text_data.split(' '):
            if word.lower() not in stop_words:
                text_data_clean.append(word)
        text_data_clean = ' '.join(text_data_clean)
    else:
        text_data_clean = text_data
    cloud.generate(text_data_clean)
    plt.imshow(cloud)
    plt.title(cloud_title)
    plt.axis("off")
    plt.show()
    return cloud


def get_top_n_df(list_of_pd_series, list_new_df_titles, tokenizer_obj, stop_words_list, 
                  plot_title = 'Term Frequency distribution', num_terms=20, 
                  figsize = (10,10), ngram_number=1, lower_case=True, normalise=True
                 ):
    
    top_n_df = pd.DataFrame()
    
    list_of_topn_df = []
    plt.ioff()
    for i,pd_series in enumerate(list_of_pd_series):
        
        cur_var = list_new_df_titles[i]
        
        f_dist_dict = plot_freqdist_from_series(pd_series, tokenizer_obj, stop_words_list, 
                              title = '{} for {}'.format(plot_title, cur_var), num_terms=num_terms, 
                              figsize = figsize, ngram_number=ngram_number, lower_case=lower_case);
        
        sort_dict = { k: v for k, v in sorted(f_dist_dict.items(), key=lambda item: item[1] , reverse=True)}
        
        sort_val = list(sort_dict.values())
        sort_keys = list(sort_dict.keys())

        total_n = len(sort_val)
        top_n_v = sort_val[:num_terms]
        if normalise:
            top_n_v = [n_val/total_n for n_val in top_n_v]
        
        top_n_k = sort_keys[:num_terms]

        top_n_df_single = pd.DataFrame({'terms':top_n_k, 
                                 'freq_in_{}'.format(cur_var):top_n_v})

        
        list_of_topn_df.append(top_n_df_single)
        
    for pd_df in list_of_topn_df:
        top_n_df = top_n_df.append(pd_df)

#     plt.ion()
    return top_n_df


def get_top_n_terms(series, n, tokenizer_obj = default_tk, stop_words_list  =gen_stop_words, ngram_number=1,):
    plt.ioff()
    f_dist_dict = plot_freqdist_from_series(series, tokenizer_obj, stop_words_list, ngram_number = ngram_number)
    
    sort_dict = { k: v for k, v in sorted(f_dist_dict.items(), key=lambda item: item[1] , reverse=True)}

    sort_val = list(sort_dict.values())
    sort_keys = list(sort_dict.keys())
    
    total_n = len(sort_val)
    top_n_v = sort_val[:n]
    top_n_k = sort_keys[:n]
    if ngram_number!=1:
        top_n_k = [' '.join(term) for term in top_n_k]
    else:
        top_n_k = [term[0] for term in top_n_k]
    len_corpus = len(series)

    data = pd.DataFrame({'terms': top_n_k, 
                        'frequency':top_n_v,
                        })
    data['norm_frequency'] = (data.frequency / len_corpus) * 100
    
    
    return data

def plot_term_bar(final_title, 
                  list_of_pd_series, list_new_df_titles, tokenizer_obj, stop_words_list_, 
                  plot_title = 'Term Frequency distribution', num_terms=20, 
                  figsize = (10,10), ngram_number=1, lower_case=True, normalise=True,
                  save_fig=False, save_fig_name=None
                 ):
    
    plot_df = get_top_n_df(list_of_pd_series, list_new_df_titles, tokenizer_obj, stop_words_list_, 
                  plot_title = plot_title, num_terms=num_terms, 
                  figsize = (figsize[0]/4, figsize[1]/4), ngram_number=ngram_number, lower_case=lower_case, normalise=normalise
                 )
    

    # I mean to return to this fn to add docstring and properly generalize the barplotting element
    print(plot_df.columns)
#     merged_df_1 = plot_df.iloc[:,:2].merge(plot_df.iloc[:,2:4], how='outer')
#     merged_df = merged_df_1.iloc[:,:2].merge(plot_df.iloc[:,4:6], how='outer')
    
    merged_df = pd.melt(plot_df, id_vars='terms', var_name='search_term', value_name='frequency')
    plt.figure(figsize=(figsize))
    if normalise:
        x_label = 'normalised_frequency'
    else:
        x_label = 'absolute frequency'
    
    plot = sns.barplot(y = 'terms', x = 'frequency', data=merged_df, hue='search_term' )
    plt.xlabel(x_label)
    plt.title(final_title)
    
    fig=plot.get_figure()
    
    if save_fig:
        fig.savefig('{}.jpeg'.format(save_fig_name))
    
    
    return merged_df
    

def preview_topic_jobs(data, topic_col = 'topic_1', num_jobs = 3):
    topic_x_sort = data.sort_values(by=topic_col, ascending=False)
    for i in range(num_jobs):
        print('Job title: - ', topic_x_sort.job_title.iloc[i], '\n')
        print('Company: - ', topic_x_sort.company.iloc[i])
        print('Location (from title): - ', topic_x_sort.Loc_from_title.iloc[i])
        print('Job salary: - ', topic_x_sort.salary_from_page_source_as_stated.iloc[i])
        print('\n')
        print(topic_x_sort.job_descr.iloc[i])
        print(150*'-',' \n')
    return


def print_top_trig_collocs(word, pd_series, tokenizer, frac_corpus = 0.1, stopwords = gen_stop_words):
    corpus = [tokenizer.tokenize(x) for x in pd_series.to_list()]
    finder = TrigramCollocationFinder.from_documents(corpus)
    finder.apply_freq_filter(round(frac_corpus*len(pd_series)))
    main_trigrams = finder.nbest(trigram_measures.likelihood_ratio, 100000)
    for trigram in main_trigrams:
        if word in trigram:
            print(trigram)
        
    return


#################################SUMMARY TABLES CREATION#####################################



#############################MODEL BUILDING, GRIDSEARCH AND PIPELINES#####################################


#############################MODEL EVALUATION (METZ, ROC CURVE, CONF_MAT)#####################################


#################################API-REQUESTS#####################################



class APICaller:
    def __init__(self, base_url, token=None, ignore_token=False):
        self.token = os.getenv('TOKEN')
        if ignore_token==False:
            if len(self.token) == 0:
                raise ValueError('Missing API token!')
        self.base_url=base_url
        
    def retrieve_one(self,url_extension,location=None, date=None, date1=None):  
        if date1!=None:
            response = req.get(self.base_url+url_extension+f'{location}/StartDate={date}/EndDate={date1}/Json').json()
        elif (date!=None and date1==None):
            response = req.get(self.base_url+url_extension+f'{location}/Date={date}/Json').json()
        else:
            print(self.base_url+url_extension)
            response = req.get(self.base_url+url_extension).json()
        return response
    
    
    def retrieve_many(self,location_list, date_list, var, limit):
        data = []
        counter=0
        for location in location_list:
            for date in date_list:
                if counter==limit-1:
                    time.sleep(60)
                response = req.get(f'{self.url}/{key}/{location}/{date}/{var}').json()
                data.append(response)
                counter+=1
        data_df = pd.read_json(data)    
        return data_df


def is_london(x):
    if re.findall('london', x.lower()):
        return 'London'
    else:
        return x
    