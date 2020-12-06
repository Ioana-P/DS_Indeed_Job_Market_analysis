# Exploring the UK Data Science job market through Indeed job posts

I scraped 1098 job posts from Indeed.co.uk, searching for 3 different types of data science roles via the title, and then analysed the data. Please bear in mind that all the data was off one site only (I hope to be able to get job data off LinkedIn/Glassdoor at some point in the future) in late Nov 2020.

Bear in mind that this is an **exploratory project** - findings should be taken with a pinch of salt, and really the results need to be corroborated with a replication of the project in 2021 or with other research. 

### Findings 
1. Most Data Science jobs do not advertise their salary openly (only 58.5% do)
2. The median Data Science salary is about £45k; the mean is about £51k
3. The majority of roles are based in London
4. The median Data Analyst role is under £40k, whereas the median Data Scientist and Machine Learning jobs are above £60k. Roles with "Scientist" and/or "Machine learning" earn £20k more (**on average**) than roles with "Analyst" in the title. 

![](https://github.com/Ioana-P/DS_Indeed_Job_Market_analysis/blob/master/fig/dist_annual_salaries_ALL.jpg)

5. London and Cambridge have the highest average salaries; the former has the widest spread of salaries
6. If you're a median-salary London Data Analyst, you're getting paid as well as the median salary for all Data Science roles outside the company.
7. Just over 14% of overall roles are looking for a "Senior" hire. Less than 2% explicitly advertise for a 'Junior' hire. 
8. The most popular langauges / skills (based on our search) were: Python, SQL, R, AWS and Azure (in decreasing order of mentions). 

![](https://github.com/Ioana-P/DS_Indeed_Job_Market_analysis/blob/master/fig/percentage_mentions_by_group.jpg)

9. Using feature engineering and 3 different iterations of models, I was not able to predict salary from job description and title data. This indicates that either more data is needed (likely since there only 274 data points in the training data) or that it's not possible to reliably predict the salary from such data. 
10. There was insufficient data to determine if years of experience required / requested correlated with annual salary. 
11. After using topic modelling, a few topics stood out - most noticeably the `Academic_&_Research` topic which seems to be the only one that has a moderate correlation with salary.


Full method and implementation in notebook - Data_Scientist_UK_Q4_Job_market_analysis.ipynb


#### Repo navigation:
- **index.ipynb** - principal notebook; questions and project plan located there. 
- **scrape_indeed_gui.py** - script for running Job Scraper Tool (built using pySimpleGui)
- archive/
-  cleaning.ipynb - notebook to check the outputs of the scraping tool's results
-  unititled.ipynb - nb used to load and check extraneous data (e.g. ONS salary information for sector)
-  clean_data/ - folder preprocessed data
-  raw_data/ folder including data as immediately outputed after webscraping stage
-  LDA_vis_plot.html - interactive visualisation of Latent Dirichlet Allocation.
-  functions.py - scripted functions and classes stored here, including webscraping tool
-  topic_mod.py - functions for topic modelling 
-  fig/ - noteable visualisations saved here
 


### How-to replicate:
1. Data collection:
* Run 
>> python scrape_indeed_gui.py >>
* in your terminal. There will be 3 option buttons: Scrape, Retrieve and Clean. Click on Scrape, then enter the job search term, e.g. data scientist. Including quotes ("data scientist") will only return job posts that have that exact phrase somewhere. I recommend scraping by title - i.e. title:(data,scientist). Careful with spaces inside the brackets, that can affect search results (Indeed's search engine is a bit overly sensitive). 
* The first part of the scraping will involve the job post scraper opening a page of search results on an automated chrome page. I recommend return to this open window periodically and checking the number of jobs stored to see if the nr is stagnating. In that case, it might be worth scrolling up/down the page a bit. If Indeed returns 500 jobs for a particular search, I recommend you search for 450 of those jobs - I have not yet been able to get the Selenium Chrome driver to consistently find hrefs in the middle of the page. That will be fixed another time. 
* Once all the hrefs are stored, a headless Chrome driver will get to work opening and closing each one individually, retrieving the job post description and title. This will be headless - i.e. no chrome window will pop up open. This stage will take a while and you will get an update in your terminal every 50 jobs that are scraped. 
* Once done all the scraped job posts will be stored in the destination you specified initially (e.g. as a csv inside the raw_data folder)

2. Clean data:
* most of the data is cleaned automatically through Clean process. I recommend opening up a notebook and inspecting the raw data file once, and then using the gui to clean it thorroughly. 
* to clean using GUI, open the GUI again, click "Clean", enter location of raw_data (including the file type at end) and specify a destination file name (**without** the .csv filetype) - e.g. raw_data/RAW_DS_2020_11.csv; clean_data/CLEAN_DS_2020_11
* load the clean file into a separate nb and do quick, commonsense checks. Recommend checking for duplicates as a priority (I have added loads of steps in the scraper to remove duplicate jobs, yet still, miraculously, one comes out every now and again); you will notice a large amount of null values, first in salary and then in Location - the first one is because a lot of jobs don't advertise salary and the second because Location will only be filled with what you specify as the location search when scraping. So if you leave it blank, i.e. search for the entire UK, it will return nan in those fields, and the location from title / text will be done in the cleaning process. 

3. Most of the rest of the analysis should be replicatable by following teh code in the Data_Scientist_UK_Q4_Job_market_analysis notebook. It's just a lot of Pandas wrangling and seaborn/plt work. There are, however, a few parts that are a bit more opaque:
* Notebook section 5 - the function used is functions.get_pattern_count_and_store(), which goes through the specified input columns and searches explicitly for the terms you stated in your list and stored the outputs as columns. get_pattern_count_and_store() calls on get_pattern_count() which looks for all the possible mentions of the particular pattern in the text using regex. It's a bit scrappy, to say the least, primarily because it can't handle C++ or C# (which is why there's a totally different if statement for handling those particular 2 examples), but after a lot of testing, it hasn't thrown up any other errors ... yet. 
* the function get_experience() functions in a very similar way, but you would just use it directly on a single column with a lambda function, as shown at the start of section 6. The outputs require a lot more cleaning, generally, as I show in the same section. 
* Section 7 - many things need explaining here: 
    - LemmaTokenizer class - just a tokenizer that also lemmatizes, wrapper class for WordNetLemmatizer;
    - get_top_n_terms - wrapper fn that uses the nltk FreqDist function but gives a dataframe output instead. I've not been able to stop the FreqDist graphs from popping up so always create the dataframes in one cell, with the %%capture magic at the top if you don't want your screen filling up with really hard to read graphs
    - topic.data_to_lda - wrapper fn that takes a pandas series and turns it into LDA data (i.e. if you have 5 topics, each document gets 5 scores, 1 per topic); 
    - how to use the pyLDAviz tool is probs explained a lot better by someone else [here](https://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/pyLDAvis_overview.ipynb) and also by me (in 3rd part of article) [here](https://towardsdatascience.com/latent-dirichlet-allocation-intuition-math-implementation-and-visualisation-63ccb616e094)

#### References:

<!-- * ONS Sector Data - https://www.ons.gov.uk/filters/c60ed96a-df5b-4dbe-bbda-ab407c9639d6/dimensions  -->
* SlashData Report - 'State of the Developer Nation 19th Edition' - https://slashdata-website-cms.s3.amazonaws.com/sample_reports/y7fzAZ8e5XuKCL1Q.pdf 
<!-- * Assumptions of Point Biserial Correlation - https://statistics.laerd.com/spss-tutorials/point-biserial-correlation-using-spss-statistics.php -->
<!-- * Point Biserial Correlation with Python - https://towardsdatascience.com/point-biserial-correlation-with-python-f7cd591bd3b1 -->
* Assumptions of Ordinal (Linear) Regression

#### Key Assumptions to bear in mind:
* **Data sourcing** - Data was sourced purely from Indeed.co.uk over a limited time span. The individual time of scraping is recorded for each job post inside the raw_data file raw_data/SE_jobs_20_11_2020.csv. Further sampling over time will be needed to replicate or falsify findings.
* **National averages** - ONS data was filtered to retrieve data for 2019 as 2020 data is a. not fully available yet broken down by and b. 2020 survey data was affected by the Covid19 lockdown and the move to telephone polling. Although the 2019 mean salary is not a fair comparison for data retrieved in Q3 2020, it provides a rough benchmark against which we can compare our sample. 
* **Data mining** - The salary extraction method is reliant on the pattern finding of text that "looks like salary data" - i.e. the python script that I wrote searched specifically for text that included "£" followed by any length of numbers (continuous or punctuated by a comma) and a time period phrase such as "per day", "per annum", "a year", etc. Spot-checking showed the method to be robust. Where a range was stated (e.g. "£40,000 - £50,000") the mean was taken. The same applied to regex mining for programming languages: particular challenges were encountered where languages had many variants (JavaScript for instance) or even different spelling (Javascript) - I tried to capture as many as feasible in the regex patterns. It might seem strange that the values for C++ and C# came out near zero, but, after multiple rounds of testing, I still could not find any, even though my function searching for those two languages had specific conditions written to ensure that they were detected.
* **Webscraping process** - The webscraping tool generally managed to retrieve the maximum of 19 jobs posted per page on Indeed's search result pages. The first part of the scraping had a human-in-the-loop (i.e. myself) monitoring the scraper navigating the page, to ensure data retrieval quality. Occasionally, due to an unforeseen pop-up or a nested html element, I had to scroll the automated Chrome browser in the right direction so that it would carry on retrieving new job post URLs. However, this means that some job posts may have been overlooked. This shouldn't pose a significant problem, since I was sampling from the much larger number of total available job posts, but the sampling method's randomness is reliant on the order in which Indeed presents results. 
* **Searching by title** - since we're relying on Indeed.co.uk's internal search functions, I specifically chose to input the text "title:(software, developer)"- this would only bring up a job if it contained either or both of those 2 words in any order. This would allow as well for examples such as "Software & Architecture Developer" and other valid variations. I realise of course that this would exclude perfectly valid choices from our selection, but I decided to prioritise sample purity foremost.
