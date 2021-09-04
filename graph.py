import nltk
from nltk.corpus import stopwords
from selenium import webdriver



PATH = "C:/Program Files (x86)/chromedriver.exe"

driver = webdriver.Chrome(PATH)
from nltk.tokenize import word_tokenize 
set(stopwords.words('english'))
text = "depression"
stop_words = set(stopwords.words('english')) 

 
word_tokens = word_tokenize(text) 
    
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 

filtered="+".join(filtered_sentence)
print(filtered)
driver.get("https://www.youtube.com/results?search_query=motivational+video+to+overcome+"+filtered)
user_data = driver.find_elements_by_xpath('//*[@id="video-title"]')
links = []
for i in user_data:
    links.append(i.get_attribute('href'))
    #https://www.youtube.com/embed/tgbNymZ7vqY
driver.quit()
new_url_list = list()
for address in links:
    new_address = address.replace("watch?v=", "embed/")
    new_url_list.append(new_address)

links = new_url_list
print(links)