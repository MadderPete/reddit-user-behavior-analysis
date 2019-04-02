
# coding: utf-8

# ## 1. Scraping the data...
# This is the very first process of our project, we scrape the data from reddit which contains the title, text, ups(how many people love the post), downs(how many people don't like the post), comment numbers, main comment's scores and contents, sub comment's scores and contents and the creat time of all the posts and comments. In the end, we save this result into a csv file named reddit_ten_years.csv
# All the posts are from 01/10/2013 to 01/11/2013

# In[ ]:


# try to write the content into csv file
import pandas as pd
import numpy as np
import praw
from bs4 import BeautifulSoup
import requests
import re
import datetime
from psaw import PushshiftAPI
# the psaw package can help us to scarpe data from different time series
around = 0
# first let's find a way to get the text of the web
def get_text(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    a=soup.get_text()
    pattern='\w+\w'
    match= re.findall(pattern, a)
    b=' '
    return b.join(match)

#convert the time
def convert_time(time):
    return datetime.datetime.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
# The password was changed for security, but it doesn't matter the result
reddit = praw.Reddit(client_id ='S2Y4qpYTuRZ49w',
                    client_secret='B-NZvkDbwhs9Go85YVFa67RYJ5o',
                    username='madBearPete',
                    password='#######',
                    user_agent='prawtest1'
                    )

unit_title=[]
unit_ups=[]
unit_downs=[]
unit_ratio=[]
unit_text=[]
unit_creat=[]
comment_count=[]
comment_mainlist_body=[]
comment_mainlist_score=[]
comment_sublist_body=[]
comment_sublist_score=[]
comment_mainlist_creat=[]
comment_sublist_creat=[]
api = PushshiftAPI(reddit)# try to get data from 5 years ago
#after means the start time and before means the end time in the time series
#subreddit means the community
#filter means how to chose the post
submissions=api.search_submissions(after=1380600000,
                            before=1383278400,
                            subreddit='worldnews',
                            filter=['url','author', 'title','subreddit'],
                            limit=10000)
for submission in submissions:
    # if the post is sticked from other community, drop it
    if not submission.stickied:
        url=submission.url
        unit_title.append(submission.title)
        unit_ups.append(submission.ups)
        unit_downs.append(submission.downs)
        unit_creat.append(convert_time(submission.created_utc))
        try:
            unit_text.append(get_text(url))
        except Exception:
            print('a bad web')
            print(url)
            unit_text.append('error')
            pass
        unit_ratio.append(submission.upvote_ratio)
        comment_count.append(submission.num_comments)
        submission.comments.replace_more(limit=0)
        comments=submission.comments.list()
        commentlist1=[]#main comment body
        commentlist2=[]#sub comment body
        commentlist3=[]#main comment score
        commentlist4=[]#sub comment score
        commentlist5=[]#main comment creat time
        commentlist6=[]#sub comment creat time
        for comment in comments:
            if comment.parent()==submission.id:
                commentlist1.append(comment.body)
                commentlist3.append(comment.score)
                commentlist5.append(convert_time(comment.created_utc))
            else:
                commentlist2.append(comment.body)
                commentlist4.append(comment.score)
                commentlist6.append(convert_time(comment.created_utc))
        for i in range(max(len(commentlist1),len(commentlist2))-1):
            unit_title.append('')
            unit_ups.append('')
            unit_downs.append('')
            unit_text.append('')
            unit_ratio.append('')
            comment_count.append('')
            unit_creat.append('')
        if len(commentlist1)>len(commentlist2):
            for j in range(len(commentlist1)-len(commentlist2)):
                commentlist2.append('')
                commentlist4.append('')
                commentlist6.append('')
        else:
            for p in range(len(commentlist2)-len(commentlist1)):
                commentlist1.append('')
                commentlist3.append('')
                commentlist5.append('')
        if len(commentlist1)+len(commentlist2)==0:
            comment_mainlist_body.append('')
            comment_mainlist_score.append('')
            comment_mainlist_creat.append('')
            comment_sublist_body.append('')
            comment_sublist_score.append('')
            comment_sublist_creat.append('')

        comment_mainlist_body=comment_mainlist_body+commentlist1
        comment_mainlist_score=comment_mainlist_score+commentlist3
        comment_mainlist_creat=comment_mainlist_creat+commentlist5
        comment_sublist_body=comment_sublist_body+commentlist2
        comment_sublist_score=comment_sublist_score+commentlist4
        comment_sublist_creat=comment_sublist_creat+commentlist6
        around = around+1
        print(around)
        
example ={'title':unit_title,'ups':unit_ups,'downs':unit_downs,'text':unit_text,
          'comment_count':comment_count,'score_ration':unit_ratio,'creat_time':unit_creat,'main_comment':comment_mainlist_body,
          'main_score':comment_mainlist_score,'main_creat':comment_mainlist_creat
                  ,'sub_comment':comment_sublist_body,'sub_score':comment_sublist_score,'sub_creat':comment_sublist_creat}
#print(unit_title)
#print(example)
print(len(unit_title))
print(len(unit_ups))
print(len(unit_downs))
print(len(unit_text))
print(len(comment_mainlist_body))
print(len(comment_mainlist_score))
print(len(comment_sublist_body))
print(len(comment_sublist_score))
data = pd.DataFrame(example)
data.to_csv('/Users/hongyu/Downloads/BIA 660/new_reddit_ten_years.csv',index=False,encoding='utf-8')


# ## 2. Processing the data ...
# ### 2.1. calculate the tf_idf matrix first 
# We just define two functions here, get the tf_idf matrix and similarity of the posts. 
# Besides the positive/negative word list,we import the bad-words list created by MIT.

# In[5]:


import pandas as pd
import numpy as np
import nltk,re,string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.spatial import distance
from sklearn.preprocessing import normalize
row_data = pd.read_csv('/Users/hongyu/Downloads/BIA 660/new_reddit_ten_years.csv')
stop_words = stopwords.words('english')
with open('/Users/hongyu/Downloads/BIA 660/files/positive-words.txt','r') as f:
    positive_words=[line.strip() for line in f]
with open('/Users/hongyu/Downloads/BIA 660/files/negative-words.txt','r') as p:
    negative_words=[line.strip() for line in p]
with open('/Users/hongyu/Downloads/BIA 660/groupwork/bad-words.txt','r') as d:
    dirty_words=[line.strip() for line in d]
def tf_idf(docs):
    dtm=pd.DataFrame.from_dict({ind:values for (ind,values) in enumerate(docs)},orient='index')
    dtm=dtm.fillna(0)
    tf=dtm.values
    doc_len=tf.sum(axis=1)
    tf=np.divide(tf.T, doc_len).T
    df=np.where(tf>0,1,0)
    print('done')
    idf=np.log(np.divide(len(docs),np.sum(df, axis=0)))+1
    tf_idf=normalize(tf*idf)
    return tf_idf
def similar(docs):
    n_tidf=tf_idf(docs)
    similarity =1-distance.squareform(distance.pdist(n_tidf, 'cosine'))
    return similarity  


# In[ ]:


import time
# this two method is used to calculate the post's lasting time, covert the normal time to unixtime, then calcluate it
# to the lasting days
def covert_to_unixtime(s):
    return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())
def last_time(t):
    return int(datetime.datetime.utcfromtimestamp(t).strftime('%d'))


# ### 2.2. Sentiment analysis
# This is the sentiment analysis part, here we find a way to calculate the sentiment scores of the post's comments.
# The idea is that we calculate the frequent distribution in each comment,and then find how many words are included in negative&positive words file, and sum their frequence, then the result can be seen as the sentiment score.
# We also use the frequency of bad words contain in each comment to represent the civilized level of reddit users.
# Also, the mimic, sentiment scores are calcluted by this function.

# In[ ]:


import pandas as pd
import string
import time
import datetime
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import operator

sid = SentimentIntensityAnalyzer()
# get the positive and negative score
main_score=[]
sub_score=[]
main_latest=[]
sub_latest=[]
#this is the sentiment score, which is different from the count of sentiment words
sentiment_m=[]
sentiment_s=[]
#the mimic from comments
m_mimic=[]
s_mimic=[]
# the dirty words from comments
m_dirty=[]
s_dirty=[]
#this is for the whole mimic
row_mimic=[]
row_data = pd.read_csv('/Users/hongyu/Downloads/BIA 660/new_reddit_recent.csv')
#process the data in order to deal with them easier
test = row_data[['comment_count',"main_comment","main_score",'main_creat',"sub_comment","sub_score",'sub_creat']].fillna(0)
test[['title','creat_time']]=row_data[['title','creat_time']].fillna(method='ffill')
count=0
original=row_data[['title','creat_time']].dropna()
for i in range(len(original)):
    disgussion=original.iloc[i]
    print(disgussion.title)
    AM_count = 0
    NM_count = 0
    AS_count = 0
    NS_count = 0
    DM_count = 0
    DS_count = 0
    #calculate the score
    sentiment_mm=[]
    sentiment_ss=[]
    main=''
    sub=''
    time_main=[]
    time_sub=[]

    for order in range(count,len(test)):
        # add the time so when there is no comment, the program won't cause mistakes
        time_main.append(covert_to_unixtime(test.iloc[order].creat_time))
        time_sub.append(covert_to_unixtime(test.iloc[order].creat_time))
        # deal with the main comment
        if (test.iloc[order].title==disgussion.title and test.iloc[order].creat_time==disgussion.creat_time 
            and test.iloc[order]['main_comment']!=0):
            #this print is used to debug, in order to understand where the bug is, the same with the prints beblow
            print('round one',order)
            #get the score of the comment
            sentiment_mm.append(sid.polarity_scores(test.iloc[order]['main_comment'])['compound'])
            #add the word in to mimic box
            row_mimic.append(test.iloc[order]['sub_comment'])
            main=main +' '+test.iloc[order].main_comment
            if (test.iloc[order]['main_creat']!=0):
                time_main.append(covert_to_unixtime(test.iloc[order].main_creat))
            else:
                time_main.append(covert_to_unixtime(test.iloc[order].creat_time))
        #deal with the sub comment
        if (test.iloc[order].title==disgussion.title and test.iloc[order].creat_time==disgussion.creat_time 
            and test.iloc[order]['sub_comment']!=0):
            print('round two',order)
            #the socre of the sub comment
            sentiment_ss.append(sid.polarity_scores(test.iloc[order]['sub_comment'])['compound'])
            #add the word into mimic box
            row_mimic.append(test.iloc[order]['sub_comment'])
            # active and negative score for sub comment
            sub=sub+' '+test.iloc[order].sub_comment
            if (test.iloc[order]['sub_creat']!=0):
                time_sub.append(covert_to_unixtime(test.iloc[order].sub_creat))
            else:
                time_sub.append(covert_to_unixtime(test.iloc[order].creat_time))
            time_sub.append(covert_to_unixtime(test.iloc[order].sub_creat))
            
        # when it comes to the next post, break the loop, turn to the next one    
        elif(test.iloc[order].title!=disgussion.title or test.iloc[order].creat_time!=disgussion.creat_time):
            print('round three',order,test.iloc[order].title)
            count=order
            break
            
        # when there is no comments
        elif(test.iloc[order].title==disgussion.title and test.iloc[order].creat_time==disgussion.creat_time 
             and test.iloc[order]['main_comment']==0 
             and test.iloc[order]['sub_comment']==0):
            print('round four :',order)
            if(test.iloc[order+1].title!=disgussion.title or test.iloc[order+1].creat_time!=disgussion.creat_time):
                time_main.append(covert_to_unixtime(test.iloc[order].creat_time))
                time_sub.append(covert_to_unixtime(test.iloc[order].creat_time))
                count=order+1
                break
            else:
                print('nah')
                pass
        # when there is only sub comment   
        elif(test.iloc[order].title==disgussion.title and test.iloc[order].creat_time==disgussion.creat_time 
            and test.iloc[order]['sub_comment']==0):
            print('round five',order)
            time_sub.append(covert_to_unixtime(test.iloc[order].creat_time))
            
    
    # get the slangs or mimic or you can say the new language style of reddit users        
    # this is the main comments
    # use the strings stored in main
    # get the top single words
    
    if main!='':
        vect1 = CountVectorizer(ngram_range=(0,1),stop_words='english')
        analyzer1 = vect1.build_analyzer()
        listNgramQuery1 = analyzer1(main)
        NgramQueryWeights1 = nltk.FreqDist(listNgramQuery1)
        # get the top pharse
        vect2 = CountVectorizer(ngram_range=(2,4))
        analyzer2 = vect2.build_analyzer()
        listNgramQuery2 = analyzer2(main)
        NgramQueryWeights2 = nltk.FreqDist(listNgramQuery2)
        #get the main freqdist top 20 words
        m_top_20 = dict(NgramQueryWeights2+NgramQueryWeights1)
        m_top_20 = sorted(m_top_20.items(),key=operator.itemgetter(1),reverse=True)[0:20]

        m_mimic.append(m_top_20)
    else:
        m_mimic.append('no comments')
    
    # this is the sub comments
    # use the strings stored in sub
    if sub!='':
        s_vect1 = CountVectorizer(ngram_range=(0,1),stop_words='english')
        s_analyzer1 = s_vect1.build_analyzer()
        s_listNgramQuery1 = s_analyzer1(sub)
        s_NgramQueryWeights1 = nltk.FreqDist(s_listNgramQuery1)
        # get the top pharse
        s_vect2 = CountVectorizer(ngram_range=(2,4))
        s_analyzer2 = s_vect2.build_analyzer()
        s_listNgramQuery2 = s_analyzer2(sub)
        s_NgramQueryWeights2 = nltk.FreqDist(s_listNgramQuery2)
        #get the main freqdist top 20 words
        s_top_20 = dict(s_NgramQueryWeights2+s_NgramQueryWeights1)
        s_top_20 = sorted(s_top_20.items(),key=operator.itemgetter(1),reverse=True)[0:20]

        s_mimic.append(s_top_20)
    else:
        s_mimic.append('no comments')
    #other frequency
    mwordlist=main.strip().lower().split()
    swordlist=sub.strip().lower().split()
    freq_mdist = nltk.FreqDist(mwordlist)
    freq_sdist = nltk.FreqDist(swordlist)
    #print('heyyyyyyyyyyyyyy',mwordlist)
    pm_inter = list(set(mwordlist).intersection(set(positive_words)))
    nm_inter = list(set(mwordlist).intersection(set(negative_words)))
    ps_inter = list(set(swordlist).intersection(set(positive_words)))
    ns_inter = list(set(swordlist).intersection(set(negative_words)))
    dirty_m=list(set(mwordlist).intersection(set(dirty_words)))
    dirty_s=list(set(swordlist).intersection(set(dirty_words)))
    for word,freq in freq_mdist.items():
        if word in dirty_m:
            DM_count=DM_count+freq
        if word in pm_inter:
            AM_count=AM_count+freq
        if word in nm_inter:
                NM_count=NM_count+freq
    for word,freq in freq_sdist.items():
        if word in dirty_s:
            DS_count=DS_count+freq
        if word in pm_inter:
            AS_count=AS_count+freq
        if word in nm_inter:
            NS_count=NS_count+freq
    print(order)
    m_dirty.append(DM_count)
    s_dirty.append(DS_count)
    main_score.append((AM_count,NM_count))
    sub_score.append((AS_count,NS_count))
    main_latest.append(sorted(time_main)[-1])
    sub_latest.append(sorted(time_sub)[-1])
    # get the average sentiment score
    sentiment_m.append(sum(sentiment_mm)/(len(sentiment_mm)+0.00000001))
    sentiment_s.append(sum(sentiment_ss)/(len(sentiment_ss)+0.00000001))


# ### 2.3. Combine all the data
# This step, we merged all the data we need and transfer them into a new dataframe score. 
# All the clusterings and visulizations are based on this dataframe.

# In[1]:


#get the latest reply of a post
latest=[main_latest[i] if main_latest[i]>sub_latest[i] else sub_latest[i] for i in range(len(main_latest))]
#calculate the lasting time of a post
lasting = [(last_time(latest[i]-covert_to_unixtime(list(row_data.creat_time.dropna())[i]))) for i in range(len(latest))]
score1=pd.DataFrame(main_score)
score2=pd.DataFrame(sub_score)
score3=pd.DataFrame(lasting)
score4=pd.DataFrame(sentiment_m)
score5=pd.DataFrame(sentiment_s)
score6=pd.Series(m_mimic)
score7=pd.Series(s_mimic)
score8=pd.DataFrame(m_dirty)
score9=pd.DataFrame(s_dirty)
score0=pd.DataFrame(row_data[['title','comment_count','ups','downs']].dropna().values)
#combine all of the data
score=pd.concat([score0,score1,score2,score4,score5,score6,score7,score8,score9,score3],axis=1)
score.columns=['title','comment_count','ups','downs','main_pos','main_neg','sub_pos','sub_neg',
               'sentiment_main','sentiment_sub','main_mimic','sub_mimic','main_dirty','sub_dirty','last']
a=row_data[row_data['title'].notna()]
score['creat_time']=[int(i[11:13]) for i in a.creat_time.values]
print(len(a))
score['total_dirty']=(score['main_dirty']+score['sub_dirty'])/score['comment_count']
score=score[['title','comment_count','ups','downs','main_pos','main_neg','sub_pos','sub_neg',
               'sentiment_main','sentiment_sub','main_mimic','sub_mimic','main_dirty','sub_dirty','total_dirty','last']]
#save the data
score.to_csv('/Users/hongyu/Downloads/BIA 660/score_ten_year.csv')


# ## 3. Cluster the data
# ### 3.1. Use hierarchical clustering
# We used scipy package to get the hierarchical cluster of the posts.Then we find a good parameter 0.95411 as the distance of the cluster, in this way, the cluster is similar to our previous prediction. To get the correct cluster distance, we can randomly test different values, when it reaches 0.95411, the cluster's first 100 results is most similar to what we've done by hand.
# 
# Then we labeled the cluster with the label we used in the previous prediction.

# In[17]:


import pandas as pd
score=pd.read_csv('/Users/hongyu/Downloads/BIA 660/score_ten_year.csv')
# we scrape the data which comment counts is bigger than 10, so we have the data which can be seen hot
# because most of the posts have no comment, which means they can't trigger reddit user's interests
ten_year=score[score.comment_count>=10]


# In[18]:


import scipy
import scipy.cluster.hierarchy as sch
import seaborn as sns
import matplotlib.pyplot as plt
#use the title to do the clustering
aas=ten_year['title'].dropna()
aas=[nltk.FreqDist(row.strip().lower().split()) for row in aas if row.strip()
        not in stop_words and row.strip()not in string.punctuation]
TF2=tf_idf(aas)
disMat2 = sch.distance.pdist(TF2,'cosine') 
Z2=sch.linkage(disMat2,method='average') 
plt.figure(figsize=(40,20))
P=sch.dendrogram(Z2)
plt.show()


# In[20]:


# with the value of 0.95411, we got the best clusters and this method is used to find the index of the cluster
# and compare them with the cluster we did by hand
t_cluster=sch.fcluster(Z2,0.95411, criterion='distance')
ten_year['cluster']=t_cluster
c=t_cluster[0:100]
b=np.where(c==156)
print(len(np.unique(c)))
print(c)
print(len(b[0]))
print(b)
print(len(np.unique(t_cluster)))


# In[21]:


#label the cluster
cluster_label={'156':'president obama','138':'environment','157':'NSA international spying','128':'mideast teenager female right situation','62':'Palestine and Israel conflict',
               '129':'human conficts','90':'US drone kill','87':'nuclear between US and Iran','146':'America with Snowden','159':'google with NSA','117':'canada primier scandal'}
lt_cluster=t_cluster.tolist()
for index,key in cluster_label.items():
    for i in range(len(lt_cluster)):
        if lt_cluster[i]==int(index):
            lt_cluster[i]= key
ten_year['ups']=[int(i) for i in ten_year['ups']]
t_cluster=np.array(lt_cluster)
ten_year['cluster_label']=t_cluster


# In[22]:


ten_year.cluster


# ### 3.2. Data visualization
# This is the distribution of the topic among these posts, we can see that the murder case in saudi is really a big topic in reddit, over one forth people are discussing it. And about a half topic are some other interesting news around the world, such as a girl has pullen a sword from stone and so on, these news are hard to clustered. 
# 
# Then the top 10 topics are president (Obama Spying), environment, NSA international spying, Mideast teenager female rights, Palestine and Israel conflict, human conflicts, US drone kill, nuclear between US and Iran, American with Snowden, and google with NSA.

# In[23]:


# get the percentage of different cluster
a=ten_year['cluster_label'].value_counts()[0:11].rename(' ')
print(a)
colors = ['tomato', 'red', 'darkgrey','blue','goldenrod', 'green', 'y']
a.plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0,0.1,0,0,0,0,0,0,0,0,0],colors=colors)
plt.title('Distribution differernt cluster')
fig=plt.gcf()
fig.set_size_inches(7,7)
plt.show()
#get the posts contained in different clusters
ten_year.groupby('cluster')['title'].count().plot(color='y')
fig=plt.gcf()
fig.set_size_inches(12,6)


# In[25]:


#calculate the dirty rate per comment in 2013
ten_year['total_dirty'].plot(color='y')
fig=plt.gcf()
fig.set_size_inches(12,6)
ten_year['total_dirty'].mean()


# In[28]:


#calculate the average main comment sentiment score in 2013
ten_year['sentiment_main'].plot(color='y')
fig=plt.gcf()
fig.set_size_inches(12,6)
ten_year['sentiment_main'].mean()


# In[29]:


#calculate the average sub comment sentiment score in 2013
ten_year['sentiment_sub'].plot(color='y')
fig=plt.gcf()
fig.set_size_inches(12,6)
ten_year['sentiment_sub'].mean()


# In[30]:


# calculate the average scores of each cluster which can represent how much reader like the topic
Tsort=ten_year.ix[[ten_year['cluster_label'].iloc[i] in cluster_label.values() for i in range(len(ten_year))]]
Tsort1=Tsort.groupby('cluster_label',as_index=False).mean()
plt.figure(figsize=(40,20))
A=Tsort1.groupby('cluster_label',as_index=False).mean()
ax2 = sns.barplot(x='cluster_label',y='ups', data=A)
ax2.set_xlabel('Different clusters', fontsize=15)
ax2.set_ylabel('Average scores',  fontsize=15)
ax2.set_title('different average scores of the cluster', fontsize=20)


# In[31]:


# the average attitude word counts in main comment per topic(this is differernt from the sentiment score,
# this is the count of sentiment word)
plt.figure(figsize=(30,15))
C=A[['cluster_label','main_pos','sub_pos']]
D=A[['cluster_label','main_neg','sub_neg']]
E=A[['cluster_label','sentiment_main']]
F=A[['cluster_label','sentiment_sub']]
E['sentiment']='main'
F['sentiment']='sub'
C['type']='active'
D['type']='negative'
C.columns=['cluster','main_attitude','sub_attitude','type']
D.columns=['cluster','main_attitude','sub_attitude','type']
E.columns=['cluster','attitude','sentiment']
F.columns=['cluster','attitude','sentiment']
A=C.append(D,ignore_index=True)
B=E.append(F,ignore_index=True)
ax3 = sns.barplot(x='cluster',y='main_attitude',hue='type', data=A)
ax3.set_xlabel('Different clusters', fontsize=15)
ax3.set_ylabel('average attitude',  fontsize=15)
ax3.set_title('different attitude scores of the cluster', fontsize=20)


# In[32]:


# the average attitude word counts in main comment per topic(this is differernt from the sentiment score,
# this is the count of sentiment word)
plt.figure(figsize=(30,15))
ax4 = sns.barplot(x='cluster',y='sub_attitude',hue='type', data=A)
ax4.set_xlabel('Different clusters', fontsize=15)
ax4.set_ylabel('average attitude',  fontsize=15)
ax4.set_title('different sub attitude scores of the cluster', fontsize=20)


# In[2]:


#get the count of the frequent word in comments
#however, this result is hard to translate
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
def get_count(x):
    x=' '.join(x)
    s_vect1 = CountVectorizer(ngram_range=(0,1),stop_words='english')
    s_analyzer1 = s_vect1.build_analyzer()
    s_listNgramQuery1 = s_analyzer1(x)
    print(s_listNgramQuery1)
    s_vect2 = CountVectorizer(ngram_range=(2,4))
    s_analyzer2 = s_vect2.build_analyzer()
    s_listNgramQuery2 = s_analyzer2(x)
    print(s_listNgramQuery2)
    result=s_listNgramQuery1+s_listNgramQuery2
    print(result)
    #get the main freqdist top 20 words
    return result
x=get_count(row_mimic)
def plotword(a):
    wordcloud = WordCloud(stopwords=STOPWORDS,background_color='black',width=1500,
                          height=1500,max_words=100,collocations=True).generate(' '.join(a))
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
plotword(x)
# the output should be like as below


# ![expression%20ten%20year.png](attachment:expression%20ten%20year.png)

# ### 3.2. Some other comparison
# About the data we've got from 01/10/2018 to 01/11/2018, most of the process are the same, so we won't provide the data processing process in this file, but just some result will be added.

# ![dirty%20talk%20recent.png](attachment:dirty%20talk%20recent.png) this is the dirty talk rate per comment in 2018, the value is 0.4758097876552743
# 
# ![sentiment_mian_now.png](attachment:sentiment_mian_now.png)
# 
# ![sentiment_sub_recent.png](attachment:sentiment_sub_recent.png)
# these are the average sentiment score per post in 2018, the main value is -0.02774209201581032 and the sub value is -0.035908644320762906
# 
# ![scores%20of%20topic%20recent.png](attachment:scores%20of%20topic%20recent.png) this is the socre of the topic in 2018
# 
# ![recent_topic.png](attachment:recent_topic.png) this is the top ten topics in 2018
# 
# ![topic_number_recent.png](attachment:topic_number_recent.png) this is the number of topics in 2018
# 
# ![s![expression%20recent.png](attachment:expression%20recent.png)ub%20attitude%20recent.png](attachment:sub%20attitude%20recent.png)
# 
# ![recent%20count%20of%20main%20sentiment.png](attachment:recent%20count%20of%20main%20sentiment.png)
# these are the count of sentiment word in 2018
# 
# ![expression%20recent.png](attachment:expression%20recent.png)this is the mimic of 2018
