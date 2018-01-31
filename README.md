# Twitter
This project is a data science project in R, where I have created a bot identification system. 

Bot accounts are a persistent problem on twitter with around 9 to 15 percent of tweets coming from bot accounts. These tweets have the ability to spam out favorable news stories and influence politics by actions such as retweeting, liking, direct messaging. It is becoming a matter of utmost importance to classify tweets & users as bots or not from the traffic.

GOALS: 

To identify if a tweet is spam or not by looking at the words
Associate specific words with bot & human accounts
To improve the accuracy of the models by using ensemble method
To identify patterns in tweets of bots
Model should scale well with the similar datasets of equal or larger sizes


Main Findings:-

Words like Learn, Read, Check, Good were amongst the most important variables
Most of the spam tweets were advertisements for products such as shoes, music, articles, etc.
Most frequently used words by spam bots from Word Cloud were “like, can, best, follow”, etc. 
Bots were most active on Wednesdays and around midnight everyday of the week

This project was driven by our common interest and fascination for text mining. Having completed a project on R in MIS 749 class, we wanted to do something different. Twitter bots was something we closely follow. It has been a topic of discussion for us specially after 2016 election.

The discovery of this dataset began from the ides of exploring fake accounts on social media platforms. Our search initially was broad till we narrowed it down to twitter. We found a dataset that had an interesting story attached to it. Italian guys, Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., & Tesconi, M. started observing fake accounts & their activity as they follow any random new human account. They opened thousands of new accounts for the same purpose.

The Data Preparation task for this dataset was particularly painstaking. We had multiple csv files for human & Bot tweets along with the CSVs for their profile. We faced problems such as too large datasets to directly import into R, Uncleaned rows, large amount of special characters. Due to this we didn’t limit ourselves to just R. We went for python for initial text handling. We performed joins to consolidate data to have a final dataset of 4.3 million rows with 64 columns. Removed undesired columns and performed all standard with some advanced text cleaning to have a model ready dataset. 
These are the hypothesis we have formalized for our project-

Hypothesis 1:
Our team would like to predict where a tweet is a spam or not; thereby concluding if the user sending the tweet is a spambot or a genuine user.

Ho: To predict if the tweet is a spam.  
Ha: To predict if the tweet is genuine. 
 
Hypothesis 2:
In addition, we would also like to test if a particular tweet is a positive or negative tweet by performing sentiment text analysis on words used in the tweets.

Ho: To predict if the words used in the tweet are positive.
Ha: To predict if the words used in the tweet are negative.

We decided to use K fold validation approach with values of K ranging from 5-10 as suited with respective models. The training and test set distribution was consistent for all models as 85:15 respectively. 

We started modelling with GLM and rpart which gave average results. After these we moved to some complex and sophisticated models like SVM, Neural Network, Random Forest, Naive Bayes, LDA and ensemble (glm + rpart).

Detailed results and methods are explained in further sections.
In conclusion, here are some of the insights and recommendations we have
Insights:

Bots used consistent number of hashtags as compared to human accounts.
Bots tweeted most around midnight every day 
Bots were tweeting most on wednesdays so middle of the week

Recommendations- 

Apart from what we have done already, We would like to classify analysis on the basis of tweeter’s location segregated by age & gender.
We would like to expand this to analyse emoticons
Other account metrics can be used like followers, followed, likes & favorites pattern for analysis

As per the business scenario, we identified that our problem was a hybrid of text mining and classification problem. Our Response/Dependent Variable was the BotOrNot Flag which would help us predict that the tweet was from a Genuine User or a Spambot. 
In order to accomplish this, we applied a number of models on the DTM created as  explained in the section 3.6. 
The following list were the models that we used while working with the test dataset in order to get the best performing model:

 Rpart

      2.    GLM
      3.    Naïve Bayes
      4.    RandomForest
      5.    Ensemble (GLM+Rpart)
      6.    LDA (Linear Dirichlet Allocation)
      7.    Neural Net
      8.    SVM

We have extensively used the caret package for our modelling. Apart from that, for getting more improved results we decided to combine the results of two relatively opposite models (glm and rpart) but providing similar accuracy, which in other words is called Ensembling.
 For implementing the Ensemble model, we have made use of the caretEnsemble package.
In both these implementation, we have used Cross Validation as our data resampling strategy. The selection of the number of folds for modelling was driven by the complexity of the models we ran and at the same time the size of the dataset. 

