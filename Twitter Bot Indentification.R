library(doParallel)
library(tm) #for corpus and term document matrix creation/processing
library(SnowballC) #for stemming
library(wordcloud)
library(cluster)
library(rpart)


setwd("C:/aniket_study/MIS620/Project/datasets_full.csv/Consolidated Datasets")

###Set the cores. REMEMBER TO release the laptop cores at the end of the file!!!
cl <- makePSOCKcluster(50)     #register 2 cores to split of training
clusterSetRNGStream(cl, 620)  #set seed for every member of cluster
registerDoParallel(cl)
getDoParWorkers()             #list number of workers/cores


### Genuine Twitter Data consolidated
GenuineTweets<-read.csv("GenuineTweets.csv")
GenuineTweets

GenuineUsers <- read.csv("GenuineUsers.csv")
length(GenuineUsers)

GenuineTwitterData <- merge(GenuineUsers,GenuineTweets,by="user_id")

### read columns for the dataset
colnames(GenuineTwitterData)

### Remove Garbage columns 
GenuineTwitterData <- GenuineTwitterData[-c(41,42)]

### Adding predictor Column to the Genuine Data, by assignig flag value as '0' 
GenuineTwitterData$BotOrNot <- 0

##################################################################################


### Spambots 2 Twitter Data consolidated
Spambots2Tweets <- read.csv("SpamBots2.1Tweets-Geo.csv")
Spambots2Tweets

Spambots2Users <- read.csv("Spambots2.1-users.csv")
Spambots2Users

SpamBotTwitter2_Data <- merge(Spambots2Users,Spambots2Tweets,by="user_id")
SpamBotTwitter2_Data

### read columns for the dataset
colnames(SpamBotTwitter2_Data)

### Remove Garbage columns 
SpamBotTwitter2_Data <- SpamBotTwitter2_Data[-c(49)]

### Adding predictor Column to the Genuine Data, by assignig flag value as '1' 
SpamBotTwitter2_Data$BotOrNot <- 1


#################################################################################

### Spambots 3 Twitter Data consolidated
Spambots3Tweets <- read.csv("SpamBots3Tweets-Geo.csv")
Spambots3Tweets

Spambots3Users <- read.csv("Spambots3-users.csv")
Spambots3Users

SpamBotTwitter3_Data <- merge(Spambots3Users,Spambots3Tweets,by="user_id")
SpamBotTwitter3_Data

### read columns for the dataset
colnames(SpamBotTwitter3_Data)

### Remove Garbage columns 
SpamBotTwitter3_Data <- SpamBotTwitter3_Data[-c(41)]

### Adding predictor Column to the Genuine Data, by assignig flag value as '1' 
SpamBotTwitter3_Data$BotOrNot <- 1

##################################################################################

#### Creating a SpamBotsTwitter Datasets
SpamBots <- rbind(SpamBotTwitter2_Data,SpamBotTwitter3_Data)
colnames(SpamBots)

#################################################################################
#### Combining SpamBots and GenuineTwitterData Datasets for analysis 
### Modifying "id" column for countering Dataype mismatch
GenuineTwitterData$id <- as.numeric(GenuineTwitterData$id)
CompleteTwitterData <- rbind(GenuineTwitterData,SpamBots)


### Cleaning Data for unnecessary columns
colnames(CompleteTwitterData)

CompleteTwitterData.ColRemoved <- CompleteTwitterData[-c(9,11:33,35:40,43:45,47:50,53:56,61:63)]

##rm(CompleteTwitterData.ColRemoved)  Test code
colnames(CompleteTwitterData.ColRemoved)
write.csv(CompleteTwitterData.ColRemoved,"CompleteTwitterData.DateRemoved.csv")


##### Working On dates when tweets were created
head(CompleteTwitterData.ColRemoved,5)

CompletePart1 <- CompleteTwitterData.ColRemoved[1:863295,]
write.csv(CompletePart1,"ADD_time_day1.csv")
CompletePart2 <- CompleteTwitterData.ColRemoved[863296:1726591,]
write.csv(CompletePart2,"ADD_time_day2.csv")
CompletePart3 <- CompleteTwitterData.ColRemoved[1726592:2589887,]
write.csv(CompletePart3,"ADD_time_day3.csv")
CompletePart4 <- CompleteTwitterData.ColRemoved[2589888:3453183,]
write.csv(CompletePart4,"ADD_time_day4.csv")
CompletePart5 <- CompleteTwitterData.ColRemoved[3453183:4316478,]
write.csv(CompletePart5,"ADD_time_day5.csv")


ADD_time_day1 <- read.csv("ADD_time_day1.csv")
ADD_time_day2 <- read.csv("ADD_time_day2.csv")
ADD_time_day3 <- read.csv("ADD_time_day3.csv")
ADD_time_day4 <- read.csv("ADD_time_day4.csv")
ADD_time_day5 <- read.csv("ADD_time_day5.csv")

#### Combining the dataset for analysis

CompleteTwitterData.V1 <- rbind(ADD_time_day1,ADD_time_day2,ADD_time_day3,ADD_time_day4,ADD_time_day5)
colnames(CompleteTwitterData.V1)

#### Taking a subset of data for visualizations in Tableau
table(CompleteTwitterData.V1$Day,CompleteTwitterData.V1)
# Fri    Mon    Sat    Sun    Thu    Tue    Wed 
# 607488 600981 580562 583883 636991 645651 660923 

table(CompleteTwitterData.V1$Time)
# 0      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15     16 
# 223168 233417 238701 238911 218840 176581 141273 114161  98645  92995  89370  98271 118123 160038 186611 202016 207020 
# 17     18     19     20     21     22     23 
# 204353 205067 209066 213867 214574 213894 217517 

table(CompleteTwitterData.V1$Day,CompleteTwitterData.V1$BotOrNot)
#       0      1
# Fri 405116 202372
# Mon 414641 186340
# Sat 359608 220954
# Sun 390570 193313
# Thu 424279 212712
# Tue 419924 225727
# Wed 425223 235700


table(CompleteTwitterData.V1$Time,CompleteTwitterData.V1$BotOrNot)
#       0      1
# 0  133320  89848
# 1  145133  88284
# 2  154269  84432
# 3  158495  80416
# 4  150970  67870
# 5  136486  40095
# 6  115362  25911
# 7   95860  18301
# 8   80166  18479
# 9   74214  18781
# 10  71362  18008
# 11  77275  20996
# 12  88090  30033
# 13 105455  54583
# 14 118188  68423
# 15 124353  77663
# 16 127177  79843
# 17 125613  78740
# 18 124002  81065
# 19 123131  85935
# 20 127506  86361
# 21 127374  87200
# 22 126794  87100
# 23 128766  88751



#### Some of the models requring large computational power will be executed on the dataset. 
### We have made sure the distribution of entries are similar to that in the complete data set below
CTD.Reshuffle.V1 <- CompleteTwitterData.V1[sample(nrow(CompleteTwitterData.V1)),]
write.csv(CTD.Reshuffle.V1,"Subset.csv")
head(CTD.Reshuffle.V1)
CTD.Reshuffle.V1 <- CTD.Reshuffle.V1[1:300000,]

(table(as.factor(CTD.Reshuffle.V1$BotOrNot))[1])/((table(as.factor(CTD.Reshuffle.V1$BotOrNot))[1])
                                                  +(table(as.factor(CTD.Reshuffle.V1$BotOrNot))[2]))

(table(as.factor(CTD.Reshuffle.V1$BotOrNot))[2])/((table(as.factor(CTD.Reshuffle.V1$BotOrNot))[1])
                                                  +(table(as.factor(CTD.Reshuffle.V1$BotOrNot))[2]))


####### 


###### Analysis ######
#convert tweet text to character, read.csv defaults to Factor

#####
CTD.Reshuffle.V1$text <- as.character(CTD.Reshuffle.V1$text)

#gsub looks for pattern and replaces gsub(pattern, replacement, text)
#gsub("!", "", c("hi!", "hi hi hi!!! jo"))

### Checking structure of the data
str(CTD.Reshuffle.V1)

### trying to analyze tweets for cleaning data
head(CTD.Reshuffle.V1$text,20)

#convert to text multibyte encoding to UTF form
#this was neccesary after importing on Ubuntu Server, but might not be for you
#encoding differences will often need to reconciled between platforms and editors
CTD.Reshuffle.V1$text <- iconv(CTD.Reshuffle.V1$text, to="utf-8",sub="")


##regular expression
## remove letters, digits, and punctuation haracters starting with @ remove usernames and replace with "USER"
CTD.Reshuffle.V1$text <- gsub("@\\w*"," USER",   CTD.Reshuffle.V1$text)


##Remove website links and replace with "URL"
CTD.Reshuffle.V1$text  <- gsub("http[[:alnum:][:punct:]]*"," WEBADDRESS",   tolower(CTD.Reshuffle.V1$text ))
CTD.Reshuffle.V1$text  <- gsub("www[[:alnum:][:punct:]]*"," WEBADDRESS",   tolower(CTD.Reshuffle.V1$text ))


#remove html entitties like &quot; starting with 
#note perfect but we will remove remaining punctation at later step
CTD.Reshuffle.V1$text <-gsub("\\&\\w*;","", CTD.Reshuffle.V1$text)
head(CTD.Reshuffle.V1$text,20)


#remove any letters repeated more than twice (eg. hellooooooo -> helloo)
CTD.Reshuffle.V1$text  <- gsub('([[:alpha:]])\\1+', '\\1\\1', CTD.Reshuffle.V1$text)
head(CTD.Reshuffle.V1$text,20)

### replace troublesome text with blanks in the book
CTD.Reshuffle.V1$text <- gsub("*--*|*;*|*_*|*:*","",CTD.Reshuffle.V1$text)
head(CTD.Reshuffle.V1$text,20)

## Handling ascii characters
CTD.Reshuffle.V1$text <- gsub("\u2028","",CTD.Reshuffle.V1$text)
head(CTD.Reshuffle.V1$text,20)

#additional cleaning removing leaving only letters numbers or spaces
CTD.Reshuffle.V1$text <- gsub("[^a-zA-Z0-9 ]","",CTD.Reshuffle.V1$text)
head(CTD.Reshuffle.V1$text,20)

#### Create Data Back    ## Savepoint
CTD.Reshuffle.V1_Backup <- CTD.Reshuffle.V1

#create corpus and clean up text before creating docu ent term matrix
CTD.Reshuffle.V1_Corpus <- Corpus(VectorSource(CTD.Reshuffle.V1$text))

CTD.Reshuffle.V1_Corpus  <- tm_map(CTD.Reshuffle.V1_Corpus , stemDocument)
CTD.Reshuffle.V1_Corpus <- tm_map(CTD.Reshuffle.V1_Corpus, removeWords, stopwords("english"))
CTD.Reshuffle.V1_Corpus <- tm_map(CTD.Reshuffle.V1_Corpus, stripWhitespace)  
CTD.Reshuffle.V1_Corpus <- tm_map(CTD.Reshuffle.V1_Corpus, removeWords, c('user','webaddress'))

#create term document matrix (terms as rows, documents as columns)
CTD.Reshuffle.V1_tdm <- TermDocumentMatrix(CTD.Reshuffle.V1_Corpus)
CTD.Reshuffle.V1_tdm
CTD.Reshuffle.V1_tdm$nrow
CTD.Reshuffle.V1_tdm$ncol
CTD.Reshuffle.V1_tdm$dimnames[1]


CTD.Reshuffle.V2_tdm <- removeSparseTerms(CTD.Reshuffle.V1_tdm,0.995)
CTD.Reshuffle.V2_tdm
CTD.Reshuffle.V2_tdm$nrow
CTD.Reshuffle.V2_tdm$ncol
CTD.Reshuffle.V2_tdm$dimnames[1]

inspect(CTD.Reshuffle.V1_tdm[1:100,1:10])

## define tdm matrix
CTD.Reshuffle.Matrix <- as.matrix(CTD.Reshuffle.V2_tdm)
CTD.Reshuffle.Sort <- sort(rowSums(CTD.Reshuffle.Matrix),decreasing=TRUE)
CTD.Reshuffle.Sorted.df <- data.frame(word = names(CTD.Reshuffle.Sort),freq=CTD.Reshuffle.Sort)
head(CTD.Reshuffle.Sorted.df,10)

?wordcloud
### creating a word cloud
wordcloud(words = CTD.Reshuffle.Sorted.df$word, freq = CTD.Reshuffle.Sorted.df$freq, min.freq = 1,
          random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))


#lets make a bar chart of frequent words
barplot(CTD.Reshuffle.Sorted.df[1:15,]$freq, las = 2, names.arg = CTD.Reshuffle.Sorted.df[1:15,]$word,
        col ="lightblue", main ="Most frequent words",
        ylab = "Word frequencies")

CTD.Reshuffle.V2_tdm
#lets cluster the documents, but first find optimal k
wss <- numeric(15)
options(warn=1)
for (k in 1:10) wss[k] <- sum(kmeans(CTD.Reshuffle.V2_tdm, centers=k)$withinss)
plot(wss, type="b") #seems like 2 or 3 will cover it

CTD.Reshuffle.V2_tdm.kmeans <- kmeans(CTD.Reshuffle.V2_tdm,11)
CTD.Reshuffle.V2_tdm.kmeans$cluster #lets looks at cluster membership

#create Document Term Matrix (so terms are columns or attributes)
CTD.Reshuffle.V1_dtm <- DocumentTermMatrix(CTD.Reshuffle.V1_Corpus)
CTD.Reshuffle.V2_dtm <- removeSparseTerms(CTD.Reshuffle.V1_dtm, 0.99) #remove sparse terms
CTD.Reshuffle.V2_dtm
inspect(dtm[1:4,1:10])


#convert to matrix
CTD.Reshuffle.V2.labeledTerms <- as.data.frame(as.matrix(CTD.Reshuffle.V2_dtm))
head(CTD.Reshuffle.V1$BotOrNot)
tail(CTD.Reshuffle.V1$BotOrNot)
CTD.Reshuffle.V2.labeledTerms$BotOrNot <- CTD.Reshuffle.V1$BotOrNot #merge with labels

## Making sure if the predictor column is added correctly
table(CTD.Reshuffle.V2.labeledTerms$BotOrNot)

## understanding the newly created dataset
str(CTD.Reshuffle.V2.labeledTerms)
ncol(CTD.Reshuffle.V2.labeledTerms)

#### Backup model ready dataset
CTD.Reshuffle.ModelReady <- CTD.Reshuffle.V2.labeledTerms
write.csv(CTD.Reshuffle.ModelReady,"CTD.Reshuffle.ModelReady.csv")




### Adding the predictor column to the dataframe
CTD.Reshuffle.ModelReady$BotOrNot <- as.factor(CTD.Reshuffle.ModelReady$BotOrNot)
levels(CTD.Reshuffle.ModelReady$BotOrNot) <- make.names(levels(CTD.Reshuffle.ModelReady$BotOrNot))

##levels(CTD.Reshuffle.ModelReady$BotOrNot) <- rev(levels(CTD.Reshuffle.ModelReady$BotOrNot))

CTD.Reshuffle.ModelReady$BotOrNot <- relevel(CTD.Reshuffle.ModelReady$BotOrNot, ref ="X1")

table(CTD.Reshuffle.ModelReady$BotOrNot)



TDENS.train <- CTD.Reshuffle.ModelReady[1:255000,]
TDENS.test <- CTD.Reshuffle.ModelReady[255001:300000,]

##################################################################################################


#if you see an error when using parallel referencing "optimismBoot"
#its a bug in the current version of caret 6.0.77
#https://github.com/topepo/caret/issues/706
#install the dev release of caret for fix using:
devtools::install_github('topepo/caret/pkg/caret') 

### Using Cross Validation as our resampling technique and setting the number of folds for the same 
?createFolds
Folds<- createFolds(y=CTD.Reshuffle.ModelReady.y.train,k=10) 

#this time we are creating our folds in advanced to ensure all models use same folds during training
#for apples to apples comparisons
?trainControl
ctrl <- trainControl(method = "cv", number=10, summaryFunction=twoClassSummary,
                     indexOut =Folds, 
                     classProbs=T, savePredictions=T) #saving predictions from each resample fold


##confusionMatrix(pred.Model.rpart,CTD.Reshuffle.ModelReady.y.test) #calc accuracies with confusion matrix on test set
## Decision tree modelling
set.seed(620)
Model.rpart <- train(y=TDENS.train$BotOrNot, x=TDENS.train[,-65],
                     trControl = ctrl,
                     metric = "ROC", #using AUC to find best performing parameters
                     method = "rpart")

#Decision tree by normal method to plot the tree

Model.rpart1 <- rpart(TDENS.train$BotOrNot~.,data=TDENS.train,method="class")
rpart.plot(Model.rpart1, type=4, extra=2, clip.right.labs=FALSE, varlen=0, faclen=3)

pred.rpart1 <- predict(Model.rpart1,TDENS.test[,-65],type="class")
ConfusionMatrix(pred.rpart1,TDENS.test$BotOrNot)

prob.rpart1.test <- predict(Model.rpart1,TDENS.test[,-65], type="prob")

prob.rpart1.test[,1]
roc.rpart1.test <- roc(response = TDENS.test$BotOrNot, predictor = prob.rpart1.test[,1])
plot(roc.rpart1.test)


#ROC for rpart

Model.rpart
varImp(Model.rpart)
getTrainPerf(Model.rpart)
rpart.plot(Model.rpart$finalModel)
pred.Model.rpart<- predict(Model.rpart,TDENS.test[,-65])
confusionMatrix(pred.Model.rpart,TDENS.test$BotOrNot)

prob.rpart.test <- predict(Model.rpart,TDENS.test[,-65], type="prob")

roc.rpart.test <- roc(response = TDENS.test$BotOrNot, predictor = prob.rpart.test[[2]])
plot(roc.rpart.test)

##Logistic Regression (no parameters here, but will get cross validated perfomrance measures)
modelLookup("glm")
set.seed(620)


Model.glm<- train(y=TDENS.train$BotOrNot, x=TDENS.train[,-65],
                  trControl = ctrl,
                  metric = "ROC", #using AUC to find best performing parameters
                  method = "glm")



Model.glm
varImp(Model.glm)
getTrainPerf(Model.glm)

#rm(pred.Model.glm)
pred.Model.glm<- predict(Model.glm,TDENS.test[,-65])
table(pred.Model.glm)
### Model predictions 
pred.Model.glm
confusionMatrix(Model.glm$pred$pred,TDENS.train$BotOrNot)
confusionMatrix(pred.Model.glm,TDENS.test$BotOrNot)

#table(TDENS.train$BotOrNot)
#table(TDENS.test$BotOrNot)

###################### Plotting the ROC and optimizing the threshold ################
prob.Model.glm<- predict(Model.glm,TDENS.test[,-65], type = "prob")
test.glm.roc <- roc(response = TDENS.test$BotOrNot, predictor = prob.Model.glm[[1]])
plot(test.glm.roc)

new.threshold <- coords(test.glm.roc, x="best", best.method = "closest.topleft")
new.threshold

new.pred.glm <- factor(ifelse(prob.Model.glm[[1]] > 0.30, "X1", "X0"))
confusionMatrix(new.pred.glm, TDENS.test$BotOrNot)

###################
####################### Applying Neural Network
###########################################
require(neuralnet)
Model.nn <- train(y=TDENS.train$BotOrNot, x=TDENS.train[,-65],
                  trControl = ctrl,
                  #scale variables
                  method = "nnet")

plot(m.nn)
getTrainPerf(m.nn)



#################################################################################
### Perfomring caret ensemble on two models rpart and glm
### Using Cross Validation as our resampling technique and setting the number of folds for the same 
Folds<- createFolds(y=TDENS.train$BotOrNot,k=10) 


####  writing control for caret ensembling
library(caretEnsemble)
Ensemble_ctrl <- trainControl(
  method="cv",
  number=10,
  savePredictions=TRUE,
  classProbs=TRUE,
  indexOut=Folds,
  summaryFunction=twoClassSummary
)
?trainControl

colnames(TDENS.train)
TDENS.train.x <- TDENS.train[,-65]
###  defining list of models for caret ensembling
###   We use two completely opposite set of models in our caret ensembling to get the best results
?caretList
model_list1 <- caretList( x=TDENS.train.x, y=TDENS.train$BotOrNot,
                          trControl=Ensemble_ctrl,
                          methodList=c("glm", "rpart")
)

model_list1

####  
##xyplot(resamples(model_list))

##### 

greedy_ensemble <- caretEnsemble( model_list1, 
                                  metric="ROC", 
                                  trControl=Ensemble_ctrl
)

?caretEnsemble
summary(greedy_ensemble)
varImp(greedy_ensemble)

colnames(greedy_ensemble)

#####library("caTools")
## getting predictions of the caretEnsemble
model_preds <- lapply(model_list1, predict, newdata=TDENS.test, type="prob")
model_preds <- lapply(model_preds, function(x) x[,"X1"])
model_preds <- data.frame(model_preds)
ens_preds <- predict(greedy_ensemble, newdata=TDENS.test, type="prob")
model_preds$ensemble <- ens_preds
caTools::colAUC(model_preds, TDENS.test$BotOrNot)
ens_preds2 <- predict(greedy_ensemble, newdata=TDENS.test)
confusionMatrix(ens_preds2,TDENS.test$BotOrNot)

######################ROC for ensemble
test.ens.roc <- roc(response = TDENS.test$BotOrNot, predictor = ens_preds)
plot(test.ens.roc)
new.threshold <- coords(test.ens.roc, x="best", best.method = "closest.topleft")
new.threshold
new.pred.ens <- factor(ifelse(prob.Model.glm[[1]] > 0.243, "X1", "X0"))
confusionMatrix(new.pred.ens, TDENS.test$BotOrNot)
#### CaretStack was not used in our modeling but was used for testing purpose
#glm_ensemble <- caretStack(  model_list,
#                             method="glm",
#                             metric="ROC",
#                             trControl=Ensemble_ctrl
#                          )
#
#glm_ensemble

#model_preds2 <- model_preds
#model_preds2$ensemble <- predict(glm_ensemble, newdata=TDENS.test, type="prob")
#CF <- coef(glm_ensemble$ens_model$finalModel)[-1]
#colAUC(model_preds2, TDENS.test$BotOrNot)
##########################Making ROC plots############################################
rm(f.ensemble)
#lets compare all resampling approaches
f.models<- list("rpart"=Model.rpart, "glm"=Model.glm)

f.resamples = resamples(f.models)
#plot performance comparisons
bwplot(f.resamples, metric="ROC")
bwplot(f.resamples, metric="Sens") #predicting default dependant on threshold
bwplot(f.resamples, metric="Spec") 

f.ensemble <- list("Ensemble"= greedy_ensemble)
resamples(f.ensemble)


library(pROC)
Model.glm
##################################Random Forest Model#########################################

modelLookup("rf")
set.seed(192)

#manually specify randform parameters to search through

#rf.grid <- expand.grid(mtry=c(7,14,20,30, 40))
m.rf<- train(y=TDENS.train$BotOrNot, x=TDENS.train[,-65],
             trControl = ctrl, 
             ##  tune.grid=rf.grid,
             metric = "ROC", #using AUC to find best performing parameters
             method = "rf")
m.rf

getTrainPerf(m.rf)

varImp(m.rf)

pred.m.rf<- predict(m.rf,TDENS.test[,-65])
table(pred.m.rf)

pred.m.rf
confusionMatrix(pred.m.rf,TDENS.test$BotOrNot)


#################
#adjust threshold
library(pROC)
pred.m.rf1<- predict(m.rf,TDENS.test[,-65],type = "prob")
table(pred.m.rf1)

test.rf.roc <- roc(response = TDENS.test$BotOrNot, predictor = pred.m.rf1[[1]])

plot(test.rf.roc)

new.threshold <- coords(test.rf.roc, x="best", best.method = "closest.topleft")
?coords
new.threshold


new.pred.rf1 <- factor(ifelse(pred.m.rf1[[1]] > 0.15, "X1", "X0"))

confusionMatrix(new.pred.rf1, TDENS.test$BotOrNot)

###########################################################
### SVM was modelled on a small subset but we again have made sure that distribution was constant for all the models 
TDENS.svm.train <- CTD.Reshuffle.ModelReady[1:4500,]
TDENS.svm.test <- CTD.Reshuffle.ModelReady[4501:5000,]


library(kernlab)
library(MLmetrics)

ctrl_svm <- trainControl(method="cv",number=5,
                         classProbs=TRUE,
                         #function used to measure performance
                         summaryFunction = multiClassSummary, 
                         allowParallel = TRUE) #default looks for parallel backend

##svm with radial kernel
modelLookup("svmRadial")

expand.grid(fL=c(TRUE, FALSE), usekernel=c(TRUE,FALSE),
            adjust=c(TRUE,FALSE))

m.svmlinear <- train(y=TDENS.svm.train$BotOrNot, x=TDENS.svm.train[,-65],
                     trControl = ctrl_svm, metric = "Accuracy", #Accuracy over all classes
                     #preProc = c("scale"), #scale variables
                     method = "svmLinear")

m.svmlinear
getTrainPerf(m.svmlinear)

varImp(m.svmlinear)

pred.m.svmlinear<- predict(m.svmlinear,TDENS.svm.test[,-65])
table(pred.m.svmlinear)

pred.m.svmlinear
confusionMatrix(pred.m.svmlinear,TDENS.svm.test$BotOrNot)


#################
#adjust threshold
library(pROC)
pred.m.svmlinear1<- predict(m.svmlinear,TDENS.svm.test[,-65],type = "prob")
table(pred.m.svmlinear1)
test.svmlinear.roc <- roc(response = TDENS.svm.test$BotOrNot, predictor = pred.m.svmlinear1[[1]])

plot(test.svmlinear.roc)

new.threshold.svm <- coords(test.svmlinear.roc, x="best", best.method = "closest.topleft")
?coords
new.threshold.svm


new.pred.m.svmlinear1<- factor(ifelse(pred.m.svmlinear1[[1]] > 0.26, "X1", "X0"))

confusionMatrix(new.pred.m.svmlinear1, TDENS.svm.test$BotOrNot)





##########################Making ROC plots############################################
rm(f.ensemble)
#lets compare all resampling approaches
f.models<- list("rpart"=Model.rpart, "glm"=Model.glm)

f.resamples = resamples(f.models)
#plot performance comparisons
bwplot(f.resamples, metric="ROC")
bwplot(f.resamples, metric="Sens") #predicting default dependant on threshold
bwplot(f.resamples, metric="Spec") 

f.ensemble <- list("Ensemble"= greedy_ensemble)
resamples(f.ensemble)

########
#############################################Topic Modelling#################################

library("plyr")
library("stringr")
library("tm")
library("SnowballC")
library("lda")
library("LDAvis")

t.subset <- read.csv("Subset.csv")

head(t.subset$text,10 )
Tweets_corpus <- Corpus(VectorSource(t.subset$text))


Tweets_corpus <- tm_map(Tweets_corpus, tolower)



# remove punctuation
Tweets_corpus <- tm_map(Tweets_corpus, removePunctuation)
# remove numbers
Tweets_corpus <- tm_map(Tweets_corpus, removeNumbers)

# remove URLs
Tweets_corpus <- tm_map(Tweets_corpus, function(x) gsub("http[[:alnum:]]*","", x))
# remove NonASCII characters
Tweets_corpus <- tm_map(Tweets_corpus, function(x) iconv(x, "latin1", "ASCII", sub=""))

# remove stopwords
Tweets_corpus <-tm_map(Tweets_corpus, removeWords,stopwords("SMART"))

# remove specific words
Tweets_corpus <- tm_map(Tweets_corpus, removeWords,c("london", "im","ive", "dont", "didnt"))

Tweets_corpus <- tm_map(Tweets_corpus, stripWhitespace)

Tweets_corpus <- tm_map(Tweets_corpus, PlainTextDocument)
Tweets_corpus <- tm_map(Tweets_corpus, stemDocument)

# unlist the text corpus
Tweet_Clean<-as.data.frame(unlist(sapply(Tweets_corpus[[1]]$content,'[')), stringsAsFactors=F)
# remove extra whitespace in text
Tweet_Clean <- lapply(Tweet_Clean[,1], function(x) gsub("^ ", "", x)) #multiple spaces
Tweet_Clean <- lapply(Tweet_Clean, function(x) gsub("^[[:space:]]+", "", x)) #space at the begining
Tweet_Clean <- lapply(Tweet_Clean, function(x) gsub("[[:space:]]+$", "", x)) #space at the end


# bind clean text with Twitter data
traffic$Tweet_Clean<-Tweet_Clean
# check the first 10 Tweets
traffic[1:10,]

# tokenize on space and output as a list:
doc.list <- strsplit(unlist(Tweet_Clean), "[[:space:]]+")


# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)
# remove terms that are stop words or occur fewer than 3 times:
term.table <- term.table[term.table>3]

vocab <- names(term.table)

# put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

# Compute some statistics related to the data set:
D <- length(documents) # number of documents
W <- length(vocab) # number of terms in the vocab
doc.length <- sapply(documents, function(x) sum(x[2, ])) # number of tokens per document
N <- sum(doc.length) # total number of tokens in the data
term.frequency <- as.integer(term.table) # frequencies of terms in the corpus


### fit LDA model
# parameters
K <- 10
G <- 1000
alpha <- 0.1
eta <- 0.1
t1 <- print(Sys.time())
lda_fit <- lda.collapsed.gibbs.sampler (documents = documents, K = K, vocab = vocab, num.iterations =
                                          G, alpha = alpha, eta = eta)
t2 <- print(Sys.time())
t2-t1

top_words<-top.topic.words(lda_fit$topics,20,by.score=TRUE)

theta <- t(apply(lda_fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(lda_fit$topics) + eta, 2, function(x) x/sum(x)))


Tweet_Topics <- list(phi = phi, theta = theta, doc.length = doc.length, vocab = vocab,
                     term.frequency = term.frequency)


Tweet_Topics_json <- with(Tweet_Topics,createJSON(phi, theta, doc.length, vocab, term.frequency))

library(jsonlite)

write_json(Tweet_Topics_json, "LDA.json" )
serVis(Tweet_Topics_json)
?serVis()

doc_topic <- apply(lda_fit$document_sums, 2, function(x) which(x == max(x))[1])
Twitter$topic<-doc_topic




TwitterDf <- data.frame(lapply(Twitter, as.character), stringsAsFactors=FALSE)
write.csv(TwitterDf,"TwitterWithTopic.csv")


##################################################################################
#Making color vectors
colors<- c("green", "red", "orange", "yellow")
#, "purple", "violet", "blue", "cyan", "coral4", "brown"
colors[1]
linetype<- c(1:100) 
plotchar<- seq(18,18+100,1)
plot(test.ens.roc, type="l",legacy.axes=T,col=colors[1],lty=linetype[1], pch=seq(1))
plot(test.glm.roc, type="l", add=T,col=colors[2], lty=linetype[2], pch=seq(2))
plot(roc.rpart.test, type="l", add=T,col=colors[3], lty=linetype[3], pch=seq(3))
#plot(test.gam.roc, type="l", add=T,col=colors[4], lty=linetype[4], pch=seq(4))
#plot(test.nb.roc, type="l", add=T,col=colors[5], lty=linetype[5], pch=seq(5))
#plot(test.log.roc, type="l", add=T,col=colors[6], lty=linetype[6], pch=seq(6))
#plot(test.lda.roc, type="l", add=T,col=colors[7], lty=linetype[7], pch=seq(7))
#plot(test.qda.roc, type="l", add=T,col=colors[8], lty=linetype[8], pch=seq(8))
#plot(test.decisiontree.roc, type="l", add=T,col=colors[9], lty=linetype[9], pch=seq(9))

legend(x=0, y=0.5,
       legend=c("ensemble", "glm","rpart"),
       col=colors, lty=linetype ,horiz = FALSE, lwd=2)
#legend(x=0.5, y=1.05,
#       legend=c("ensemble", "glm","rpart"),
#       col=colors, lty=linetype ,horiz = FALSE, lwd=2)


##########################Making ROC plots############################################
#Making color vectors
colors<- c("green", "red", "orange", "blue", "coral4")
#, "purple", "violet", "blue", "cyan", "coral4", "brown"
#colors[1]
#pch = c(16, 17, 18)
#pch[1]
linetype<- c(1:100) 
plotchar<- seq(18,18+100,1)
plot(test.ens.roc, type="l",legacy.axes=T,col=colors[1],lty=linetype[1], pch=seq(1:2) )
plot(test.glm.roc, type="l", add=T,col=colors[2], lty=linetype[2], pch=seq(30))
plot(roc.rpart.test, type="l", add=T,col=colors[3], lty=linetype[3], pch=seq(25))
plot(test.rf.roc, type="l", add=T,col=colors[4], lty=linetype[4], pch=seq(10))
plot(test.svmlinear.roc, type="l", add=T,col=colors[5], lty=linetype[5], pch=seq(3))

#adding the labbles in the plot
legend(x=0.2, y=0.7,
       legend=c("ensemble", "glm","rpart", "rf", "svm"),
       col=colors, lty=linetype ,horiz = FALSE, lwd=2)


f.models<- list("glm"=Model.glm, "rpart"=Model.rpart, "rf"=m.rf)
f.resamples = resamples(f.models)
#plot performance comparisons
bwplot(f.resamples, metric="ROC")
bwplot(f.resamples, metric="Sens") #predicting default dependant on threshold
bwplot(f.resamples, metric="Spec") 



