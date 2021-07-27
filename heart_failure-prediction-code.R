# **********************************************************************************************
# # Installing required Package 
# **********************************************************************************************

#### Download the required Package
if(!require(tidyverse))             install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret))                 install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(PerformanceAnalytics))  install.packages("PerformanceAnalytics", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer))          install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")
if(!require(randomForest))          install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(rpart))                 install.packages("rpart", repos = "http://cran.us.r-project.org")


#### Use the required library
library(tidyverse)
library(caret)
library("PerformanceAnalytics")
library(corrplot)
library(RColorBrewer)
library(randomForest)
library(rpart)



# **********************************************************************************************
#### import and read the database
# **********************************************************************************************
# the github repo with the data set "telecom_users" is available here: https://github.com/MSammoudi
# the file "heart_failure_clinical_records_dataset.csv" provided in the github repo must be included in the working (project) directory for the code below to run
# you can download the data base file "heart_failure_clinical_records_dataset.csv" from kaggle website from the following link: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

# read in the Heart Failure dataset and save the data in heart_failure_data variable
heart_failure_data <- read.csv("heart_failure_clinical_records_dataset.csv", header = TRUE)

# I have another object for data called heartfailure.dat, I will use it to split the data later
# and to use it with machine learning models.
#I will use read.csv function to read th file, so I can use the argument StringsAfFacotrs
heartfailure.dat = read.csv("heart_failure_clinical_records_dataset.csv",stringsAsFactors = FALSE)
#Define the factors in our dataset
factors = c("anaemia","diabetes","high_blood_pressure","sex","smoking","DEATH_EVENT")
#Change the variables data type in facrots vector to be factor datatype
heartfailure.dat[factors] = lapply(heartfailure.dat[factors],factor)

#show the first 6 rows in our dataset
head (heartfailure.dat)

#This step is necessary in most dataset, so that no variable can change the output more than others due to its value.
dat.scaled = heartfailure.dat %>% mutate_if(is.numeric, scale)

####**** 1.2 Data Overview:
dim(heart_failure_data) ## return dataset dimention 299 * 13

### show the structure of  the Data
str(heart_failure_data)

# **********************************************************************************************
#### 2. Data Analysis :(data cleaning, Data Overview,  data exploration and visualization)
# **********************************************************************************************

# Checking if there are any missing values in the dataset
sum(is.na(heart_failure_data))

# Having a look at the correlation matrix
plot(cor(heart_failure_data))

#Find and plot the correlation matrix between the variables
df_cor <-cor(heart_failure_data)
corrplot(df_cor, type="full", order="hclust",
         col=brewer.pal(n=8, name="RdYlBu"))

######################################################################################
# Visualization of variables and how they affect the death event.
###################################################################################

#Age of patients distribution
heart_failure_data %>%
  ggplot(aes(age, fill=as.factor(DEATH_EVENT)))+
  geom_histogram(binwidth = 5, position = "identity",alpha = 0.5,color = "white")+
  xlab("Age") +
  ylab("Number of patients") +
  theme_classic() +
  labs(caption = "Age Distribution with Death Event")
#The average age of the patients seems to be between 55 to 75 years. With the maximum age
#being 95 and the minimum being 40 years.

##Blood pressure distribution
pie(table(heart_failure_data$high_blood_pressure),
    labels = c("Normal Blood Pressure", "High Blood Pressure"),
    main = "Blood Pressure Distribution",
    col = c("yellow", "red"))
## Blood pressure effect on death_event.
ggplot(heart_failure_data,aes(x = high_blood_pressure,fill = as.factor(DEATH_EVENT)))+
  geom_bar() + theme_classic()

## Diabetes distribution between patients
heart_failure_data %>%
  group_by(diabetes) %>%
  ggplot(aes(diabetes))+
  geom_bar(aes(fill = ..count..))
## Diabetes effect on death_event.
ggplot(heart_failure_data,aes(x = diabetes,fill = as.factor(DEATH_EVENT)))+
  geom_bar() + theme_classic()

## Anaemia distribution between patients
heart_failure_data %>%
  group_by(anaemia) %>%
  ggplot(aes(anaemia, fill = as.factor(DEATH_EVENT)))+
  geom_bar()

## Sex distribution between patients
ggplot(heart_failure_data,aes(x = sex,fill = as.factor(DEATH_EVENT)))+
  geom_bar() + theme_classic()

##Smoking distribution among patients
pie(table(heart_failure_data$smoking),
    labels = c("Non-Smoker", "Smoker"),
    main = "Smoking Distribution",
    col = c("green","red"))
#Smoking effect on death event
ggplot(heart_failure_data,aes(x = smoking,fill = as.factor(DEATH_EVENT)))+
  geom_bar() + theme_classic()

#Ejection Fraction¶
ggplot(heart_failure_data,aes(x = ejection_fraction,fill = as.factor(DEATH_EVENT)))+
  geom_bar() + theme_classic()

# PlateLets
ggplot(heart_failure_data,aes(x = platelets,fill = as.factor(DEATH_EVENT)))+
  geom_density(alpha = 0.2) + theme_classic() 

#serum_sodium
ggplot(heart_failure_data,aes(x = serum_sodium,fill = as.factor(DEATH_EVENT)))+
  geom_bar() + theme_classic()

################################################################################################
# **********************************************************************************************
###   Modeling and Evaluation approach :(Logistic Regression,QDA,LDA,Decision Tree and Random Forest)
# **********************************************************************************************

# **********************************************************************************************


# Create train and test sets (80%) from telecom_data train is used to train various models and
# test set (20%) is used to assess their performances

set.seed(1989, sample.kind="Rounding")

#split the dataset to trainig set and test set (80% training and 20$ testing)
trainIndex = sample(1:length(heartfailure.dat$DEATH_EVENT),0.8*length(heartfailure.dat$DEATH_EVENT))
#Training set with 80% of observation and 13 variables
train = heartfailure.dat[trainIndex,-c(12,14:16)]
#Testing set with 20% of observations and 12 variables (all variables except the dependent varibble)
test.x = heartfailure.dat[-trainIndex,-c(12,13:16)]
#Factor vector for the dependent variable DEATH.EVENT in the testing set.
test.y = heartfailure.dat[-trainIndex,13]

# **********************************************************************************************
#####  The Logistic Regression Model
# **********************************************************************************************

#Logistic Regression (glm) model will train data in train_data and test it again test_data.
model_glm <- glm(DEATH_EVENT ~ ., data = train, family = "binomial")

### Print the result of Logistic regression model
model_glm

### Define predictions for the glm model
preds_glm <- predict(model_glm, newdata = test.x, type = "response")

y_hat_glm <- ifelse(preds_glm > 0.5, 1, 0)
confusionMatrix(as.factor(y_hat_glm), test.y)$overall[["Accuracy"]]
confusionMatrix(as.factor(y_hat_glm), test.y)



results <- data_frame(method = "Model 1:Logistic Regression",
                      Model_Accuracy = 0.7666667,
                      Model_Sensitivity = 0.9091,
                      Model_Specificity = 0.3750)
results %>% knitr::kable()

#####################################################################################333##

# **********************************************************************************************
#####  The Random Forest Model
# **********************************************************************************************

###################################################################3
control_rf <- trainControl(method = "cv", number = 10,returnResamp = "all", 
                           classProbs = TRUE, summaryFunction = twoClassSummary)
train_rf_cv <- train(ifelse(DEATH_EVENT==1,"YES","NO")~.,
                     method = "rf",
                     tuneGrid = data.frame(mtry = 3:11),
                     data = train,
                     trControl = control_rf)
opt_mtry_rf <- train_rf_cv$bestTune %>% as.numeric()
model_rf <- randomForest(DEATH_EVENT ~., data = train, mtry = opt_mtry_rf, importance = TRUE, ntree=300, do.trace=FALSE)
varImpPlot(model_rf)
preds_rf <- predict(model_rf, test.x) # Calclating the prediction metrics on test data
cm_rf <- confusionMatrix(as.factor(preds_rf), test.y)$overall[1]
cm_rf
######

# **********************************************************************************************
#####  The classification (Disision) tree Model
# **********************************************************************************************
# fit a classification tree and plot it
train_rpart <- train(DEATH_EVENT ~ .,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     data = train)
plot(train_rpart)

# compute accuracy
confusionMatrix(predict(train_rpart, test.x), test.y)$overall["Accuracy"]

#############################################################################3

# **********************************************************************************************
#####  The QDA Model
# **********************************************************************************************
# Fit QDA model
train_qda <- train(DEATH_EVENT ~ .,
                   method = "qda",
                   data = train)
# Obtain predictors and accuracy
y_hat <- predict(train_qda, test.x)
confusionMatrix(data = y_hat, reference = test.y)$overall["Accuracy"]
#############################################################################
# **********************************************************************************************
#####  The LDA Model
# **********************************************************************************************
# Fit LDA model
train_lda <- train(DEATH_EVENT ~ ., method = "lda", data = train)
predict_lda <- predict(train_lda, test.x)
confusionMatrix(data = predict_lda, reference = test.y)$overall["Accuracy"]
###########################################################################
