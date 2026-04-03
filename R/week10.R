# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)
library(haven)
library(jtools)

# Data Import and Cleaning
# zap_missing() converts tagged missing values into standard R NA value. Used read_sav() from haven to read in GSS2016.sav
gss_tbl <- read_sav("../data/GSS2016.sav", user_na = TRUE) %>% 
  zap_missing() %>% 
  # this mutates the mosthrs variable to be numeric, which is necessary later on when training the models
  mutate(mosthrs = as.numeric(mosthrs)) %>% 
  # using filter() here removes anyone who has a missing value for work hours
  filter(!is.na(mosthrs)) %>% 
  # using select() here removes the other variables referencing hours per week
  select(-hrs1, -hrs2) %>% 
  # using select_if() to only select columns with less than 75% missingness. The mean shows the proportion of NA values in the given column
  select_if(~ mean(is.na(.)) < 0.75)
  
# Visualization
# this creates a histogram of work hours
ggplot(gss_tbl, aes(x = mosthrs)) +
  geom_histogram(binwidth = 10, fill = "#A80000") +
  xlab("Work Hours") +
  ylab("Count") +
  theme_apa()

# Analysis
# setting the seed allows anyone rerunning the code to reproduce the same results
set.seed(42)
# this creates the 75/25 split, with 75% going to training by designating p = 0.75
holdout_indices <- createDataPartition(gss_tbl$mosthrs, p = 0.25, list = F)
# this creates the testing data from the holdout indices
gss_holdout <- gss_tbl[holdout_indices, ]
# this creates the training data from the holdout indices
gss_training <- gss_tbl[-holdout_indices, ]

# we need to keep fold composition the same across models
# using createFolds() will now provide the same 10-fold splits for all models
fold_indices <- createFolds(gss_training$mosthrs, k = 10)
gss_training <- gss_training %>% 
  mutate(mosthrs = as.numeric(mosthrs))

# this is for the OLS regression model
ols_model <- train(
  mosthrs ~ ., # this tells the train function to model mosthrs as a function of all other variables in gss_training, which is defined in the following line
  data = gss_training, # this tells the model to use gss_training as the data set
  method = "lm", # this instructs the function fit an OLS regression model
  na.action = na.pass, # this tells the train function to keep missing values
  preProcess = "medianImpute", # this specifies that we use median imputation to impute any remaining missing values
  trControl = trainControl( # this defines the control parameters
    method = "cv", # this defines the use of K-fold cross-validation
    number = 10, # this defines the number of folds, so 10-fold cross-validation
    verboseIter = TRUE, # this tracks the progress of the training
    indexOut = fold_indices # this uses the same folds for all models
  )
)

# this is for the elastic net model
elastic_model <- train(
  mosthrs ~ ., # this tells the train function to model mosthrs as a function of all other variables in gss_training, which is defined in the following line
  data = gss_training, # this tells the model to use gss_training as the data set
  method = "glmnet", # this specifies elastic net regression
  na.action = na.pass, # this tells the train function to keep missing values
  preProcess = c("medianImpute", "center", "scale"), # this specifies that we use median imputation to impute any remaining missing values as well as centers and standardizes
  trControl = trainControl( # this defines the control parameters
    method = "cv", # this defines the use of K-fold cross-validation
    number = 10, # this defines the number of folds, so 10-fold cross-validation
    verboseIter = TRUE, # this tracks the progress of the training
    indexOut = fold_indices # this uses the same folds for all models
  ),
  tuneGrid = expand.grid( # this hyperparameter creates a data frame that contains every possible combination of the values
    alpha = c(0, 1), # this defines the balance between LASSO and ridge,  0 is LASSO, 1 is ridge
    lambda = seq(0.0001, 0.1, length = 10) # this is the cost function which controls the penalty strength on the model's coefficients
  )
)

# this is for the random forest model
forest_model <- train(
  mosthrs ~ ., # this tells the train function to model mosthrs as a function of all other variables in gss_training, which is defined in the following line
  data = gss_training, # this tells the model to use gss_training as the data set
  method = "ranger", # this specifies the random forest model
  na.action = na.pass, # this tells the train function to keep missing values
  preProcess = "medianImpute", # this specifies that we use median imputation to impute any remaining missing values
  tuneLength = 10, # this hyperparameter tells the function try 10 mtry
  trControl = trainControl( # this defines the control parameters
    method = "cv", # this defines the use of K-fold cross-validation
    number = 10, # this defines the number of folds, so 10-fold cross-validation
    verboseIter = TRUE, # this tracks the progress of the training
    indexOut = fold_indices # this uses the same folds for all models
  )
)

# this is for the eXtreme Gradient Boosting model
