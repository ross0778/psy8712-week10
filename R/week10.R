# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)
library(haven)
library(jtools)
library(xgboost)
library(devtools)

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
  tuneLength = 10, # this hyperparameter tells the function try 10 mtry values
  trControl = trainControl( # this defines the control parameters
    method = "cv", # this defines the use of K-fold cross-validation
    number = 10, # this defines the number of folds, so 10-fold cross-validation
    verboseIter = TRUE, # this tracks the progress of the training
    indexOut = fold_indices # this uses the same folds for all models
  )
)

# this is for the eXtreme Gradient Boosting model
extreme_model <- train(
  mosthrs ~ ., # this tells the train function to model mosthrs as a function of all other variables in gss_training, which is defined in the following line
  data = gss_training, # this tells the model to use gss_training as the data set
  method = "xgbLinear", # this specifies the eXtreme Gradient Boosting model
  na.action = na.pass, # this tells the train function to keep missing values
  preProcess = "medianImpute", # this specifies that we use median imputation to impute any remaining missing values
  tuneLength = 10, # this hyperparameter tells the function try 10 mtry values
  trControl = trainControl( # this defines the control parameters
    method = "cv", # this defines the use of K-fold cross-validation
    number = 10, # this defines the number of folds, so 10-fold cross-validation
    verboseIter = TRUE, # this tracks the progress of the training
    indexOut = fold_indices # this uses the same folds for all models
  )
)

# Publication
# this function calculates the correlation between the predictions from the trained models on the holdout data and finds the correlation between these predictions and the actual values. I then square to find R^2
holdout_rsq <- function(model, holdout) { # this creates a function that takes the trained model and the holdout data. Made a function so this process wouldn't have to be repeated for the four models
  predicted <- predict(model, holdout, na.action = na.pass) # this gets the predicted mosthrs values for each row in the holdout data using the trained model
  cor(predicted, holdout$mosthrs)^2 # this finds the correlation between the predicted values and the actual mosthrs values and then finds the R^2 by squaring it
}

table1_tbl <- tibble( # this creates the tibble
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"), # this specifies the type of algorithm used for each
  cv_rsq = c( # wrapping max() on each model's results provides the highest CV R^2
    max(ols_model$results$Rsquared, na.rm = TRUE),
    max(elastic_model$results$Rsquared, na.rm = TRUE),
    max(forest_model$results$Rsquared, na.rm = TRUE),
    max(extreme_model$results$Rsquared, na.rm = TRUE)
  ),
  ho_rsq = c( # this applies the function created at the top of this section to each model using the gss_holdout data, giving the final holdout CV R^2 for each algo
    holdout_rsq(ols_model, gss_holdout),
    holdout_rsq(elastic_model, gss_holdout),
    holdout_rsq(forest_model, gss_holdout),
    holdout_rsq(extreme_model, gss_holdout)
  )
) %>% 
  # this rounds all values to 2 decimal places using formatC() and gets rid of the leading zeros using str_remove() and regex
  mutate(across(c(cv_rsq, ho_rsq), ~ formatC(round(.x, 2), format = "f", digits = 2) %>% 
                  str_remove("^0")))

table1_tbl
write_csv(table1_tbl, "../out/table1.csv")

# 1. How did your results change between models? Why do you think this happened, specifically?
# Results significantly improved when moving from OLS to the following elastic net, random forest, and extreme gradient boosting models. 
# OLS had the lowest CV R^2 of .13, which does make sense, as OLS only accounts for linear relationships.
# Moving to the elastic net model improved the CV R^2 drastically to 0.88. I think this is probably because elastic net reduces overfitting.
# Random forest continued to improve up to .92, which I think is because it is able to capture both non-linear relationships and interactions.
# Finally, extreme gradient boosting had the highest CV R^2 at .99. This seems extremely high and I believe this may be due to overfitting. This could also be because I used xgbLinear instead of xgbTree, which fits a linear model instead of building sequential decision trees.

# 2. How did your results change between k-fold CV and holdout CV? Why do you think this happened, specifically?
# Each model showed a clear decrease from CV R^2 to holdout R^2. Some of these changes were especially significant, with the drop in OLS regression showing that the OLS model failed to generalize beyond the trianing data with the holdout R^2 at .00.
# I think this happened because the holdout data is what is not used during training, meaning this may be a more accurate estimate.

# 3. Among the four models, which would you choose for a real-life prediction problem, and why? Are there tradeoffs?
# I would use the random forest model for a real-life prediction problem.
# This model had a CV R^2 of .92 and a holdout R^2 of .51.
# I would choose the random forest model instead of the extreme gradient boosting model, despite the latter having a slightly higher CV R^2 and holdout R^2, because the nearly perfect CV R^2 seems to imply overfitting may be an issue.
# There are tradeoffs however. Random forest models used bagged modeling and reduce variance without needing the hyperparameters that the extreme gradient boosting requires.
# On the other hand, with models like elastic net, you're able to read individual predictor coefficients, which you can't do with random forest models.
