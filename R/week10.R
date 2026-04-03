# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)
library(haven)

# Data Import and Cleaning
# zap_missing() converts tagged missing values into standard R NA value. Used read_sav() from haven to read in GSS2016.sav
gss_tbl <- read_sav("../data/GSS2016.sav", user_na = TRUE) %>% 
  zap_missing() %>% 
  # this converts haven_labelled vectors to factors
  mutate(across(where(is.labelled), as_factor)) %>% 
  # using filter() here removes anyone who has a missing value for work hours
  filter(!is.na(mosthrs)) %>% 
  # using select() here removes the other variables referencing hours per week
  select(-hrs1, -hrs2) %>% 
  # using select_if() to only select columns with less than 75% missingness. The mean shows the proportion of NA values in the given column
  select_if(~ mean(is.na(.)) < 0.75)
  

