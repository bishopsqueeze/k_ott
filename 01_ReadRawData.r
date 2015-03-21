##------------------------------------------------------------------
## Process Raw Data Files
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------
#library(data.table)

##------------------------------------------------------------------
## Clear the workspace
##------------------------------------------------------------------
rm(list=ls())

##------------------------------------------------------------------
## Set the working directory
##------------------------------------------------------------------
setwd("/Users/alexstephens/Development/kaggle/otto/data/raw")

##------------------------------------------------------------------
## Read in the various files
##------------------------------------------------------------------
train.raw	<- read.csv("train.csv", header=TRUE)               ## 61878 x 95
test.raw	<- read.csv("test.csv", header=TRUE)                ## 144368 x 94
sub.raw     <- read.csv("sampleSubmission.csv", header=TRUE)    ## 144368 x 10

##------------------------------------------------------------------
## Write the raw data to .Rdata file(s)
##------------------------------------------------------------------
save(train.raw, file="/Users/alexstephens/Development/kaggle/otto/data/01_trainRaw.Rdata")
save(test.raw,  file="/Users/alexstephens/Development/kaggle/otto/data/01_testRaw.Rdata")
save(sub.raw,   file="/Users/alexstephens/Development/kaggle/otto/data/01_subRaw.Rdata")