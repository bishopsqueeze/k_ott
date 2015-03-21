##------------------------------------------------------------------
## Process Raw Data Files
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------
#require(xgboost)
require(methods)
require(randomForest)

##------------------------------------------------------------------
## Clear the workspace
##------------------------------------------------------------------
rm(list=ls())

##------------------------------------------------------------------
## Set the working directory
##------------------------------------------------------------------
setwd("/Users/alexstephens/Development/kaggle/otto/data")

##------------------------------------------------------------------
## Read in the various files
##------------------------------------------------------------------
load("/Users/alexstephens/Development/kaggle/otto/data/02_ProcessData_RawData.Rdata")
#load("/Users/alexstephens/Development/kaggle/otto/data/02_ProcessData_Base2Data.Rdata")

##------------------------------------------------------------------
## Modify the target
##------------------------------------------------------------------
y = train.target
y = gsub('L','',y)
y = as.integer(y)-1         ## xgboost take features in [0,numOfClass)

##------------------------------------------------------------------
## Combine train/test
##------------------------------------------------------------------
x = rbind(train.feat, test.feat)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train.feat)+1):nrow(x)

##------------------------------------------------------------------
## Set parameters
##------------------------------------------------------------------
param <- list(  "objective" = "multi:softprob",
                "eval_metric" = "mlogloss",
                "num_class" = 9,
                "nthread" = 8,
                "eta" = 0.1,
                "max_depth" = 9,
                "min_child_weight" = 1,
                "subsample" = 0.9,
                "colsample_bytree" = 1)

##------------------------------------------------------------------
## Run Cross Valication
##------------------------------------------------------------------
cv.nround = 500
system.time({
    bst.cv = xgb.cv(param=param, data = x[trind,], label = y, nfold = 5, nrounds=cv.nround, verbose=2)
})

##------------------------------------------------------------------
## Train the model
##------------------------------------------------------------------
nround = 500
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

##------------------------------------------------------------------
## Make prediction
##------------------------------------------------------------------
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

##------------------------------------------------------------------
## Output submission
##------------------------------------------------------------------
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))

##------------------------------------------------------------------
#param <- list(  "objective" = "multi:softprob",
#                "eval_metric" = "mlogloss",
#                "num_class" = 9,
#                "nthread" = 4,
#                "eta" = 0.01,
#                "max_depth" = 9,
#                "min_child_weight" = 1,
#                "subsample" = 0.9,
#                "colsample_bytree" = 1)
##------------------------------------------------------------------
## nround = 500
##------------------------------------------------------------------
#write.csv(pred,file='01_submission.csv', quote=FALSE,row.names=FALSE)


##------------------------------------------------------------------
#param <- list(  "objective" = "multi:softprob",
#                "eval_metric" = "mlogloss",
#                "num_class" = 9,
#                "nthread" = 4,
#                "eta" = 0.01,
#                "max_depth" = 9,
#                "min_child_weight" = 1,
#                "subsample" = 0.9,
#                "colsample_bytree" = 1)
##------------------------------------------------------------------
## nround = 3000
##------------------------------------------------------------------
##write.csv(pred,file='02_submission.csv', quote=FALSE,row.names=FALSE)


