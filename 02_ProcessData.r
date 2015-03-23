##------------------------------------------------------------------
## Process Raw Data Files
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------
library(caret)
library(car)

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
load("/Users/alexstephens/Development/kaggle/otto/data/01_trainRaw.Rdata")  ## raw train data
load("/Users/alexstephens/Development/kaggle/otto/data/01_testRaw.Rdata")   ## raw test data

##------------------------------------------------------------------
## Split the data into id/target/features
##------------------------------------------------------------------

## train
train.id        <- train.raw$id
train.target    <- as.factor(gsub("Class_", "L", train.raw$target))
train.feat      <- train.raw[,2:(ncol(train.raw)-1)]
train.feat.b2   <- log2(1 + train.feat)

## test
test.id         <- test.raw$id
test.feat       <- test.raw[,2:ncol(test.raw)]
test.feat.b2    <- log2(1 + test.feat)


##------------------------------------------------------------------
## Combine the feature set
##------------------------------------------------------------------
comb.feat       <- as.matrix(rbind(train.feat, test.feat))
comb.feat.b2    <- as.matrix(rbind(train.feat.b2, test.feat.b2))
comb.id         <- as.vector(c(train.id, test.id))


##------------------------------------------------------------------
## Examine train dataset
##------------------------------------------------------------------

##------------------------------------------------------------------
## near-zero variables
##------------------------------------------------------------------
## c(5, 6, 23, 31, 45, 47, 51, 61, 77, 81, 82, 84, 93)
train.nzv.save  <- nearZeroVar(train.feat, saveMetrics = TRUE)
train.nzv       <- nearZeroVar(train.feat)

##------------------------------------------------------------------
## Constuct additional feautres (feature engineering)
##------------------------------------------------------------------

##------------------------------------------------------------------
## compute class distances on raw values (then combine & normalize)
##------------------------------------------------------------------
train.centroids     <- classDist(train.feat, train.target)
train.newDist       <- predict(train.centroids, train.feat)
test.newDist        <- predict(train.centroids, test.feat)

comb.newDist            <- rbind(train.newDist, test.newDist)
comb.newDist.preProc    <- preProcess(comb.newDist, method=c("center", "scale"))
comb.newDist.unif.feat  <- predict(comb.newDist.preProc, newdata=comb.newDist)

##------------------------------------------------------------------
## compute class distances on log2 values (then combine & normalize)
##------------------------------------------------------------------
train.centroids.b2  <- classDist(train.feat.b2, train.target)
train.newDist.b2    <- predict(train.centroids.b2, train.feat.b2)
test.newDist.b2     <- predict(train.centroids.b2, test.feat.b2)

comb.newDist.b2             <- rbind(train.newDist.b2, test.newDist.b2)
comb.newDist.preProc.b2     <- preProcess(comb.newDist.b2, method=c("center", "scale"))
comb.newDist.unif.feat.b2   <- predict(comb.newDist.preProc.b2, newdata=comb.newDist.b2)

##------------------------------------------------------------------
## compute additional feautres
##------------------------------------------------------------------

## sum of all feature counts
train.cntSum        <- apply(train.feat, 1, sum)
test.cntSum         <- apply(test.feat, 1, sum)



##------------------------------------------------------------------
## Normalize feature set
##------------------------------------------------------------------

##------------------------------------------------------------------
## Simple [0,1] normalization
##------------------------------------------------------------------
comb.feat.unif              <- comb.feat %*% diag(1/apply(comb.feat, 2, max))
colnames(comb.feat.unif)    <- colnames(comb.feat)

comb.feat.b2.unif           <- comb.feat.b2 %*% diag(1/apply(comb.feat.b2, 2, max))
colnames(comb.feat.b2.unif) <- colnames(comb.feat.b2)


##------------------------------------------------------------------
## Split transformed data into train/test pairs
##------------------------------------------------------------------
train.feat.unif          <- comb.feat.unif[train.id,]
test.feat.unif           <- comb.feat.unif[-train.id,]

train.feat.unif.b2       <- comb.feat.b2.unif[train.id,]
test.feat.unif.b2        <- comb.feat.b2.unif[-train.id,]

train.distFeat.unif      <- comb.newDist.unif.feat[train.id,]
test.distFeat.unif       <- comb.newDist.unif.feat[-train.id,]

train.distFeat.unif.b2   <- comb.newDist.unif.feat.b2[train.id,]
test.distFeat.unif.b2    <- comb.newDist.unif.feat.b2[-train.id,]


##------------------------------------------------------------------
## Save results
##------------------------------------------------------------------

## raw data
save(   train.id, train.target, train.feat, train.feat.unif, train.distFeat.unif,
        test.id, test.feat, test.feat.unif, test.distFeat.unif,
        train.nzv,
        file="/Users/alexstephens/Development/kaggle/otto/data/02_ProcessData_RawData.Rdata")


## base2 data
save(   train.id, train.target, train.feat.b2, train.feat.unif.b2, train.distFeat.unif.b2,
        test.id, test.feat.b2, test.feat.unif.b2, test.distFeat.unif.b2,
        train.nzv,
        file="/Users/alexstephens/Development/kaggle/otto/data/02_ProcessData_Base2Data.Rdata")






