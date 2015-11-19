
library(readr)
library(xgboost)
library(caret)
library(pROC)

split.df <- function(df, ratio) {
  h = createDataPartition(df$QuoteConversion_Flag, p = ratio, list = FALSE, times = 1)
  list(part1 = df[h, ], part2 = df[-h, ])
}

set.seed(88888)

cat("reading the train and test data\n")
train.full <- read_csv("./input/train.csv")
test.full  <- read_csv("./input/test.csv")

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)


# seperating out the elements of the date column for the train set
train.full$month <- as.integer(format(train.full$Original_Quote_Date, "%m"))
train.full$year <- as.integer(format(train.full$Original_Quote_Date, "%y"))
train.full$day <- weekdays(as.Date(train.full$Original_Quote_Date))

# removing the date column
train.full <- train.full[,-c(2)]

# seperating out the elements of the date column for the test set
test.full$month <- as.integer(format(test.full$Original_Quote_Date, "%m"))
test.full$year <- as.integer(format(test.full$Original_Quote_Date, "%y"))
test.full$day <- weekdays(as.Date(test.full$Original_Quote_Date))

# removing the date column
test.full <- test.full[,-c(2)]

feature.names <- names(train.full)[c(3:301)]

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train.full[[f]])=="character") {
    levels <- unique(c(train.full[[f]], test.full[[f]]))
    train.full[[f]] <- as.integer(factor(train.full[[f]], levels=levels))
    test.full[[f]]  <- as.integer(factor(test.full[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train.full)
cat("test data column names after slight feature engineering\n")
names(test.full)

# set.seed(9)
# train <- train[sample(nrow(train), 5000),]
gc()

tra<-train[,feature.names]
# tra<-tra[,c(1:50,65:105,110:165,180:230,245:290)]
dim(tra)
dim(test)

nrow(train)
h<-sample(nrow(train),2000)



dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$QuoteConversion_Flag[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$QuoteConversion_Flag[-h])
# dtrain<-xgb.DMatrix(data=data.matrix(tra[,]),label=train$QuoteConversion_Flag)

watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "binary:logistic", 
                booster = "gbtree",
                eval_metric = "auc",
                eta                 = 0.02, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.85, # 0.7
                colsample_bytree    = 0.66 # 0.7
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 10, #1800, 
                    verbose             = 0,  #1
                    #early.stop.round    = 150,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

val_pred = predict(clf, data.matrix(tra[h, ]))
auc(train[h, ]$QuoteConversion_Flag, val_pred)

test1<-test[,feature.names]
dim(test)
# test1<-test1[,c(1:50,65:105,110:165,180:230,245:290)]

pred1 <- predict(clf, data.matrix(test1))
submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
cat("saving the submission file\n")
write_csv(submission, "output/xgb_stop_1800_10.csv")
