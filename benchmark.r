
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
train.full[is.na(train.full)]   <- -1
test.full[is.na(test.full)]   <- -1

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

#### prepare for training ####

set.seed(9)

train = train.full

train.rows.percent = 0.99

train.mini = split.df(train, 0.3)$part1
train.mini.split = split.df(train.mini, train.rows.percent)
train.mini.t = train.mini.split$part1
train.mini.v = train.mini.split$part2

train.split = split.df(train, train.rows.percent)
train.t = train.split$part1
train.v = train.split$part2

train.on.v = train.v
train.on.t = train.t
# train.on.v = train.mini.v
# train.on.t = train.mini.t

dval   = xgb.DMatrix(data=data.matrix(train.on.v[, feature.names]),label=train.on.v$QuoteConversion_Flag)
dtrain = xgb.DMatrix(data=data.matrix(train.on.t[, feature.names]),label=train.on.t$QuoteConversion_Flag)

watchlist = list()
if (nrow(dval) > 0)
  watchlist$val = dval
watchlist$train = dtrain

param = list(  objective           = "binary:logistic", 
                booster = "gbtree",
                eval_metric = "auc",
                max_depth           = 8, #changed from default of 8
                eta                 = 0.01, # 0.0025, 0.006, 0.01, 0.015, 0.02, 0.06
                subsample           = 0.85, # 0.7
                colsample_bytree    = 0.7 # 0.7
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)

nrounds = 3000
early = round(nrounds / 5)

model = xgb.train(  params              = param, 
                    data                = dtrain, 
                    nrounds             = nrounds,
                    verbose             = 0,  #1
                    early.stop.round    = early,
                    watchlist           = watchlist,
                    maximize            = TRUE
)

val_pred = predict(model, data.matrix(train.on.v[, feature.names]), ntreelimit = model$bestInd)
auc(train.on.v$QuoteConversion_Flag, val_pred)

test1<-test.full[,feature.names]
# test1<-test1[,c(1:50)]

pred1 <- predict(model, data.matrix(test1), ntreelimit = model$bestInd)
submission <- data.frame(QuoteNumber=test.full$QuoteNumber, QuoteConversion_Flag=pred1)

cat("saving the submission file\n")
exp_name = paste(sep = "", train.rows.percent, "_", early, "_", nrounds, "_", 
                 param$max_depth, "_", param$eta, "_", param$subsample, "_", param$colsample_bytree, "")
write_csv(submission, paste("output/", exp_name, ".csv", sep = ""))
xgb.save(model, paste("models/", exp_name, ".xgb", sep = ""))
