library(readr)
library(xgboost)
library(caret)
library(pROC)
library(tidyr)
library(dplyr)
library(e1071)
split.df <- function(df, ratio) {
  h = createDataPartition(df$QuoteConversion_Flag, p = ratio, list = FALSE, times = 1)
  list(part1 = df[h, ], part2 = df[-h, ])
}
add.date.features <- function(df, date.cols) {
  for(dc in date.cols) {
    #col = strptime(df[, dc], format='%d%b%y:%H:%M:%S', tz="UTC")
    col = as.POSIXlt(df[[dc]], origin="1970-01-01", tz = "UTC")
    tmp.df = data.frame(
      mday = col$mday,
      mon = col$mon,
      year = col$year,
      wday = col$wday,
      quarters = as.numeric(gsub("Q", "", quarters(col))),
      days_since_origin = as.double(julian(col))
    )
    names(tmp.df) = paste0(dc, "_", names(tmp.df))
    # TODO: also calculate time diff using `difftime`
    #     for(dc2 in date.cols) {
    #       #col2 = strptime(df[, dc2], format='%d%b%y:%H:%M:%S', tz="UTC")
    #       col2 = as.POSIXlt(df[, dc2], origin="1970-01-01", tz = "UTC")
    #       diff.df = data.frame(day_diff = as.double(difftime(col, col2, units = "days")),
    #                            week_diff = as.double(difftime(col, col2, units = "weeks")))
    #       names(diff.df) = paste0(dc, "_timediff_", dc2, names(diff.df))
    #     }
    #     df = cbind(df, diff.df)
    df = cbind(df, tmp.df)
  }
  df
}
gc()
set.seed(88888)
cat("reading the train and test data\n")
train.full <- read_csv("./input/train.csv")
test.full  <- read_csv("./input/test.csv")
train.backup = train.full
test.backup = test.full
id_name = "QuoteNumber"
target_name = "QuoteConversion_Flag"
id_col = train.backup[[id_name]]
target_col = train.backup[[target_name]]
id_col_test = test.backup[[id_name]]
# best so far - cv, 50 rounds, 10-fold, 10% of data:
# 0.954719+0.004781 nzv7
# 0.954759+0.004728 2cols_nas_hardcode
# 
# 
# 
# 




exp_suffix = "linear_dep"
run_cv = 0
run_train = 1 - run_cv
full_data = 1 - run_cv
nrounds = ifelse(run_cv, 50, 2000)
eval_metric = "auc"
param_bests = list(
  max_depth = 7,
  eta = 0.02,
  subsample = 0.82,
  colsample_bytree = 0.66
)
eval_metric_max = c("auc"=TRUE, "rmse"=FALSE, "error"=FALSE, "logloss"=FALSE)

# best_cv_score = 0
# best_threshold = 0
# thresholds_df = data.frame()

# for (nzv_threshold in c(7)) {

gc()
set.seed(88888)

train.full = train.backup
test.full = test.backup

#   train.full$PersonalField84[is.na(train.full$PersonalField84)] = 100
#   train.full$PropertyField29[is.na(train.full$PropertyField29)] = 100
#   test.full$PersonalField84[is.na(test.full$PersonalField84)] = 100
#   test.full$PropertyField29[is.na(test.full$PropertyField29)] = 100

train.full[is.na(train.full)]   <- 100
test.full[is.na(test.full)]   <- 100

train.full = add.date.features(train.full, c("Original_Quote_Date"))
train.full <- train.full %>% select(-Original_Quote_Date)

test.full = add.date.features(test.full, c("Original_Quote_Date"))
test.full <- test.full %>% select(-Original_Quote_Date)


#   load(file = "near_zero_vars.obj")
#   nzv <- nearZeroVar(train.full[, !grepl(target_name, names(train.full))], saveMetrics= TRUE)
#   nzv[nzv$nzv,]
#   nzv_sorted = nzv[order(nzv$freqRatio, decreasing = T), ]
#   plot(subset(nzv_sorted, freqRatio > 50)$freqRatio[2:3])
#   
#   near_zero_to_remove = names(train.full)[!grepl(target_name, names(train.full))][nzv$nzv]
#   near_zero_to_remove = names(train.full)[nzv$freqRatio > nzv_threshold]
#   if (nzv_threshold < 1) {
#     near_zero_to_remove = c()
#   } else {
#     near_zero_to_remove = row.names(nzv_sorted[1:nzv_threshold, ])
#   }
#   if (nzv_threshold != -1)
#     near_zero_to_remove = c(near_zero_to_remove, row.names(nzv[nzv$zeroVar, ]))

nzv_cols = c("GeographicField10B", "PropertyField20", "PropertyField9", "PersonalField8", "PersonalField69", "PersonalField73", "PersonalField70")
zv_cols = c("PropertyField6", "GeographicField10A")
many_nas = c() #c("PersonalField84", "PropertyField29")
uniform = c(
#     "CoverageField1B"
#     "CoverageField2B"
#     "CoverageField3B"
#     "CoverageField4B"
#     "CoverageField11B"
#     "SalesField1B"
#     "SalesField2B"
#     "SalesField8"
#     "PropertyField16B",
#     "PropertyField21B",
#     "PropertyField24B",
#     "PropertyField26B",
#     "PropertyField39B"
#     "GeographicField1B",
#     "GeographicField2B",
#     "GeographicField3B",
#     "GeographicField3B",
#     "GeographicField3B",
#     "GeographicField3B",
#     "GeographicField4B",
#     "GeographicField17B",
#     "GeographicField18B",
#     "GeographicField19B",
#     "GeographicField20B",
#     "GeographicField28B",
#     "GeographicField29B",
#     "GeographicField30B"
#     "GeographicField31B"..62B (-47B)
            )

to_remove = c(nzv_cols, zv_cols)

train.full = train.full[, !(names(train.full) %in% to_remove)]
test.full = test.full[, !(names(test.full) %in% to_remove)]
# names(train.full)
# names(test.full)
# feature.names
feature.names <- names(train.full)[-c(1, 2)]

#   cat("starting preprocessing\n")
#   all_data = rbind(train.full[, feature.names], test.full[, feature.names])
#   pp_vals = preProcess(all_data, method = c("center", "scale", "BoxCox"))
#   all_data_pp = predict.preProcess(pp_vals, all_data)
#   train.full = all_data_pp[1:nrow(train.backup), ]
#   test.full = all_data_pp[(nrow(train.backup) + 1):(nrow(train.backup) + nrow(test.backup)), ]
#   train.full[[id_name]] = id_col
#   train.full[[target_name]] = target_col
#   test.full[[id_name]] = id_col_test
#   cat("finished preprocessing\n")

for (f in feature.names) {
  if (class(train.full[[f]])=="character") {
    levels <- unique(c(train.full[[f]], test.full[[f]]))
    train.full[[f]] <- as.integer(factor(train.full[[f]], levels=levels))
    test.full[[f]]  <- as.integer(factor(test.full[[f]],  levels=levels))
  }
}

# corrs = cor(train.full[, feature.names])
# highlyCorDescr <- findCorrelation(corrs, cutoff = 0.8)
# high_corrs = names(train.full[, feature.names])[highlyCorDescr]

train_without_target = train.full[, feature.names]
comboInfo = findLinearCombos(train_without_target)
linear_dep_to_remove = comboInfo$remove
linear_dep = names(train.full[, feature.names])[linear_dep_to_remove]

to_remove = c(to_remove, linear_dep)

train.full = train.full[, !(names(train.full) %in% to_remove)]
test.full = test.full[, !(names(test.full) %in% to_remove)]
feature.names <- names(train.full)[-c(1, 2)]

#### prepare for training ####

gc()
set.seed(9)

train.rows.percent = 0.9
if (full_data) {
  train.split = split.df(train.full, train.rows.percent)
  train.t = train.split$part1
  train.v = train.split$part2
  train.on.both = train.full
  train.on.v = train.v
  train.on.t = train.t
} else {
  train.mini = split.df(train.full, 0.1)$part1
  train.mini.split = split.df(train.mini, train.rows.percent)
  train.mini.t = train.mini.split$part1
  train.mini.v = train.mini.split$part2
  train.on.both = train.mini
  train.on.v = train.mini.v
  train.on.t = train.mini.t
}

#### set params ####

param = list(   objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = eval_metric,
                max_depth           = param_bests$max_depth,
                eta                 = param_bests$eta,
                subsample           = param_bests$subsample,
                colsample_bytree    = param_bests$colsample_bytree
                # num_parallel_tree   = 2,
                # alpha               = 0.0001,
                # lambda              = 1
)
# dval   = xgb.DMatrix(data=data.matrix(train.on.v[, feature.names]),label=train.on.v$QuoteConversion_Flag)
# dtrain = xgb.DMatrix(data=data.matrix(train.on.t[, feature.names]),label=train.on.t$QuoteConversion_Flag)
dtrain = xgb.DMatrix(data=data.matrix(train.on.both[, feature.names]),label=train.on.both$QuoteConversion_Flag)
exp_name = paste(sep = "_", param$eval_metric, nrow(dtrain), nrounds,
                 param$max_depth, param$eta, param$subsample, param$colsample_bytree, 
                 exp_suffix)

# watchlist = list()
# if (nrow(dval) > 0)
#   watchlist$val = dval
# watchlist$train = dtrain
# early = round(nrounds / 10)

#### Cross Validation ####

if (run_cv) {
  
  time_before_cv = Sys.time()
  nrow(dtrain)
  
  gc()
  set.seed(110389)
  
  cv.res = xgb.cv(data = dtrain,
                  objective = "binary:logistic",
                  eval_metric = "auc",
                  nrounds = nrounds,
                  nfold = 10,
                  max_depth           = param_bests$max_depth,
                  eta                 = param_bests$eta,
                  subsample           = param_bests$subsample,
                  colsample_bytree    = param_bests$colsample_bytree,
                  verbose = 1
  )
#   save(cv.res, file = paste(exp_name, "cv", 
#                             paste(cv.res[nrow(cv.res)]$test.auc.mean, cv.res[nrow(cv.res)]$test.auc.std, sep = "+"),
#                             sep = "_"))
  
#   if (best_cv_score < cv.res[nrow(cv.res)]$test.auc.mean) {
#     best_cv_score = cv.res[nrow(cv.res)]$test.auc.mean
#     best_threshold = nzv_threshold
#   }
#   
#   thresholds_df = rbind(thresholds_df, data.frame(nzv_threshold, cv.res[nrow(cv.res)]$test.auc.mean))
  cat(cv.res[nrow(cv.res)]$test.auc.mean, "\n")
  load(file = "best_so_far")
  compare_cv = data.frame(best = best_so_far, test = cv.res$test.auc.mean)
  plot = ggplot(compare_cv %>% gather(key, value, best, test)) + 
    geom_line(aes(x = c(seq(1, 50), seq(1, 50)), y = value, fill = key, color = key)) +
    coord_cartesian(ylim = c(0.95, 0.957))
  print(plot)

  if (F) {
    save(best_so_far, file = "best_so_far")
  }
  
  time_before_training = Sys.time()
  cat("cv: "); time_before_training - time_before_cv
  
}
  
# } # end of loop

#### Train the model ####
if (run_train) {
  time_before_training = Sys.time()
  gc()
  set.seed(110389)
  
  model = xgb.train(  params              = param, 
                      data                = dtrain, 
                      nrounds             = nrounds,
                      verbose             = 1,  #1
                      # early.stop.round    = early,
                      # watchlist           = watchlist,
                      maximize            = FALSE
                      # print.every.n       = nrounds
  )
  # val_pred = predict(model, data.matrix(train.on.v[, feature.names]), ntreelimit = model$bestInd)
  # score = as.numeric(auc(train.on.v$QuoteConversion_Flag, val_pred))
  
  test1 <- test.full[,feature.names]
  pred1 <- predict(model, data.matrix(test1), ntreelimit = model$bestInd)
  submission <- data.frame(QuoteNumber=test.full$QuoteNumber, QuoteConversion_Flag=pred1)
  
  cat("saving the submission file\n")
  write_csv(submission, paste("output/", exp_name, ".csv", sep = ""))
  xgb.save(model, paste("models/", exp_name, ".xgb", sep = ""))
  gc()
  exp_name
  
  time_end = Sys.time()
  cat("tr: "); time_end - time_before_training
}
