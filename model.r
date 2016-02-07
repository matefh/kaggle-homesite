stop_before_cv_train = 0
cv_states = c(1)

#### libraries ####
if (!("readr" %in% installed.packages())) {
  install.packages("readr")
  install.packages("tidyr")
  install.packages("dplyr")
  install.packages("e1071")
}
library(readr)
library(xgboost)
library(caret)
library(pROC)
library(tidyr)
library(dplyr)
library(lazyeval)
library(e1071)
library(neuralnet)
library(nnet)
#### functions ####
split.df <- function(df, ratio) {
  h = createDataPartition(df$QuoteConversion_Flag, p = ratio, list = FALSE, times = 1)
  list(part1 = df[h, ], part2 = df[-h, ])
}
add.date.features <- function(df, date.cols) {
  for(dc in date.cols) {
    #col = strptime(df[, dc], format='%d%b%y:%H:%M:%S', tz="UTC")
    col = as.POSIXlt(df[[dc]], origin="1970-01-01", tz = "UTC")
    tmp.df = data.frame(
      week = ceiling(col$yday / 7) ,
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

map.missing <- function(df, to.replace = "na", threshold = 10, impute_method = "mean") {
  names = names(df)
  names = names[!names %in% c(target_name, id_name)]
  sum(is.na(df)) / (nrow(df) * ncol(df))
  for (f in names) {
    col = df[[f]]
    value = NULL
    if (class(col) == "numeric" | class(col) == "double" | class(col) == "integer") {
      if (impute_method == "mean") {
        value = mean(col, na.rm = T)
      } else if (impute_method == "median") {
        value = median(col, na.rm = T)
      } else if(impute_method == "mostfreq") {
        value = as.numeric(names(which.max(table(col[col != to.replace]))))
      } else if(impute_method == "max+1") {
        value = max(col, na.rm = T) + 1
      } else if(impute_method == "min-1") {
        value = min(col, na.rm = T) - 1
      }
      if (length(value) > 0) {
        if (tolower(to.replace) == "na") {
          df[[f]][is.na(col)] = value
        } else if (tolower(to.replace) == "inf") {
          df[[f]][looks.like.inf(col)] = value
        } else if (to.replace == -1) {
          df[[f]][col == -1] = value
        }
      }
    }
  }
  df
}
count_per_row <- function(df, val) {
  if (is.na(val)) {
    rowSums(is.na(df))
  } else {
    rowSums(df == val)
  }
}
replace_transformed <- function(tr_name, df, tr, features) {
  colnames(tr) = paste0(tr_name, "_", features)
  cbind(df[, !names(df) %in% features], tr)
}
add_transformed <- function(tr_name, df, tr, features) {
  colnames(tr) = paste0(tr_name, "_", features)
  cbind(df, tr)
}
create_features <- function(df, df2, feature_names) {
  names(df2) = feature_names
  cbind(df, df2)
}
to_char <- function(df) {
  for (f in names(df))
    df[[f]] = as.character(df[[f]])
  df
}

#### reading data ####
gc()
set.seed(88888)
cat("reading the train and test data\n")
train.full = read_csv("./input/train.csv")
test.full = read_csv("./input/test.csv")
train.backup = train.full
test.backup = test.full
id_name = "QuoteNumber"
target_name = "QuoteConversion_Flag"
id_col = train.backup[[id_name]]
target_col = train.backup[[target_name]]
id_col_test = test.backup[[id_name]]
# load(file = "models/imp_auc_260753_1800_7_0.02_0.82_0.66_")
# groups = 5
# featgroup = 4
# chunksize = nrow(imp) / groups
# train_with_features = imp[seq(1 + chunksize * featgroup, chunksize * (featgroup + 1))]$Feature
train_with_features = c()#imp[seq(0, nrow(imp) - 1) %% groups == featgroup]$Feature
# train_with_features = imp$Feature

# corr vars
# save input auc_260753_1800_7_0.02_0.82_0.66_csbc_lg_sqrt_numr2_missingcounts2_dummies_datedummies_lg_sqrt_imp_golden_features_keep_cols_most_corr_target_0.15_0.3_1139.csv

#### initializing vars ####
# 0.83, 0.88
search_grid = expand.grid(subsample = c( 0.76, 0.81 ), colsample_bytree = c(0.6, 0.65, 0.67, 0.72), max_depth = c(7, 6, 8, 5, 9))


for(run_cv in cv_states)
for(keep_cols in c(1))
for(with_corr_features in c(0))
for(with_missing_counts in c(0)) # 0 or 2 but not 1
for(with_transform_csbc in c(0)) # 0 or 2 but not 1
for(with_logs_sqrt_all in c(0))
for(with_dummyvars in c(0))
for(with_date_dummies in c(0))
for(with_date_split in c(0))
for(with_value_counts in c(0))
for(with_cat_target in c(0))
for(with_csbc in c(0)) # keep 0
for(with_logs_sqrts_imp in c(0))
for(with_lindep in c(0))
for(with_most_corr_target in c(0))
for(with_golden_features in c(0))
{
  nrounds = ifelse(run_cv, 1400, 1800)
  harcoded_remove = c()
  full_data = 0# - run_cv # keep it 1 due to issue with split.df and value_counts
  train.rows.percent = 0.9
  train_with_all = 1
  save_data = 1
  load_data = save_data
  run_train = 1 - run_cv
  seed_value = 36 # 819 98253   516    36   925    28     7     2   580 62929, 110389
  
  model_name = 'xgb'
  corr_iterations = 1
  highest_corr_threshold = 0.15
  final_corr_threshold = 0.30
  corr_features_count = 50
  
  exp_list = c(
    ifelse(with_transform_csbc, paste0("csbc_lg_sqrt_numr", with_transform_csbc), ""),
    ifelse(with_missing_counts, paste0("missingcounts", with_missing_counts), ""),
    ifelse(with_corr_features, paste0("corr_features", corr_features_count), ""),
    ifelse(with_dummyvars, "dummies", ""),
    ifelse(with_date_dummies, "datedummies", ""),
    ifelse(with_date_split, "full_date_split", ""),
    ifelse(with_cat_target, "cat_target_keep_orig", ""),
    ifelse(with_csbc, paste0("csbc_lg_sqrt_all", with_csbc), ""),
    ifelse(with_logs_sqrts_imp, "lg_sqrt_imp", ""),
    ifelse(with_lindep, "lindep", ""),
    ifelse(with_value_counts, "value_counts", ""),
    ifelse(with_golden_features, "golden_features", ""),
    ifelse(keep_cols, "keep_cols", ""),
    ifelse(with_most_corr_target, paste("most_corr_target", corr_iterations, highest_corr_threshold, final_corr_threshold, sep = "_"), "")
  )
  exp_suffix = paste0(collapse = "_", exp_list[exp_list != ""])
  cat(exp_suffix, "\n")
  
  eval_metric = "auc"
  param_bests = list(
    eta = 0.01,
    max_depth = 7,
    subsample = 1,
    colsample_bytree = 0.6
    #     eta                 = 0.023, # 0.06, #0.01,
    #     max_depth           = 6, #changed from default of 8
    #     subsample           = 0.83, # 0.7
    #     colsample_bytree    = 0.77 # 0.7
  )
  eval_metric_max = c("auc"=TRUE, "rmse"=FALSE, "error"=FALSE, "logloss"=FALSE)
  gc()
  set.seed(88888)
  
  # best_cv_score = 0
  # best_threshold = 0
  # thresholds_list = list()
  
#### preprocessing ####
  if (load_data & file.exists(paste0("input/train_pp_", exp_suffix, ".rds"))) {
    train.full = readRDS(paste0("input/train_pp_", exp_suffix, ".rds"))
    test.full = readRDS(paste0("input/test_pp_", exp_suffix, ".rds"))
    feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
  } else {
    train.full = train.backup
    test.full = test.backup
    categ = names(train.full[, sapply(train.full, is.character)])
    train.full$GeographicField63[train.full$GeographicField63 == " "] = ""
    
    train.full[is.na(train.full)]   <- 100
    test.full[is.na(test.full)]   <- 100
    
    #       if (impute_method != "") {
    #         train.full = map.missing(train.full, to.replace = -1, impute_method = impute_method)
    #         test.full = map.missing(test.full, to.replace = -1, impute_method = impute_method)
    #       }
    
    train.full = add.date.features(train.full, c("Original_Quote_Date"))
    train.full <- train.full %>% dplyr::select(-Original_Quote_Date)
    
    test.full = add.date.features(test.full, c("Original_Quote_Date"))
    test.full <- test.full %>% dplyr::select(-Original_Quote_Date)
    
    if (T) {
      neg_tab = sort(sapply(train.backup, function(col) sum(col == -1))) / nrow(train.backup)
      all_neg_cols = neg_tab[neg_tab > 0.01]
      neg_cols = names(all_neg_cols[7:length(all_neg_cols)])
      # neg_cols = c("GeographicField61A", "GeographicField5A", "GeographicField60A", "PropertyField11A", "GeographicField21A", "GeographicField10A")
      nzv_cols = c("GeographicField10B", "PropertyField20", "PropertyField9", "PersonalField8", "PersonalField69", "PersonalField73", "PersonalField70")
      zv_cols = c("PropertyField6", "GeographicField10A")
      # linear_dep = c("PersonalField65", "PersonalField67", "PersonalField80", "PersonalField81", "PersonalField82")
      to_remove = c()
      if (keep_cols) {
        to_remove = c(nzv_cols, zv_cols, harcoded_remove)
        to_remove = c(neg_cols, to_remove)
        train.full = train.full[, !(names(train.full) %in% to_remove)]
        test.full = test.full[, !(names(test.full) %in% to_remove)]
      }
    }
    
    feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
    original_feature.names = feature.names
    
    print(1)
    if (with_missing_counts > 0) {
      train_missing_negones = count_per_row(train.full[, feature.names], -1)
      train_missing_blank = count_per_row(train.full[, feature.names], "")
      train_missing_na = count_per_row(train.backup[, -which(names(train.backup) %in% c(target_name, id_name))], NA) # note backup
      train_missing_zeros = count_per_row(train.full[, feature.names], 0)
      
      test_missing_negones = count_per_row(test.full[, feature.names], -1)
      test_missing_blank = count_per_row(test.full[, feature.names], "")
      test_missing_na = count_per_row(test.backup[, -which(names(test.backup) %in% c(target_name, id_name))], NA) # note backup
      test_missing_zeros = count_per_row(test.full[, feature.names], 0)
      
      if (with_missing_counts == 1) {
        train.full$missing_all = train_missing_negones + train_missing_blank + train_missing_na + train_missing_zeros
        test.full$missing_all = test_missing_negones + test_missing_blank + test_missing_na + test_missing_zeros
      }
      
      if (with_missing_counts == 2) {
        train.full$missing_blank = train_missing_blank
        train.full$missing_na = train_missing_na
        train.full$missing_negones = train_missing_negones
        train.full$missing_zeros = train_missing_zeros
        train.full$missing_zero_na = train_missing_zeros + train_missing_na
        train.full$missing_blank_na = train_missing_blank + train_missing_na
        train.full$missing_negones_na = train_missing_negones + train_missing_na
        train.full$missing_zero_blank = train_missing_zeros + train_missing_blank
        train.full$missing_zero_neg_ones = train_missing_zeros + train_missing_negones
        train.full$missing_negones_blank = train_missing_negones + train_missing_blank
        train.full$missing_negones_blank_na = train_missing_negones + train_missing_blank + train_missing_na
        train.full$missing_zero_blank_na = train_missing_blank + train_missing_na + train_missing_zeros
        train.full$missing_zero_blank_negones = train_missing_negones + train_missing_blank + train_missing_zeros
        train.full$missing_zero_negones_na = train_missing_negones + train_missing_na + train_missing_zeros
        train.full$missing_all = train_missing_negones + train_missing_blank + train_missing_na + train_missing_zeros
        test.full$missing_blank = test_missing_blank
        test.full$missing_na = test_missing_na
        test.full$missing_negones = test_missing_negones
        test.full$missing_zeros = test_missing_zeros
        test.full$missing_zero_na = test_missing_zeros + test_missing_na
        test.full$missing_blank_na = test_missing_blank + test_missing_na
        test.full$missing_negones_na = test_missing_negones + test_missing_na
        test.full$missing_zero_blank = test_missing_zeros + test_missing_blank
        test.full$missing_zero_neg_ones = test_missing_zeros + test_missing_negones
        test.full$missing_negones_blank = test_missing_negones + test_missing_blank
        test.full$missing_negones_blank_na = test_missing_negones + test_missing_blank + test_missing_na
        test.full$missing_zero_blank_na = test_missing_blank + test_missing_na + test_missing_zeros
        test.full$missing_zero_blank_negones = test_missing_negones + test_missing_blank + test_missing_zeros
        test.full$missing_zero_negones_na = test_missing_negones + test_missing_na + test_missing_zeros
        test.full$missing_all = test_missing_negones + test_missing_blank + test_missing_na + test_missing_zeros
      }
      
      if (with_missing_counts == 3) {
        train.full$missing_zeros = train_missing_zeros
        train.full$missing_negones_na = train_missing_negones + train_missing_na
        test.full$missing_zeros = test_missing_zeros
        test.full$missing_negones_na = test_missing_negones + test_missing_na
      }
      
      if (with_missing_counts == 4) {
        train.full$missing_zero_negones_na = train_missing_negones + train_missing_na + train_missing_zeros
        test.full$missing_zero_negones_na = test_missing_negones + test_missing_na + test_missing_zeros
      }
      
      if (with_missing_counts == 5) {
        train.full$missing_zeros = train_missing_zeros
        test.full$missing_zeros = test_missing_zeros
      }
      
      if (with_missing_counts == 6) {
        train.full$missing_zeros = train_missing_zeros
        train.full$missing_negones_na = train_missing_negones + train_missing_na
        train.full$missing_zero_negones_na = train_missing_negones + train_missing_na + train_missing_zeros
        test.full$missing_zeros = test_missing_zeros
        test.full$missing_negones_na = test_missing_negones + test_missing_na
        test.full$missing_zero_negones_na = test_missing_negones + test_missing_na + test_missing_zeros
      }
      feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
    }
    print(2)
    if (with_transform_csbc) {
      numr = names(train.full[, sapply(train.full, is.numeric)])
      numr_only = numr[sapply(numr, function(f) !(substring(f, nchar(f)) %in% c("A", "B", "missing")))]
      numr_only = numr_only[!numr_only %in% c(target_name, id_name)]
      all_data = rbind(train.full[, numr_only], test.full[, numr_only])
      pp_vals = preProcess(all_data, method = c("center", "scale", "BoxCox"))
      all_data_pp = predict.preProcess(pp_vals, all_data)
      colnames(all_data_pp) = paste0("csbc_", names(all_data_pp))
      
      if (with_logs_sqrt_all) {
        logs = numr_only[sapply(numr_only, function(f) min(train.full[[f]]) > 0)]
        sqrts = numr_only[sapply(numr_only, function(f) min(train.full[[f]]) >= 0)]
        train.full = add_transformed("log", train.full, data.frame(log(train.full[, logs])), logs)
        train.full = add_transformed("sqrt", train.full, data.frame(sqrt(train.full[, sqrts])), sqrts)
        test.full = add_transformed("log", test.full, data.frame(log(test.full[, logs])), logs)
        test.full = add_transformed("sqrt", test.full, data.frame(sqrt(test.full[, sqrts])), sqrts)
      }
      if (with_transform_csbc == 1) { # replace 
        train.full = cbind(train.full[, !names(train.full) %in% numr_only], all_data_pp[1:nrow(train.backup), ])
        test.full = cbind(test.full[, !names(test.full) %in% numr_only], all_data_pp[(nrow(train.backup) + 1):(nrow(train.backup) + nrow(test.backup)), ])
      } else { # add
        train.full = cbind(train.full, all_data_pp[1:nrow(train.backup), ])
        test.full = cbind(test.full, all_data_pp[(nrow(train.backup) + 1):(nrow(train.backup) + nrow(test.backup)), ])
      }
      train.full[[id_name]] = id_col
      train.full[[target_name]] = target_col
      test.full[[id_name]] = id_col_test
      feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
    }
    print(3)
    if (with_dummyvars) {
      train.dummies = dummyVars( ~ ., data = train.full)
      new.vars = data.frame(predict(train.dummies, newdata = train.full))
      train.full = new.vars#cbind(train.full[, -which(names(train.full) %in% names(new.vars))], new.vars)
      #       train.full[[id_name]] = id_col
      #       train.full[[target_name]] = target_col
      
      test.dummies = dummyVars( ~ ., data = test.full)
      new.vars = data.frame(predict(test.dummies, newdata = test.full))
      test.full = new.vars#cbind(test.full[, -which(names(test.full) %in% names(new.vars))], new.vars)
      
      feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
      gc()
    }
    print(4)
    if (with_date_dummies) {
      train.dummies = dummyVars( ~ factor(Original_Quote_Date_mon) + 
                                   factor(Original_Quote_Date_wday) +
                                   factor(Original_Quote_Date_mday) +
                                   factor(Original_Quote_Date_year) +
                                   factor(Original_Quote_Date_week) +
                                   factor(Original_Quote_Date_quarters)
                                 , data = train.full)
      train.full = cbind(train.full, data.frame(predict(train.dummies, newdata = train.full)))
      # train.full[[id_name]] = id_col
      # train.full[[target_name]] = target_col
      
      test.dummies = dummyVars( ~ factor(Original_Quote_Date_mon) + 
                                  factor(Original_Quote_Date_wday) +
                                  factor(Original_Quote_Date_mday) +
                                  factor(Original_Quote_Date_year) +
                                  factor(Original_Quote_Date_week) +
                                  factor(Original_Quote_Date_quarters), data = test.full)
      test.full = cbind(test.full, data.frame(predict(test.dummies, newdata = test.full)))
      
      feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
      for(col in setdiff(feature.names, names(test.full)))
        test.full[[col]] = 0
    }
    print(5)
    if (with_date_split) {
      date_feature = "Original_Quote_Date_days_since_origin"
      if (date_feature != "") {
        from = min(train.full[[date_feature]])
        to = max(train.full[[date_feature]])
        for (ds in seq(from, to, 3)) {
          new_feature = paste0(date_feature, "_", ds)
          train.full[[new_feature]] = ifelse(train.full[[date_feature]] > ds, 1, 0)
        }
        from = min(test.full[[date_feature]])
        to = max(test.full[[date_feature]])
        for (ds in seq(from, to, 3)) {
          new_feature = paste0(date_feature, "_", ds)
          test.full[[new_feature]] = ifelse(test.full[[date_feature]] > ds, 1, 0)
        }
      }
      feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
    }
    print(6)
    for (f in feature.names) {
      if (class(train.full[[f]])=="character") {
        levels <- unique(c(train.full[[f]], test.full[[f]]))
        train.full[[f]] <- as.integer(factor(train.full[[f]], levels=levels))
        test.full[[f]]  <- as.integer(factor(test.full[[f]],  levels=levels))
      }
    }
    print(6.5)
    if (with_corr_features) {
      corr_features = cor(train.full[, intersect(names(train.full), original_feature.names)])
      corr_features[lower.tri(corr_features, diag = T)] = NA
      corr_features = as.data.frame(as.table(corr_features))
      corr_features = na.omit(corr_features)
      corr_features_sorted = corr_features[order(abs(corr_features$Freq), decreasing = T), ]
      corr_features_sorted = corr_features_sorted[abs(corr_features$Freq) > 0.5, ]
      corr_features_sorted = corr_features_sorted[1:corr_features_count, ]
    }
    print(7)
    if (with_value_counts) {
      for (f in feature.names) {
        ftab = table(c(train.full[[f]], test.full[[f]]))
        train.full[[paste0(f, "_count")]] = ftab[as.character(train.full[[f]])]
        test.full[[paste0(f, "_count")]] = ftab[as.character(test.full[[f]])]
      }
      feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
    }
    print(8)
    if (with_cat_target) {
      for (f in categ) {
        vals = unique(train.full[[f]])
        replaces = list()
        for (val in vals) {
          group = train.full %>% filter_(interp(quote(x == y), x = as.name(f), y = val)) %>% select_(target_name)
          target_mean = sum(group) / nrow(group)
          replaces[[val]] = target_mean
        }
        the_mean = mean(unlist(replaces))
        train.full[[paste0(f, "_target_mean")]] = sapply(train.full[[f]], function(cell) replaces[[cell]])
        test.full[[paste0(f, "_target_mean")]] = sapply(test.full[[f]], function(cell) ifelse(cell %in% names(replaces), replaces[[cell]], the_mean))
      }
      feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
    }
    print(9)
    if (with_csbc) {
      all_data = rbind(train.full[, feature.names], test.full[, feature.names])
      pp_vals = preProcess(all_data, method = c("center", "scale", "BoxCox"))
      all_data_pp = predict.preProcess(pp_vals, all_data)
      if (with_csbc == 1) {
        train.full = all_data_pp[1:nrow(train.backup), ]
        test.full = all_data_pp[(nrow(train.backup) + 1):(nrow(train.backup) + nrow(test.backup)), ]
      } else {
        colnames(all_data_pp) = paste0("csbc_", names(all_data_pp))
        train.full = cbind(train.full, all_data_pp[1:nrow(train.backup), ])
        test.full = cbind(test.full, all_data_pp[(nrow(train.backup) + 1):(nrow(train.backup) + nrow(test.backup)), ])
        feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
      }
      train.full[[id_name]] = id_col
      train.full[[target_name]] = target_col
      test.full[[id_name]] = id_col_test
    }
    print(10)
    if (with_logs_sqrts_imp) {
      to_log = c("SalesField5", "PersonalField12", "PersonalField13")
      to_sqrt = c("PropertyField29", "PersonalField1", "PersonalField2", "SalesField5")
      for (f in to_log) {
        train.full[[paste0("log_", f)]] = log(train.full[[f]])
        test.full[[paste0("log_", f)]] = log(test.full[[f]])
      }
      for (f in to_sqrt) {
        train.full[[paste0("sqrt_", f)]] = sqrt(train.full[[f]])
        test.full[[paste0("sqrt_", f)]] = sqrt(test.full[[f]])
      }
      feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
    }
    print(11)
    if (with_lindep) {
      train_without_target = train.full[, feature.names]
      comboInfo = findLinearCombos(train_without_target)
      # save(comboInfo, file = "comboInfo") ## saving
      # load(file = "comboInfo")
      linear_dep_to_remove = comboInfo$remove
      linear_dep = names(train.full[, feature.names])[linear_dep_to_remove]
      to_remove = c(to_remove, linear_dep)
      train.full = train.full[, !(names(train.full) %in% to_remove)]
      test.full = test.full[, !(names(test.full) %in% to_remove)]
      feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
    }
    print(12)
    if (length(train_with_features) > 0) {
      train_with_features = c(train_with_features, setdiff(feature.names, imp$Feature))
      train.full = train.full[, c(id_name, target_name, train_with_features)]
      test.full = test.full[, c(id_name, train_with_features)]
    }
    print(13)
    for (corr_ind in seq(1, corr_iterations))
    if (with_most_corr_target) {
      cat("corr_ind", corr_ind, "\n")
      corr_target = cor(train.full[, feature.names], train.full$QuoteConversion_Flag)
      sorted_corr = corr_target[order(abs(corr_target), decreasing = T), ]
      highest_corr_features = names(sorted_corr[abs(sorted_corr) >= highest_corr_threshold])
      highest_corr_features = highest_corr_features[1:40]
      highest_corr_features = highest_corr_features[!is.na(highest_corr_features)]
      
      inv = highest_corr_features[sapply(highest_corr_features, function(f) !(0 %in% unique(train.full[[f]])))]
      sub = expand.grid(highest_corr_features, highest_corr_features)
      sub = sub[as.character(sub$Var1) != as.character(sub$Var2), ]
      div = expand.grid(highest_corr_features, inv)
      div = div[as.character(div$Var1) != as.character(div$Var2), ]
      add_mult = expand.grid(highest_corr_features, highest_corr_features)
      add_mult = add_mult[as.character(add_mult$Var1) < as.character(add_mult$Var2), ]
      
      sub = to_char(sub)
      div = to_char(div)
      add_mult = to_char(add_mult)
      
      inv_df = 1 / train.full[, inv]
      train.full = create_features(train.full, inv_df, paste0("1/", inv))
      inv_df = 1 / test.full[, inv]
      test.full = create_features(test.full, inv_df, paste0("1/", inv))
      
      sub_df = train.full[, sub$Var1] - train.full[, sub$Var2]
      train.full = create_features(train.full, sub_df, paste0(sub$Var1, "-", sub$Var2))
      sub_df = test.full[, sub$Var1] - test.full[, sub$Var2]
      test.full = create_features(test.full, sub_df, paste0(sub$Var1, "-", sub$Var2))
      
      div_df = train.full[, div$Var1] / train.full[, div$Var2]
      train.full = create_features(train.full, div_df, paste0(div$Var1, "/", div$Var2))
      div_df = test.full[, div$Var1] / test.full[, div$Var2]
      test.full = create_features(test.full, div_df, paste0(div$Var1, "/", div$Var2))
      
      mult_df = train.full[, add_mult$Var1] * train.full[, add_mult$Var2]
      train.full = create_features(train.full, mult_df, paste0(add_mult$Var1, "*", add_mult$Var2))
      mult_df = test.full[, add_mult$Var1] * test.full[, add_mult$Var2]
      test.full = create_features(test.full, mult_df, paste0(add_mult$Var1, "*", add_mult$Var2))
      
      add_df = train.full[, add_mult$Var1] + train.full[, add_mult$Var2]
      train.full = create_features(train.full, add_df, paste0(add_mult$Var1, "+", add_mult$Var2))
      add_df = test.full[, add_mult$Var1] + test.full[, add_mult$Var2]
      test.full = create_features(test.full, add_df, paste0(add_mult$Var1, "+", add_mult$Var2))
      
      comb_feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
      
      corr_target = cor(train.full[, comb_feature.names], train.full$QuoteConversion_Flag)
      sorted_corr = corr_target[order(abs(corr_target), decreasing = T), ]
      highest_corr_features = names(sorted_corr[abs(sorted_corr) >= final_corr_threshold])
      if (corr_ind > 1)
        highest_corr_features = highest_corr_features[1:100]
      highest_corr_features = highest_corr_features[!is.na(highest_corr_features)]
      
      gc()
      rm("inv", "inv_df", "sub", "sub_df", "div", "div_df", "add_mult", "mult_df", "add_df", 
         "corr_target", "sorted_corr")
      gc()
      
      train.full = train.full[, (names(train.full) %in% c(feature.names, highest_corr_features, target_name, id_name))]
      test.full = test.full[, (names(test.full) %in% c(feature.names, highest_corr_features, target_name, id_name))]
      feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
    }
    print(13.5)
    if (with_corr_features) {
      fs = to_char(corr_features_sorted[c("Var1", "Var2")])
      fs_df = train.full[, fs$Var1] - train.full[, fs$Var2]
      train.full = create_features(train.full, fs_df, paste0(fs$Var1, "-", fs$Var2))
      fs_df = test.full[, fs$Var1] - test.full[, fs$Var2]
      test.full = create_features(test.full, fs_df, paste0(fs$Var1, "-", fs$Var2))
    }
    print(14)
    if (with_golden_features) {
      golden_features = list(c("CoverageField1B","PropertyField21B"),
                             c("GeographicField6A","GeographicField8A"),
                             c("GeographicField6A","GeographicField13A"),
                             c("GeographicField8A","GeographicField13A"),
                             c("GeographicField11A","GeographicField13A"),
                             c("GeographicField8A","GeographicField11A"))
      for(fs in golden_features) {
        if (is.null(train.full[[paste0(fs[1], "-", fs[2])]])) {
          train.full[[paste0(fs[1], "-", fs[2])]] = train.full[[fs[1]]] - train.full[[fs[2]]]
          test.full[[paste0(fs[1], "-", fs[2])]] = test.full[[fs[1]]] - test.full[[fs[2]]]
        }
      }
      feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
    }
    print(15)
    feature.names = names(train.full[, -which(names(train.full) %in% c(target_name, id_name))])
    if (save_data) {
      saveRDS(train.full, paste0("input/train_pp_", exp_suffix, ".rds"))
      saveRDS(test.full, paste0("input/test_pp_", exp_suffix, ".rds"))
    }
    cat(feature.names, "\n", length(feature.names), "\n")
    exp_suffix = paste0(exp_suffix, "_", length(feature.names))
    # cat("finished preprocessing\n")
  }
  gc()
  if (stop_before_cv_train)
    break
  #### prepare for training ####
  
  apply(search_grid, 1, function(param_set) {
  
    set.seed(9)
    
    if (full_data) {
      train.split = split.df(train.full, train.rows.percent)
      train.t = train.split$part1
      train.v = train.split$part2
      train.on.v = train.v
      train.on.t = train.t
      train.on.both = train.full
    } else {
      train.mini = split.df(train.full, 0.3)$part1
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
                    max_depth           = param_set[['max_depth']],
                    eta                 = param_bests$eta,
                    subsample           = param_set[['subsample']],
                    colsample_bytree    = param_set[['colsample_bytree']]
                    # num_parallel_tree   = 2,
                    # alpha               = 0.0001,
                    # lambda              = 1
    )
    h = sample(nrow(train.on.both), 20000)
    if (train_with_all) {
      dval = xgb.DMatrix(data=data.matrix(train.on.both[h, feature.names]),label=train.on.both[h, ]$QuoteConversion_Flag)
      dtrain = xgb.DMatrix(data=data.matrix(train.on.both[, feature.names]),label=train.on.both$QuoteConversion_Flag)
    } else {
      dval = xgb.DMatrix(data=data.matrix(train.on.v[, feature.names]),label=train.on.v$QuoteConversion_Flag)
      dtrain = xgb.DMatrix(data=data.matrix(train.on.t[, feature.names]),label=train.on.t$QuoteConversion_Flag)
    }
    exp_name = paste(sep = "_", param$eval_metric, nrow(dtrain), nrounds,
                     param$max_depth, param$eta, param$subsample, param$colsample_bytree,
                     exp_suffix)
    
    watchlist = list()
    if (nrow(dval) > 0)
      watchlist$val = dval
    watchlist$train = dtrain
    early = nrounds
    
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
                      nfold = 3,
                      max_depth           = param$max_depth,
                      eta                 = param$eta,
                      subsample           = param$subsample,
                      colsample_bytree    = param$colsample_bytree,
                      verbose = 1,
                      print.every.n = 1
      )
      #   save(cv.res, file = paste(exp_name, "cv", 
      #                             paste(cv.res[nrow(cv.res)]$test.auc.mean, cv.res[nrow(cv.res)]$test.auc.std, sep = "+"),
      #                             sep = "_"))
      
      #   if (best_cv_score < cv.res[nrow(cv.res)]$test.auc.mean) {
      #     best_cv_score = cv.res[nrow(cv.res)]$test.auc.mean
      #     best_threshold = neg_count
      #   }
      #   thresholds_list[[neg_count]] = cv.res
      if (F) {
        best_so_far = cv.res$test.auc.mean
        save(best_so_far, file = "best_so_far")
      }
      
      cv_score = cv.res[nrow(cv.res)]$test.auc.mean
      cat(exp_name, "\nCV score:", cv_score, "\n")
      write(cv_score, paste0("cv_scores/", exp_name, "_", cv_score, ".txt"))
      
      if (F & file.exists("best_so_far")) {
        load(file = "best_so_far")
        if (length(best_so_far) == length(cv.res$test.auc.mean)) {
          compare_cv = data.frame(best = best_so_far, test = cv.res$test.auc.mean)
          plot = ggplot(compare_cv %>% gather(key, value, best, test)) + 
            geom_line(aes(x = c(seq(1, nrounds), seq(1, nrounds)), y = value, fill = key, color = key))
          # coord_cartesian(ylim = c(0.95, 0.957))
          print(plot)
        }
      }
      # cat("Time:", Sys.time() - time_before_cv, "\n")
      
    }
    # }
    
    #### Train the model ####
    if (run_train == 1) {
	    set.seed(seed_value)
      exp_name = paste0(exp_name, model_name, paste0("_seed", seed_value))
  	  if (model_name == 'xgb') {
  	    time_before_training = Sys.time()
  	    gc()
  	    
  	    model = xgb.train(  params              = param, 
  				data                = dtrain, 
  				nrounds             = nrounds,
  				verbose             = 1,  #1
  				early.stop.round    = early,
  				watchlist           = watchlist,
  				maximize            = TRUE,
  				print.every.n       = 100
  	    )
  	    val_pred = predict(model, data.matrix(train.on.v[, feature.names]), ntreelimit = model$bestInd)
  	    score = as.numeric(auc(train.on.v[, ]$QuoteConversion_Flag, val_pred))
  	    cat(exp_name, "\nTraining score:", score, "(", model$bestInd, ")", "\n")
  	    test1 <- test.full[,feature.names]
  	    pred1 <- predict(model, data.matrix(test1), ntreelimit = model$bestInd)
  	    test.full[[id_name]] = as.character(as.numeric(test.full[[id_name]]))
  	    submission <- data.frame(QuoteNumber=test.full$QuoteNumber, QuoteConversion_Flag=pred1)
  	    
  	    # cat("saving the submission file\n")
  	    write_csv(submission, paste0("output/", exp_name, ".csv"))
  	    #     saveRDS(train.on.t, paste0("calibration/train_", exp_name, ".rds"))
  	    #     saveRDS(train.on.v, paste0("calibration/val_", exp_name, ".rds"))
  	    save(feature.names, file = paste0("models/feature_names_", exp_name))
  	    xgb.save(model, paste0("models/", exp_name, ".xgb"))
  	    cat("Training Done.", exp_name, "\n")
  	    gc()
  	    
  	    # cat("Time: ", Sys.time() - time_before_training, "\n")
  	  } else if (model_name == 'nn') {
  	    nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
  	    maxSize <- max(nnetGrid$.size)
  	    fs = feature.names
  	    fs = c("PropertyField37", "SalesField5", "PersonalField9", "PropertyField29", "SalesField4", "PersonalField2")
  	    numWts <- 1*(maxSize * (length(fs) + 1) + maxSize + 1)
  	    train.full$QuoteConversion_Flag = train.backup$QuoteConversion_Flag
  	    train.full$QuoteConversion_Flag[train.full$QuoteConversion_Flag == 0] = "no"
  	    train.full$QuoteConversion_Flag[train.full$QuoteConversion_Flag == 1] = "yes"
  	    
  	    nnetFit = train(x = train.full[, fs],
                        y = factor(train.full$QuoteConversion_Flag, levels = c("no", "yes")),
                        method = "nnet",
                        metric = "AUC",
                        tuneGrid = nnetGrid[1:1, ],
                        trace = T,
                        maxit = 10,
                        MaxNWts = numWts,
  	                    trControl = trainControl(classProbs = T, allowParallel = T, summaryFunction = twoClassSummary))
  	    nnetFit
  	    
  	    d = cbind(train.full[, fs], class.ind(train.full$QuoteConversion_Flag))
  	    form = as.formula(paste("yes + no ~", paste(fs, collapse = " + ")))
  	    nnetMod = neuralnet(data = d,
  	                        form, 
  	                        hidden = c(3, 4, 5),
  	                        act.fct = "logistic")
  	    
  	    scores <- compute(nnetMod, d[, 1:5])$net.result
  	    as.numeric(auc(train.full$QuoteConversion_Flag, scores[, 1]))
  	    
  	    res = predict(nnetMod, newdata = test.full)
  	    head(res)
  	  }
    }
  })
}

# try both validation set
# choose one
# layer size(s)
