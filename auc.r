roc.auc <- function(model, df) {
  pred_col = predict(model, data.matrix(df))
  pred = prediction(pred_col, df$QuoteConversion_Flag)
  perf = performance(pred, "tpr", "fpr")
  auc <- performance(pred, "auc")
  auc <- unlist(slot(auc, "y.values"))
  par(mar=c(5,5,2,2),xaxs = "i",yaxs = "i",cex.axis=1.3,cex.lab=1.4)
  plot(perf,col="black",lty=3, lwd=3)
  auc
}
roc.auc(clf, train[h, ])

auc.prob <- function(response, predictor) {
  pscores = predictor[which(response == 1)]
  nscores = predictor[-pscores]
  mean(sample(pscores) > sample(nscores))
}
auc.prob(train[h, ]$QuoteConversion_Flag, val_pred)
