dm<-read.delim("men.txt")
dw<-read.delim("women.txt")
dm<-dm[,-c(1,2,16,17,18,19,20,21,34,35,36,37,38,39)]
dw<-dw[,-c(1,2,16,17,18,19,20,21,34,35,36,37,38,39)]

colMeans(dm[dm$Result==0,-1])
colMeans(dm[dm$Result==1,-1])

apply(dm[dm$Result==0,],2,var)
apply(dm[dm$Result==1,],2,var)

t.test(dm$ACE.1[dm$Result==0],dm$ACE.1[dm$Result==1])

cor(dm)[1,-1]

dm.lm<-lm(Result~.,dm)
summary(dm.lm)

cor(dm$Result,predict(dm.lm,newdata=dm[,-1]))
cor(dw$Result,predict(dm.lm,newdata=dw[,-1]))

dw.lm<-lm(Result~.,dw)
cor(dw$Result,predict(dw.lm,newdata=dw[,-1]))

idx<-sample(491,6,replace=F)
predict(dm.lm,newdata=dm[idx,-1])

dm$Result<-as.factor(dm$Result)
dw$Result<-as.factor(dw$Result)

dm.glm<-glm(Result~.,dm,family=binomial)
summary(dm.glm)

table(dw$Result,round(predict(dm.glm,newdata=dw[,-1],type='response'),0))

library(e1071)
dm.tune<-tune.svm(Result~.,data=dm)
dm.tune$best.model

dm.svm<-svm(Result~.,dm,cost=1,gamma=dm.tune$best.model$gamma)
table(dw$Result,predict(dm.svm,newdata=dw[,-1]))

sum(diag(table(dw$Result,predict(dm.svm,newdata=dw[,-1]))))/nrow(dw)

library(randomForest)
tuneRF(dm[,-1],dm[,1],doBest=T)

dm.rf<-randomForest(Result~.,dm,mtry=16)
importance(dm.rf)

table(dw$Result,predict(dm.rf,newdata=dw[,-1]))

library(MASS)
dm.glm.step<-stepAIC(dm.glm)

summary(dm.glm.step)

table(dw$Result,round(predict(dm.glm.step,newdata=dw[,-1],type='response'),0))

sum(diag(table(dw$Result,round(predict(dm.glm.step,newdata=dw[,-1],type='response'),0))))/nrow(dw)

dm.pc<-princomp(scale(dm[,-1]))
summary(dm.pc)
biplot(dm.pc)

dm2<-data.frame(Result=dm$Result,FSP.1=dm$FSP.1, FSP.2=dm$FSP.2, WNR.1=dm$WNR.1, NPW.1=dm$NPW.1, NPW.2=dm$NPW.2, FSW.2=dm$FSW.2, ACE.2=dm$ACE.2, SSW.1=dm$SSW.1, DBF.1=dm$DBF.1, BPW.1=dm$BPW.1, SSP.1=dm$SSP.1, SSP.2=dm$SSP.2)
dm2$Result<-as.factor(dm2$Result)
tuneRF(dm2[,-1],dm2[,1],doBest=T)

dm2.rf<-randomForest(Result~.,dm2,mtry=3)
dw2<-data.frame(Result=dw$Result,FSP.1=dw$FSP.1, FSP.2=dw$FSP.2, WNR.1=dw$WNR.1, NPW.1=dw$NPW.1, NPW.2=dw$NPW.2, FSW.2=dw$FSW.2, ACE.2=dw$ACE.2, SSW.1=dw$SSW.1, DBF.1=dw$DBF.1, BPW.1=dw$BPW.1, SSP.1=dw$SSP.1, SSP.2=dw$SSP.2)
table(dw2$Result,predict(dm2.rf,newdata=dw2[,-1]))

sum(diag(table(dw2$Result,predict(dm2.rf,newdata=dw2[,-1]))))/nrow(dw2)

dm3<-data.frame(Result=dm$Result,FSP.1=dm$FSP.1, FSW.1=dm$FSW.1, SSW.1=dm$SSW.1, DBF.1=dm$DBF.1, WNR.1=dm$WNR.1, BPC.1=dm$BPC.1, BPW.1=dm$BPW.1, FSP.2=dm$FSP.2, FSW.2=dm$FSW.2, SSW.2=dm$SSW.2, DBF.2=dm$DBF.2, UFE.2=dm$UFE.2, BPC.2=dm$BPC.2, BPW.2=dm$BPW.2, NPW.2=dm$NPW.2)
dw3<-data.frame(Result=dw$Result,FSP.1=dw$FSP.1, FSW.1=dw$FSW.1, SSW.1=dw$SSW.1, DBF.1=dw$DBF.1, WNR.1=dw$WNR.1, BPC.1=dw$BPC.1, BPW.1=dw$BPW.1, FSP.2=dw$FSP.2, FSW.2=dw$FSW.2, SSW.2=dw$SSW.2, DBF.2=dw$DBF.2, UFE.2=dw$UFE.2, BPC.2=dw$BPC.2, BPW.2=dw$BPW.2, NPW.2=dw$NPW.2)
dm3$Result<-as.factor(dm3$Result)
dw3$Result<-as.factor(dw3$Result)

dm3.tune<-tune.svm(Result~.,data=dm3)
dm3.tune$best.model

dm3.svm<-svm(Result~.,dm3,cost=1,gamma=dm3.tune$best.model$gamma)
table(dw3$Result,predict(dm3.svm,dw3[,-1]))

sum(diag(table(dw3$Result,predict(dm3.svm,dw3[,-1]))))/nrow(dw3)

tuneRF(dm3[,-1],dm[,1],doBest=T)

dm3.rf<-randomForest(Result~.,dm3,mtry=6)
table(dw3$Result,predict(dm3.rf,dw3[,-1]))

sum(diag(table(dw3$Result,predict(dm3.rf,dw3[,-1]))))/nrow(dw3)