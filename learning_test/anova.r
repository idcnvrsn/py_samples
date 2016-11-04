x<-read.table("anova1way.txt",header=T,sep=",")

F<-x$group     # F: 調理に使った脂肪質の種類
y<-x$obs       # y: 測定値

s <- aov(y ~ F)
anova(s)
summary(aov(y ~ F))
