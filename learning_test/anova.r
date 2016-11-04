x<-read.table("anova1way.txt",header=T,sep=",")

F<-x$group     # F: 調理に使った脂肪質の種類
y<-x$obs       # y: 測定値

#P値が有意水準を下回っていることから，4水準の平均値は有意差が認められる。 
s <- aov(y ~ F)
anova(s)
summary(aov(y ~ F))
