x<-read.table("anova1way.txt",header=T,sep=",")

F<-x$group     # F: �����Ɏg�������b���̎��
y<-x$obs       # y: ����l

s <- aov(y ~ F)
anova(s)
summary(aov(y ~ F))
