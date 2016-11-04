x<-read.table("anova1way.txt",header=T,sep=",")

F<-x$group     # F: ’²—‚ÉŽg‚Á‚½Ž‰–bŽ¿‚ÌŽí—Þ
y<-x$obs       # y: ‘ª’è’l

s <- aov(y ~ F)
anova(s)
summary(aov(y ~ F))
