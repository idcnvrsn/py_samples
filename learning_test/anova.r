x<-read.table("anova1way.txt",header=T,sep=",")

F<-x$group     # F: �����Ɏg�������b���̎��
y<-x$obs       # y: ����l

#P�l���L�Ӑ�����������Ă��邱�Ƃ���C4�����̕��ϒl�͗L�Ӎ����F�߂���B 
s <- aov(y ~ F)
anova(s)
summary(aov(y ~ F))
