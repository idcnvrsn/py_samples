a1<-c(63,58,64,58,77,66,52,64,49,66)
a2<-c(64,64,68,61,56,71,64,65,85,75)
a3<-c(59,87,79,71,65,65,65,71,74,58)
a4<-c(83,79,65,67,80,72,80,75,72,84)

#対応なし一元配置の分散分析
bunsan1<-data.frame(A=factor(c(rep("a1",10),rep("a2",10),rep("a3",10),rep("a4",10))),y=c(a1,a2,a3,a4))
bunsan1

boxplot(y~A,data=bunsan1,col="lightblue")

summary(aov(y~A,data=bunsan1))

#対応あり一元配置の分散分析
bunsan2<-data.frame(A= factor(c(rep("a1",10), rep("a2",10), rep("a3",10), rep("a4",10))),No= factor(rep(1:10, 4)),y=c(a1,a2,a3,a4))
bunsan2

summary(aov(y ~ A+No, bunsan2))

#二元分散分析
a1<-c(3,3,5,4,6,4,5,6,6,7,8,7)
a2<-c(3,5,2,4,3,4,5,3,3,4,3,2)

bunsan3<-data.frame(A=factor(c(rep("a1",12),rep("a2",12))),B=factor(rep(c(rep("b1",4), rep("b2",4), rep("b3",4)),2)),y= c(a1,a2))
bunsan3

#交互作用効果を無視
summary(aov(y~A+B,data=bunsan3))

#交互作用を考慮
summary(aov(y~A*B,data=bunsan3))

attach(bunsan3)
interaction.plot(A,B,y)
