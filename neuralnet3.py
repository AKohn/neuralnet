from __future__ import division
import math, random, numpy

class Net:
    def activate(self,x):
        try:
            r=[]
            for i in x:
                a=numpy.sign(i)*10 if abs(i)>10 else i
                r.append(1/(1+numpy.exp(-a)))
            return numpy.array(r)
        except TypeError:
            return 1/(1+numpy.exp(-max(min(x,10),-10)))
    def __init__(self,nnodes=None,data=None):
        numpy.random.seed(24)
        if data:
            self.nnodes=data["l"]
            self.nlayers=len(self.nnodes)
            self.W=data["w"]
        else:
            self.nlayers=len(nnodes)
            self.nnodes=nnodes
            self.W=[numpy.array([[random.uniform(-1,1) for i in range(nnodes[k])] for j in range(nnodes[k+1])]) for k in range(self.nlayers-1)]
        print [i.shape for i in self.W]
    def run(self,in_):
        result=in_
        for i in range(self.nlayers-2):
            result=self.W[i].dot(result)
            result=self.activate(result)
        return self.W[-1].dot(result)
    """def net(self,x):
        result=[[x],[x]]
        for i in range(self.nlayers-1):
            n=self.W[i].dot(result[1][-1])
            result[0].append(n)
            result[1].append(n)
            result[1][-1]=self.activate(result[1][-1])
        return result"""
    def net(self,x):
        result=[x]
        for i in range(self.nlayers-2):
            n=self.W[i].dot(result[-1])
            result.append(n)
            result[-1]=self.activate(result[-1])
        result.append(self.W[-1].dot(result[-1]))
        return numpy.array(result)
    def delta(self,layer,t,o_):
        q=o_[layer]
        if layer==self.nlayers-1:
            return [(q-t)]
        else:
            d=self.delta(layer+1,t,o_)
            print self.W[layer].T.shape
            print d[-1].shape
            print q
            print "-"
            return [self.W[layer].T.dot(d[-1])*q*(1-q)]+d
    def backpropagate(self,x,t,rate=1.):
        o=self.net(x)
        #print numpy.outer(self.delta(0,t,o),o)
        delta_=self.delta(0,t,o)
        d=numpy.array([numpy.outer(delta_[layer],o[layer-1])*-rate for layer in range(1,self.nlayers)])
        for i in range(self.nlayers-1):
            self.W[i]+=d[i]
            print i
    def partial(self,x,t,layer,i,j):
        e1=self.error(x,t)
        self.W[layer][i,j]+=0.004
        e2=self.error(x,t)
        self.W[layer][i,j]-=0.004
        return (e2-e1)/0.004
    def error(self,x,t):
        y=numpy.array([self.run(i) for i in x])
        return (1/2)*((y-t)**2).sum()
    def math(self,x,t):
        print (-(t-self.run(x))*self.activate(self.W[0][0][0]*x),-x*self.W[1][0][0]*(t-self.run(x))*self.activate(self.W[0][0][0]*x)*(1-self.activate(self.W[0][0][0]*x)))
