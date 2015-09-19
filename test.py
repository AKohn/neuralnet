from __future__ import division
import neuralnet3, pygame, sys, math, numpy
from matplotlib import pyplot as plt

n=neuralnet3.Net([1,8,1])
screen=pygame.display.set_mode((500,500))
def conv(x,y):
    return (int(500*(x/12.+.5)),int(500*(.5-y/12.)))
def unconv(x,y):
    return (12*(x/500.-.5),12*(.5-y/500.))
def graph():
    screen.fill((255,255,255))
    i=-6.0
    while i<=6.0:
        pygame.draw.circle(screen,(0,255,0),conv(i,numpy.sin(i)),2)
        pygame.draw.circle(screen,(255,0,0),conv(i,math.cos(i)),2)
        pygame.draw.circle(screen,(0,0,255),conv(i,n.run([i])[0]),2)
        pygame.draw.circle(screen,(0,0,0),conv(i,n.run([i])[0]),2)
        i+=.1
    pygame.display.flip()
print "drawing..."
graph()
print "done!"
x=numpy.arange(-6,6,.05)
while True:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            sys.exit()
        """if event.type==pygame.MOUSEBUTTONDOWN:
            x,y=unconv(*event.pos)
            n.backpropogate([x],[math.sin(x),math.cos(x)])
            graph()"""
    for i in range(1):
        #print i
        numpy.random.shuffle(x)
        for j in x:
            #print "partial"
            #print n.partial([j],[numpy.sin(j)],0,0,0),n.partial([j],[numpy.sin(j)],1,0,0)
            n.backpropagate([j],[numpy.sin(j)],rate=.05)
            #print n.partial(x,[n.run(k) for k in x],0,0,0)
    graph()
"""err=[]
for i in range(10):
    print i
    numpy.random.shuffle(x)
    for j in x:
        n.backpropagate([j],[numpy.sin(j)],rate=.05)
    err.append([i,n.error(x,numpy.sin(x))])
x,y=zip(*err)
plt.plot(x,y,"o-")
plt.show()
while True:
    for e in pygame.event.get():
        if e.type==pygame.QUIT:sys.exit()"""
