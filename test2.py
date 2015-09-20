import neuralnet3, pygame, sys, cPickle, numpy, cProfile, matplotlib.pyplot as plt


INFILE=None#"weights1"
OUTFILE="weights1"

to_do=100
if INFILE:
    print "loading weights..."
    f=open(INFILE,"rb")
    data=cPickle.load(f)
    f.close()
    data_done=data["data_done"]
else:
    data_done=0
print "initializing net..."
if INFILE:
    net=neuralnet3.Net(data=data)
else:
    net=neuralnet3.Net([3072,4000,10])
print "loading data..."
with open("cifar-10-batches-py/data_batch_1","rb") as f:
    d=cPickle.load(f)
    data=d["data"]
    l=d["labels"]
with open("cifar-10-batches-py/batches.meta","rb") as f:
    label_names=cPickle.load(f)["label_names"]
print label_names
err=[]
err_x=[]
def r():
    print "adjusting for images "+str(data_done+1)+" to "+str(data_done+to_do)+"..."
    for i in range(data_done,data_done+to_do):
        net.backpropagate(data[i],numpy.array([0 if j!=l[i] else 1 for j in range(10)]))
        err_x.append(i)
        err.append(net.error(data[i],numpy.array([0 if j!=l[i] else 1 for j in range(10)])))
        print "finished "+str(i-data_done+1)+"/"+str(to_do)
print "training..."
r()
#cProfile.run("r()")
plt.plot(err_x,err)
plt.show()
print err
1/0
if OUTFILE:
    print "saving data..."
    f=open(OUTFILE,"wb")
    cPickle.dump({"w":net.W,"l":net.nnodes,"data_done":data_done+to_do},f)
    f.close()
print "initializing pygame..."
def getsurf(d):
    r=d[:1024].reshape((32,32))
    g=d[1024:2048].reshape((32,32))
    b=d[2048:].reshape((32,32))
    return pygame.surfarray.make_surface(numpy.array([r,g,b]).transpose((2,1,0)))
pygame.init()
print "loading test data..."
test_data=cPickle.load(open("cifar-10-batches-py/test_batch","rb"))
print "testing..."
correct=0
total=0
for i in range(1000):
    result=net.run(test_data["data"][i])
    max_index=0
    for j in range(len(label_names)):
        if result[j]==max(result):
            max_index=j
    correct+=max_index==test_data["labels"][i]
    total+=1
    if not i%100:print i
print str(correct)+"/"+str(total)+" tests successful"
"""print "opening window..."
screen=pygame.display.set_mode((32,32))
i=0
while True:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            sys.exit()"""
"""        if event.type==pygame.KEYDOWN:
            screen.blit(getsurf(test_data["data"][i]),(0,0))
            pygame.display.flip()
            result=net.run(test_data["data"][i])
            max_index=0
            for j in range(len(label_names)):
                print label_names[j]+": "+str(result[j])
                if result[j]==max(result):
                    max_index=j
            probably=label_names[max_index]
            actually=label_names[test_data["labels"][i]]
            print "probably: "+probably
            print "actually: "+actually
            correct+=probably==actually
            total+=1
            print "so far, "+str(correct)+"/"+str(total)+" correct"
            i+=1"""
