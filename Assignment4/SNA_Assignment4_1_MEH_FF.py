import igraph
from igraph import *
from random import randint
from random import random
def pref_attach_graph(G,t,m):
    if (m>len(G.vs)):
        print "Damn shit you did something wrong!"
        return
    g = G
    id = 0
    for n in xrange(t):
        g.add_vertices(1)
        id = len(g.vs)-1
        while(g.degree(id) != m):
            randomID =randint(0,id-1)
            randomProp = (float(g.degree(randomID))/sum(g.degree()))
            print "Random ID"
            print randomID
            print "Random Prop"
            print randomProp
            x = random()
            print "Random Number"
            print x
            if (x < randomProp):
                g.add_edge(id,randomID)
    print "There are %i vertices in this damn Graph!" % (len(g.vs))
    return g


if __name__ == "__main__": 
    g = Graph()
    g.add_vertices(5)
    g.add_edges([(0,1),(1,2),(2,3),(3,4)])
    pref_attach_graph(g,100,3) 
