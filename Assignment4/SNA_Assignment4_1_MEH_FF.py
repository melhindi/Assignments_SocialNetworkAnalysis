import igraph
from igraph import *
from random import randint
from random import random

def generate_full_graph(num_vertices):
    g = Graph()
    g.add_vertices(num_vertices)
    for i in xrange(num_vertices):
        for n in xrange(i+1,num_vertices):
            g.add_edge(i,n)
    return g

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
            # print "Random ID"
            # print randomID
            # print "Random Prop"
            # print randomProp
            x = random()
            # print "Random Number"
            # print x
            if (x < randomProp):
                g.add_edge(id,randomID)
    return g


if __name__ == "__main__":

    num_vertices = 5
    print "Problem 1a: "
    g = generate_full_graph(num_vertices)
    t = 100
    m = 3

    g1a = pref_attach_graph(g,t,m)

    #(i) the number of vertices
    print ("Number of vertices %i" % (len(g1a.vs)))
    #(ii) the number of edges
    print ("Number of edges %i" % (len(g1a.es)))
    #(iii) the sum of the vertex degrees
    print ("Sum of vertex degrees %i" % (sum(g1a.degree())))


    print "Problem 1b: "
    t = 1000
    m = 4

    g1b = pref_attach_graph(g,t,m)

     #(i) the number of vertices
    print ("Number of vertices %i" % (len(g1b.vs)))
    #(ii) the number of edges
    print ("Number of edges %i" % (len(g1b.es)))
    #(iii) the sum of the vertex degrees
    print ("Sum of vertex degrees %i" % (sum(g1b.degree())))

    #(i) cluster coefficient
    icc = g1b.transitivity_undirected()
    print ("Global Clustering Coefficient")
    print ("iGraph \t \t analytical")
    print ("%f \t %f" % (icc, 2.0 ))

    #(ii) the average path length of the generated graph
    iagl = g1b.average_path_length()
    print ("Average Path Length")
    print ("iGraph \t \t analytical")
    print ("%f \t %f" % (iagl, 3.0 ))


    #(iii) the analytic cluster coefficient

    #(iv) the analytic average path length
