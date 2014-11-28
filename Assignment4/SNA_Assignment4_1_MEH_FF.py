import igraph
from igraph import *
from random import randint
from random import random
import math

def generate_full_graph(num_vertices):
    g = Graph()
    g.add_vertices(num_vertices)
    for i in xrange(num_vertices):
        for n in xrange(i + 1, num_vertices):
            g.add_edge(i, n)
    return g


def pref_attach_graph(G, t, m):
    """
     Takes an initial undirected graph G and returns an updated graph with t new vertices where each vertex generates m
     new edges corresponding to the preferential attachment model

    :param G: initial undirected graph
    :param t: number of new vertices
    :param m: number of edges for newly created vertices
    :return:
    """
    if (m > len(G.vs)):
        print "Damn shit you did something wrong!\n" \
              "Choose m smaller than ", len(G.vs)
        exit(-1)

    g = G.copy()
    id = 0
    for n in xrange(t):
        g.add_vertices(1)
        id = len(g.vs) - 1
        while (g.degree(id) != m):
            randomID = randint(0, id - 1)
            randomProp = (float(g.degree(randomID)) / sum(g.degree()))
            # print "Random ID"
            # print randomID
            # print "Random Prop"
            # print randomProp
            x = random()
            # print "Random Number"
            # print x
            if (x < randomProp):
                g.add_edge(id, randomID)
    return g


if __name__ == "__main__":
    num_vertices = 5
    print "\nProblem 1a: "
    g = generate_full_graph(num_vertices)
    t = 100
    m = 3

    g1a = pref_attach_graph(g, t, m)

    # (i) the number of vertices
    print ("Number of vertices %i" % (len(g1a.vs)))
    # (ii) the number of edges
    print ("Number of edges %i" % (len(g1a.es)))
    #(iii) the sum of the vertex degrees
    print ("Sum of vertex degrees %i" % (sum(g1a.degree())))

    print "\nProblem 1b: "
    t = 1000
    m = 4

    g1b = pref_attach_graph(g, t, m)

    #(ai) the number of vertices
    V = len(g1b.vs)
    print ("Number of vertices %i" % V)
    #(aii) the number of edges
    E = len(g1b.es)
    print ("Number of edges %i" % E)
    #(aiii) the sum of the vertex degrees
    print ("Sum of vertex degrees %i" % (sum(g1b.degree())))

    #(i) cluster coefficient
    icc = g1b.transitivity_undirected()

    #(ii) the average path length of the generated graph
    iagl = g1b.average_path_length()

    #(iii) the analytic cluster coefficient
    #C=(m0-1)/8 * (ln(t))^2/t
    acc=(float(num_vertices-1)/8) * ((math.log(t)**2)/t)

    #(iv) the analytic average path length
    #l=ln(|V|)/ln(ln(|V|))
    aagl = math.log(V)/math.log(math.log(V))

    print ("Global Clustering Coefficient")
    print ("iGraph\t\tanalytical")
    print ("%f\t%f" % (icc, acc))

    print ("Average Path Length")
    print ("iGraph\t\tanalytical")
    print ("%f\t%f" % (iagl, aagl ))