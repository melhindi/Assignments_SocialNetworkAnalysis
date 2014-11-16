#!/usr/bin/env python

from igraph import *
import logging
import sys
import pylab

if __name__ == "__main__":
    if len(sys.argv) != 2:
        scriptname = os.path.basename(__file__)
        print >> sys.stderr, "Usage "+ str(scriptname) +" <file>"
        exit(-1)

    filename = sys.argv[1]

    logging.basicConfig(filename=filename+'.log', level=logging.INFO)
    
    g = Graph.Read_Ncol(filename, directed=False)
    logging.info(str(g.summary()))
    logging.info("Simplifying the graph ...")
    gu = g.simplify()
    gd = gu.as_directed()
    logging.info(str(gu.summary()))
    d = gu.diameter(directed=False, unconn=True)
    logging.info("Diameter: "+str(d))
    
    b = gu.edge_betweenness(directed=False)
    m = max(b)
    logging.info("Highest betweenness centrality: "+str(m))
    top_nodes = [i for i, j in enumerate(b) if j == m]
    logging.info("following nodes [ids] have the above centrality:\n"+str(top_nodes))

    t = gd.triad_census()["300"]
    logging.info("The number of triads is: "+str(t))
    cc = gu.transitivity_undirected(mode="nan")
    logging.info("Global Clustering Coefficient: "+str(cc))
    dd = g.degree_distribution()
    #logging.info(str(dd))
    #ddd = g.degree()
    #logging.info(str(ddd))
    xs, ys = zip(*[(left, count) for left, _, count in dd.bins()])
    result = power_law_fit(ys)
    logging.info(str(result))
    #pylab.bar(xs, ys)
    #pylab.show()
