import pygsp
from spektral.data import Graph


def get_cloud(name, **kwargs):
    graph_class = getattr(pygsp.graphs, name)
    graph = graph_class(**kwargs)

    y = graph.coords
    a = graph.W.astype("f4")

    output = Graph(x=y, a=a)
    output.name = name

    return output
