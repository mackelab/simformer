from string import digits
from IPython.display import display, SVG  # type: ignore
import io
import matplotlib.pyplot as plt
import networkx as nx


import jax.numpy as jnp
from jax.experimental.pjit import pjit_p

from jax.core import Literal, Jaxpr
import re
from typing import Callable, Any, Union, Tuple, Optional, Sequence, Dict
from probjax.core.custom_primitives.random_variable import rv_p


COMPUTE_GRAPH_NODE_STYLES = {
    "const": dict(
        style="",
        color="goldenrod1",
        shape="circle",
        fontsize=9,
        margin=0.01,
        width=0.2,
        height=0.2,
        regular=True,
    ),
    "invar": dict(
        fillcolor="mediumspringgreen",
        style="filled",
        shape="circle",
        fontsize=9,
        width=0.2,
        height=0.2,
        margin=0.01,
        regular=True,
    ),
    "outvar": dict(
        style="filled",
        fillcolor="indianred1",
        color="black",
        shape="circle",
        width=0.2,
        height=0.2,
        fontsize=9,
        margin=0.01,
        regular=True,
    ),
    "operation": dict(
        shape="box",
        color="black",
        fontsize=9,
        regular=True,
        margin=0.01,
        width=0.2,
        height=0.2,
    ),
    "intermediate": dict(
        style="filled",
        color="lightgrey",
        shape="circle",
        width=0.2,
        height=0.2,
        fontsize=8,
        margin=0.01,
        regular=True,
    ),
    "random_variable": dict(
        style="filled",
        color="cornflowerblue",
        shape="circle",
        width=0.2,
        height=0.2,
        fontsize=8,
        margin=0.01,
        regular=True,
    ),
}


def to_networkx(
    jaxpr: Jaxpr,
    var_name_fn: Callable,
    compute_name_fn: Callable,
    level=0,
    maxlevel: int = jnp.inf,
) -> nx.DiGraph:
    """Converts a Jaxpr to a networkx graph.

    Args:
        jaxpr (Jaxpr): Jaxpr object.
        var_name_fn (Callable): Function to name the variables.
        compute_name_fn (Callable): Function to name the compute nodes.

    Returns:
        nx.DiGraph: Graph object.
    """
    graph = nx.DiGraph()
    constvars = jaxpr.constvars
    invars = jaxpr.invars
    outvars = jaxpr.outvars
    eqns = jaxpr.eqns

    # Adding constants
    for n in constvars:
        graph.add_node(
            var_name_fn(n, level=level),
            tag="const",
            val=n,
        )

    # Adding invars

    for n in invars:
        graph.add_node(
            var_name_fn(n, level=level),
            tag="invar",
            val=n,
            level=level,
        )

    # Adding outvars
    for n in outvars:
        graph.add_node(
            var_name_fn(n, level=level),
            tag="outvar",
            val=n,
            level=level,
        )

    scopes = 0

    # Adding equations
    for i, eqn in enumerate(eqns):
        # Add function node
        if level < maxlevel and eqn.primitive is pjit_p:
            scopes += 1
            # Different naming scope
            invars = eqn.invars
            outvars = eqn.outvars

            for n in outvars:
                name = var_name_fn(n, level=level)

                if name not in graph.nodes or graph.nodes[name] == {}:
                    if isinstance(n, Literal):
                        val = n.val
                        graph.add_node(name, tag="const", val=val, level=level)
                    else:
                        graph.add_node(
                            name,
                            tag="intermediate",
                            level=level,
                            val=n,
                        )

            sub_jaxpr = eqn.params["jaxpr"].jaxpr

            sub_graph = to_networkx(
                sub_jaxpr, var_name_fn, compute_name_fn, level=level + scopes
            )

            # Connect to outer scope vars
            invar_names = [
                node
                for node, data in sub_graph.nodes(data=True)
                if data.get("tag") == "invar"
            ]
            outvar_names = [
                node
                for node, data in sub_graph.nodes(data=True)
                if data.get("tag") == "outvar"
            ]

            outer_invar_names = [var_name_fn(n, level=level) for n in invars]
            outer_outvar_names = [var_name_fn(n, level=level) for n in outvars]

            rename_dict = dict(
                list(zip(invar_names, outer_invar_names))
                + list(zip(outvar_names, outer_outvar_names))
            )
            sub_graph = nx.relabel_nodes(sub_graph, rename_dict)

            graph = nx.compose(sub_graph, graph)
            continue

        graph.add_node(
            eqn_name_fn(i, level=level),
            tag="operation",
            index=i,
            level=level,
            xlabel=compute_name_fn(eqn),
        )

        # Add invars
        in_vars = eqn.invars
        for n in in_vars:
            name = var_name_fn(n, level=level)

            if name not in graph.nodes or graph.nodes[name] == {}:
                if isinstance(n, Literal):
                    val = n.val
                    graph.add_node(name, tag="const", val=val, level=level)
                else:
                    graph.add_node(
                        name,
                        tag="intermediate",
                        level=level,
                        val=n,
                    )

            graph.add_edge(name, eqn_name_fn(i, level=level), level=level)

        # Add outvars edges
        out_vars = eqn.outvars
        for n in out_vars:
            name = var_name_fn(n, level=level)

            if name not in graph.nodes or graph.nodes[name] == {}:
                if isinstance(n, Literal):
                    val = n.val
                    graph.add_node(name, tag="const", val=val, level=level)
                else:
                    graph.add_node(
                        name,
                        tag="intermediate",
                        level=level,
                        val=n,
                    )

            graph.add_edge(
                eqn_name_fn(i, level=level), var_name_fn(n, level=level), level=level
            )
            if eqn.primitive is rv_p:
                graph.nodes[var_name_fn(n, level=level)]["tag"] = "random_variable"

    return graph


def moralize_dag(dag: nx.DiGraph) -> nx.Graph:
    # Moralize DAG
    moral_graph = dag.to_undirected()
    for node in dag.nodes():
        parents = list(dag.predecessors(node))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                moral_graph.add_edge(parents[i], parents[j])
    return moral_graph


def subgraph(
    graph: nx.DiGraph | nx.Graph, nodes: Sequence[str]
) -> nx.DiGraph | nx.Graph:
    # Edge preserving subgraph, with subnodes.
    subgraph = graph.__class__()
    for node in nodes:
        subgraph.add_node(node, **graph.nodes[node])
        for node2 in nodes:
            if node2 != node and nx.has_path(graph, node, node2):
                subgraph.add_edge(node, node2)
    return subgraph


def eqn_name_fn(i: int, level=0) -> str:
    if level == 0:
        return f"f{i}"
    else:
        return f"f{i}_{level}"


def var_name_fn(n: str, level=0) -> str:
    if isinstance(n, Literal):
        name = str(n)[:3]
    else:
        name = str(n)

    if level == 0:
        return name
    else:
        return name + str(level)


class JaxprGraph:
    def __init__(
        self,
        jaxpr: Jaxpr,
        maxlevel: int = jnp.inf,
        graph: nx.DiGraph | None = None,
    ) -> None:
        self._jaxpr = jaxpr
        if graph is None:
            self._graph = to_networkx(
                jaxpr, var_name_fn, lambda x: str(x.primitive.name), maxlevel=maxlevel
            )

    @property
    def eqns(self):
        eqn_names = [f"f{i}" for i in range(len(self._jaxpr.eqns))]
        return dict(zip(eqn_names, self._jaxpr.eqns))

    @property
    def vars(self):
        var_names = [n for n in self._graph.nodes if not re.match(r"f\d+", n)]
        vars = [self._graph.nodes[n] for n in var_names]
        return dict(zip(var_names, vars))

    def __repr__(self):
        AGraph = nx.nx_agraph.to_agraph(self._graph)
        # Node styles by tag
        nodes = AGraph.nodes()
        max_level = 0
        for n in nodes:
            attributes = dict(n.attr)
            level = attributes.get("level", "0")
            if int(level) > max_level:
                max_level = int(level)
            n.attr.update(
                COMPUTE_GRAPH_NODE_STYLES[attributes.get("tag", "intermediate")]
            )
        # Cluster by "level"
        for i in range(max_level + 1):
            AGraph.add_subgraph(
                [n for n in nodes if n.attr["level"] == str(i)], name=f"cluster_{i}"
            )

        # Left to right in topological order
        AGraph.graph_attr["rankdir"] = "LR"
        AGraph.layout("dot")

        # Render for jupyter
        svg = io.BytesIO()
        AGraph.draw(svg, format="svg")
        svg.seek(0)
        display(SVG(svg.read()))
        return ""


class DirectedVariableGraph(JaxprGraph):
    def __init__(self, jaxpr, graph=None) -> None:
        super(DirectedVariableGraph, self).__init__(jaxpr, graph)
        self._graph = nx.bipartite.projected_graph(self._graph, list(self.vars.keys()))

    def eqns(self):
        return self._jaxpr.eqns


class UndirectedVariableGraph(DirectedVariableGraph):
    def __init__(self, jaxpr, graph=None) -> None:
        super(UndirectedVariableGraph, self).__init__(jaxpr, graph)
        self._graph = moralize_dag(self._graph.to_directed())


class DirectedGraphicalModel(DirectedVariableGraph):
    def __init__(self, jaxpr, graph=None) -> None:
        super(DirectedGraphicalModel, self).__init__(jaxpr, graph)
        random_vars = []
        for n, tag in nx.get_node_attributes(self._graph, "tag").items():
            if tag == "random_variable" or tag == "invar" or tag == "outvar":
                random_vars.append(n)
        self._graph = subgraph(self._graph, random_vars)


class UndirectedGraphicalModel(UndirectedVariableGraph):
    def __init__(self, jaxpr, graph=None) -> None:
        super(UndirectedGraphicalModel, self).__init__(jaxpr, graph)
        random_vars = []
        for n, tag in nx.get_node_attributes(self._graph, "tag").items():
            if tag == "random_variable" or tag == "invar" or tag == "outvar":
                random_vars.append(n)
        self._graph = subgraph(self._graph, random_vars)
