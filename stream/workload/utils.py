from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from networkx import DiGraph

if TYPE_CHECKING:
    from stream.workload.computation.computation_node import ComputationNode
    from stream.workload.onnx_workload import ComputationNodeWorkload


def prune_workload(g: DiGraph, keep_types=None):
    """Return a pruned workload graph with only nodes of type in 'keep_types'."""
    if keep_types is None:
        keep_types = []
    while any(any(not isinstance(node, keep_type) for keep_type in keep_types) for node in g.nodes()):
        g_copy = g.copy()
        for node in g.nodes():
            if any(not isinstance(node, keep_type) for keep_type in keep_types):
                assert g.is_directed()
                in_edges_containing_node = list(g_copy.in_edges(node))  # type: ignore
                out_edges_containing_node = list(g_copy.out_edges(node))  # type: ignore
                for in_src, _ in in_edges_containing_node:
                    for _, out_dst in out_edges_containing_node:
                        g_copy.add_edge(in_src, out_dst)
                g_copy.remove_node(node)
                break
        g = g_copy  # type: ignore
    return g


def get_real_predecessors(node: "ComputationNode", g: "ComputationNodeWorkload"):
    return list(n for n in g.predecessors(node) if n.id != node.id)


def get_real_successors(node: "ComputationNode", g: "ComputationNodeWorkload"):
    return list(n for n in g.successors(node) if n.id != node.id)


def get_real_in_edges(node: "ComputationNode", g: "ComputationNodeWorkload"):
    return list(e for e in g.in_edges(node, data=True) if e[0].id != node.id)


def get_real_out_edges(node: "ComputationNode", g: "ComputationNodeWorkload"):
    return list(e for e in g.out_edges(node, data=True) if e[1].id != node.id)


def visualize_computation_workload(
    workload: "ComputationNodeWorkload",
    filepath: str = "computation_workload.png",
    *,
    cluster_by: str | None = "core_allocation",
    node_label_fn: Callable[["ComputationNode"], str] | None = None,
    edge_label_key: str | None = None,
) -> None:
    """Render a ``ComputationNodeWorkload`` to an image using Graphviz.

    Parameters
    ----------
    workload:
        The workload graph to visualize.
    filepath:
        Output path. The suffix determines the format (``.png``, ``.pdf``, ``.svg``).
    cluster_by:
        Optional attribute name on ``ComputationNode`` used to group nodes in dashed clusters
        (defaults to ``core_allocation``). Pass ``None`` to disable clustering.
    node_label_fn:
        Optional callable to build custom node labels. When omitted, a compact label including
        the node's short name, id, sub-id, and group is used.
    edge_label_key:
        Optional key to pull from edge attribute dictionaries and use as edge labels.

    Raises
    ------
    RuntimeError
        If the optional ``pydot``/Graphviz dependencies required for visualization are missing.
    """

    try:
        from networkx.drawing.nx_pydot import to_pydot  # type: ignore
        import pydot
    except ImportError as exc:  # pragma: no cover - convenience guard
        raise RuntimeError(
            "Visualizing workloads requires 'pydot' and the Graphviz binaries to be installed."
        ) from exc

    if workload.number_of_nodes() == 0:
        raise ValueError("Cannot visualize an empty ComputationNodeWorkload.")

    dot = to_pydot(workload)
    dot.set_rankdir("LR")
    dot.set_concentrate(True)

    def default_label(node: "ComputationNode") -> str:
        base_name = getattr(node, "short_name", getattr(node, "name", str(node)))
        details = [f"id={node.id}"]
        if getattr(node, "sub_id", None) not in (None, -1):
            details.append(f"sub={node.sub_id}")
        if getattr(node, "group", None) is not None:
            details.append(f"group={node.group}")
        allocation = getattr(node, "core_allocation", None)
        if allocation not in (None, []):
            formatted_allocation = _format_cluster_value(allocation)
            if formatted_allocation not in {"unassigned", "empty"}:
                details.append(f"core={formatted_allocation}")
        return f"{base_name}\\n" + ", ".join(details)

    for node in workload.nodes():
        dot_nodes = dot.get_node(str(node))
        if not dot_nodes:
            continue
        dot_node = dot_nodes[0]
        label = node_label_fn(node) if node_label_fn else default_label(node)
        dot_node.set_label(label)
        dot_node.set_shape("box")
        dot_node.set_style("filled")
        dot_node.set_fillcolor(_color_for_node(node))
        dot_node.set_fontname("Helvetica")

    if cluster_by is not None:
        clusters: dict[str, list["ComputationNode"]] = {}
        for node in workload.nodes():
            raw_value = getattr(node, cluster_by, None)
            cluster_label = _format_cluster_value(raw_value)
            clusters.setdefault(cluster_label, []).append(node)

        for idx, (cluster_label, nodes) in enumerate(sorted(clusters.items(), key=lambda item: item[0])):
            cluster = pydot.Cluster(
                graph_name=f"cluster_{idx}",
                label=f"{cluster_by}: {cluster_label}" if cluster_by else cluster_label,
                style="dashed",
            )
            for node in nodes:
                dot_nodes = dot.get_node(str(node))
                if dot_nodes:
                    cluster.add_node(dot_nodes[0])
            dot.add_subgraph(cluster)

    if edge_label_key is not None:
        for src, dst, data in workload.edges(data=True):
            label_value = data.get(edge_label_key)
            if label_value is None:
                continue
            dot_edges = dot.get_edge(str(src), str(dst))
            if dot_edges:
                dot_edges[0].set_label(str(label_value))
                dot_edges[0].set_fontname("Helvetica")

    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".png":
        dot.write_png(str(output_path))
    elif suffix == ".pdf":
        dot.write_pdf(str(output_path))
    elif suffix == ".svg":
        dot.write_svg(str(output_path))
    else:
        # Default to PNG when no recognized suffix is provided.
        dot.write(str(output_path), format=suffix[1:] if suffix else "png")


def _format_cluster_value(value: object) -> str:
    if value is None:
        return "unassigned"
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        try:
            items = list(value)
        except TypeError:  # Guard against iterables that cannot be re-iterated
            items = []
        if items:
            return ", ".join(str(v) for v in items)
        return "empty"
    return str(value)


def _color_for_node(node: "ComputationNode") -> str:
    palette = {
        "computation": "#a2d5f2",
        "input": "#c7ecee",
        "output": "#ffcb9a",
        "weight": "#f6e58d",
        "activation": "#c2f0c2",
        "buffer": "#d3b5e5",
    }
    node_type = getattr(node, "type", "").lower()
    return palette.get(node_type, "#eeeeee")
