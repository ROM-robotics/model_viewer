"""Generate a simplified Mermaid flowchart from a parsed layer sequence.

Consecutive layers that share the same broad category (conv, norm,
activation, recurrent, attention, ...) are collapsed into a single block
node so the diagram stays readable even for models with hundreds of ops,
while still conveying the Input -> stage -> ... -> Output flow.
"""
from .common import CATEGORY_LABELS, CATEGORY_SHAPES, CATEGORY_COLORS


def _escape(text: str) -> str:
    return text.replace('"', "#quot;").replace("\n", " ")


def _group_blocks(layers):
    blocks = []
    cur_cat, cur_types, cur_count = None, [], 0
    for layer in layers:
        if layer.category in ("input", "output"):
            continue
        if layer.category == cur_cat:
            cur_count += 1
            if layer.type not in cur_types:
                cur_types.append(layer.type)
        else:
            if cur_cat is not None:
                blocks.append((cur_cat, cur_types, cur_count))
            cur_cat, cur_types, cur_count = layer.category, [layer.type], 1
    if cur_cat is not None:
        blocks.append((cur_cat, cur_types, cur_count))
    return blocks


def generate_mermaid(layers) -> str:
    blocks = _group_blocks(layers)

    lines = ["flowchart TD"]
    lines.append('Input(["Input"])')
    prev = "Input"
    class_lines = []

    for i, (category, types, count) in enumerate(blocks):
        node_id = f"B{i}"
        type_list = ", ".join(types[:4]) + ("..." if len(types) > 4 else "")
        label = f"{CATEGORY_LABELS.get(category, category.title())}<br/>({type_list})<br/>x{count}"
        open_c, close_c = CATEGORY_SHAPES.get(category, ("[", "]"))
        lines.append(f'{node_id}{open_c}"{_escape(label)}"{close_c}')
        lines.append(f"{prev} --> {node_id}")
        class_lines.append(f"class {node_id} cat_{category}")
        prev = node_id

    lines.append('Output(["Output"])')
    lines.append(f"{prev} --> Output")

    used_categories = {category for category, _, _ in blocks}
    for category in used_categories:
        color = CATEGORY_COLORS.get(category, "#eeeeee")
        lines.append(f"classDef cat_{category} fill:{color},stroke:#333,stroke-width:1px;")
    lines.extend(class_lines)

    return "\n".join(lines)
