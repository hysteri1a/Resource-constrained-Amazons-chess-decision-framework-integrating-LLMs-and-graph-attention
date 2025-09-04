# ---------- canonical key helpers ----------
def _normalize_pos(p):
    if p is None:
        return None
    if isinstance(p, (list, tuple)):
        return (int(p[0]), int(p[1])) if len(p) >= 2 else tuple(p)
    if isinstance(p, int):
        return int(p)
    if hasattr(p, "x") and hasattr(p, "y"):
        return (int(p.x), int(p.y))
    return p


def canonical_key_of_feasible(feasible):
    """
    feasible format: [orig, dest, mode, val, prev_node]
    returns key: ((ox,oy),(dx,dy), mode_int)
    """
    try:
        orig = feasible[0] if len(feasible) > 0 else None
        dest = feasible[1] if len(feasible) > 1 else None
        mode = feasible[2] if len(feasible) > 2 else None
        return (_normalize_pos(orig), _normalize_pos(dest), int(mode) if mode is not None else None)
    except Exception:
        return None


def canonical_key_of_node(node):
    """
    针对你的 Node 类：用 node.parent.pos 作为 origin，node.pos 作为 dest。
    返回与 feasible 的 canonical key 一致的格式:
        ((ox,oy), (dx,dy), mode)
    """
    try:
        # 1) 优先用 parent.pos -> node.pos（通常 child 的 parent 非 None）
        parent = getattr(node, "parent", None)
        if parent is not None:
            orig = getattr(parent, "pos", None)
            dest = getattr(node, "pos", None)
            mode = getattr(node, "mode", None)
            if orig is not None and dest is not None:
                return (_normalize_pos(orig), _normalize_pos(dest), int(mode) if mode is not None else None)

    except Exception:
        pass
    return None


# ---------- build mapper ----------
def build_node_to_index_from_candidates(all_candidates_next):
    """
    all_candidates_next: list of feasible elements (same format as your feasible list)
    Returns: mapper(q) and action_map (dict)
    """
    try:
        action_map = {}
        # primary: map by canonical key
        for idx, cand in enumerate(all_candidates_next):
            key = canonical_key_of_feasible(cand)
            if key is not None:
                # if duplicate keys exist, prefer first occurrence
                if key not in action_map:
                    action_map[key] = idx
            else:
                # fallback: identity mapping using id()
                action_map[id(cand)] = idx

        def mapper(q):
            # if q is exactly an element in list, fast path
            try:
                if q in all_candidates_next:
                    return all_candidates_next.index(q)
            except Exception:
                pass

            # try feasible-like
            kf = None
            if isinstance(q, (list, tuple)):
                kf = canonical_key_of_feasible(q)
            if kf is not None and kf in action_map:
                return action_map[kf]

            # try Node-like
            kn = None
            # avoid importing Node type; duck-typing:
            if hasattr(q, "step") or hasattr(q, "pos") or hasattr(q, "children"):
                kn = canonical_key_of_node(q)
            if kn is not None and kn in action_map:
                return action_map[kn]

            # fallback try matching by prev node relationship:
            try:
                for idx, cand in enumerate(all_candidates_next):
                    if isinstance(cand, (list, tuple)) and len(cand) > 4:
                        prev = cand[4]
                        if prev is q:  # exact object equality
                            return idx
                        # if q is a child node of prev, find that relationship
                        if hasattr(prev, "children") and q in getattr(prev, "children"):
                            return idx
            except Exception:
                pass

            # last resort: try id(q) lookup (if action_map stored ids)
            return action_map.get(id(q), None)

        return mapper, action_map

    except:
        import traceback
        traceback.print_exc()


