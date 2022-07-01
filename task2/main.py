# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import graph


def check_excluded(candidates, excluded, gr):
    for v in excluded:
        if len(set(gr.linked_nodes[v]).intersection(
                candidates)) == 0:
            return False
    return True


def bronker_bosch(inv_clique, candidates, excluded, reporter, gr):
    # if not candidates and not excluded:
    if not candidates and check_excluded(candidates, excluded, gr):
        reporter.append(inv_clique)
        return

    for v in list(candidates):
        new_candidates = candidates - set(gr.linked_nodes[v]) - {v}
        new_excluded = excluded - set(gr.linked_nodes[v])
        bronker_bosch(inv_clique + [v], new_candidates, new_excluded, reporter, gr)
        candidates.remove(v)
        excluded.add(v)

def find_max_inv_clique(reporter):
    if len(reporter) == 0:
        return []
    max_len = 0
    res = []
    for inv_clique in reporter:
        if len(inv_clique) > max_len:
            res = inv_clique
            max_len = len(inv_clique)
    return res


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    with open('res.txt', 'a') as the_file:
        for i in range(1, 6):
            gr = graph.Graph()
            # построим граф например на i + 1 вершине
            gr.create_random_graph(i + 1, 0.5)

            inv_clique = []
            candidates = set(gr.linked_nodes.keys())
            excluded = set()
            reporter = []
            bronker_bosch(inv_clique, candidates, excluded, reporter, gr)
            max_inv_clique = find_max_inv_clique(reporter)
            gr.draw('graph_{}'.format(i))
            the_file.write(" ".join(max_inv_clique))
            the_file.write("\n")
