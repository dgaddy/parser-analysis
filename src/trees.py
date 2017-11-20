import collections.abc

class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children)

    def brackets(self, advp_prt=True):
        return self._brackets(0, advp_prt)[0]

    def _brackets(self, start=0, advp_prt=True):
        results = []

        position = start
        for child in self.children:
            b, e = child._brackets(position, advp_prt)
            results.extend(b)
            position = e
        end = position

        label = self.label
        if label != 'TOP':
            if advp_prt and label =='PRT':
                label = 'ADVP'
            results.append((start, end, label))

        return results, end

class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)

    def _brackets(self, start=0, advp_prt=True):
        if self.tag in [",", ".", ":", "``", "''"]:
            return [], start
        else:
            return [], start+1

class ParseNode(object):
    pass

class InternalParseNode(ParseNode):
    def __init__(self, label, children):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]

    def iterate_spans_with_parents(self):
        for child in self.children:
            if isinstance(child, InternalParseNode):
                yield child, self
            yield from child.iterate_spans_with_parents()

class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

    def iterate_spans_with_parents(self):
        return []

class SpanList(object):
    def __init__(self, tagged_words):
        # tagged_words is list of (tag, word) tuples
        self.tagged_words = tagged_words
        self.list = []

    def add(self, left, right, label):
        assert isinstance(label, str)
        self.list.append((left, right, label))

    def convert(self):
        return self

    def brackets(self, advp_prt=True):
        location_shift = []
        i = 0
        for tag, word in self.tagged_words:
            location_shift.append(i)
            if tag not in [",", ".", ":", "``", "''"]:
                i += 1
        location_shift.append(i)
        result = []
        for left, right, label in self.list:
            if label == 'TOP':
                continue
            if advp_prt and label =='PRT':
                label = 'ADVP'
            result.append((location_shift[left], location_shift[right], label))
        return result

def load_trees(path, strip_top=True):
    with open(path) as infile:
        tokens = infile.read().replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label == "TOP":
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    return trees
