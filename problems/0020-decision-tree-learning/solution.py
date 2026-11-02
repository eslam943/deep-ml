import math
from collections import Counter

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict | str:
    """
    Builds a decision tree for classification using recursive splitting based on Information Gain.
    """

    def calculate_entropy(data: list[dict]) -> float:
        if not data:
            return 0.0
        counts = Counter(ex[target_attr] for ex in data)
        total = len(data)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def get_majority_class(data: list[dict]) -> str:
        counts = Counter(ex[target_attr] for ex in data)
        max_count = max(counts.values())
        candidates = [cls for cls, count in counts.items() if count == max_count]
        # Tie-breaking: Return the class that comes first alphabetically
        return sorted(candidates)[0]

    def calculate_information_gain(data: list[dict], attribute: str) -> float:
        parent_entropy = calculate_entropy(data)
        total = len(data)

        subsets = {}
        for ex in data:
            val = ex[attribute]
            subsets.setdefault(val, []).append(ex)

        weighted_child_entropy = 0.0
        for subset in subsets.values():
            prob = len(subset) / total
            weighted_child_entropy += prob * calculate_entropy(subset)

        return parent_entropy - weighted_child_entropy

    # Base Case 1: Empty dataset
    if not examples:
        return None

    # Base Case 2: Pure dataset
    classes = set(ex[target_attr] for ex in examples)
    if len(classes) == 1:
        return list(classes)[0]

    # Base Case 3: No remaining attributes to split on
    if not attributes:
        return get_majority_class(examples)

    # Find the best attribute using Information Gain
    best_attr = None
    max_gain = float('-inf')

    for attr in attributes:
        gain = calculate_information_gain(examples, attr)
        # Tie-breaking: First attribute in list order wins if gain is equal
        if gain > max_gain:
            max_gain = gain
            best_attr = attr

    tree = {best_attr: {}}
    remaining_attributes = [a for a in attributes if a != best_attr]

    # Process attribute values in sorted order
    unique_values = sorted(list(set(ex[best_attr] for ex in examples)))

    for val in unique_values:
        subset = [ex for ex in examples if ex[best_attr] == val]

        if not subset:
            tree[best_attr][val] = get_majority_class(examples)
        else:
            tree[best_attr][val] = learn_decision_tree(subset, remaining_attributes, target_attr)

    return tree