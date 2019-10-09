from brainyboa.ensembling import CARTClassifier, print_tree
import numpy as np

# np.array([
#     [1, 3, 1],
#     [2, 3, 1],
#     [3, 1, 2],
#     [3, 1, 2],
#     [2, 3, 3],
# ])

if __name__ == '__main__':
    dataset = training_data = np.array([
        ['Green', 3, 'Apple'],
        ['Yellow', 3, 'Apple'],
        ['Red', 1, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ], dtype = object)

    print(dataset.shape)
    cart = CARTClassifier()
    cart.fit(dataset[:, 0 : 2], dataset[:, -1])
    # print(cart.classify(np.array([['Green', 3], ['Yellow', 4], ['Red', 2], ['Yellow', 3]], dtype = object)))
    print_tree(cart.root)
