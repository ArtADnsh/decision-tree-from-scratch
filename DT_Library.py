import numpy as np

class Node:
    """
        Represents a node of the decision tree.
    
        Internal node: feature_index, threshold, left, right are set.
        Leaf node:     value (predicted class) is set, children are None.
    """
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        samples=0,
        impurity=0.0,
        class_counts=None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # Predicted class label; set ONLY for leaf nodes (majority vote), None for internal nodes
        self.samples = samples # int: Number of training samples reaching this node
        self.impurity = impurity # float: Impurity score (Gini or Entropy) of the samples at this node
        self.class_counts = class_counts # 1D array/dict: Distribution of samples per class at this node (e.g., [n_class_0, n_class_1])
        

    def is_leaf(self):
        """
        Check whether this node is a leaf node.
        Return True if the node has no children; otherwise, return False.
        """
        if self.left is None and self.right is None:
            return True
        
        return False

class DecisionTree:

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        # =========================
        # Hyperparameters (set before training)
        # =========================
        self.criterion = criterion                    # Impurity metric: "gini" or "entropy"
        self.max_depth = max_depth                    # Max levels of the tree (None = expand until leaves are pure)
        self.min_samples_split = min_samples_split    # Min samples in a node required to attempt a split
        self.min_samples_leaf = min_samples_leaf      # Min samples required in each resulting child node
        self.max_features = max_features              # Number of features to inspect per split (None = all features)
        self.random_state = random_state              # Seed for reproducible random feature selection
        
        # =========================
        # Learned Attributes (set during fit)
        # =========================
        self.root_ = None                 # Node: The root Node instance holding the learned tree structure
        self.n_features_ = None           # int: Total number of input columns/features in training X
        self.classes_ = None              # np.ndarray: Sorted 1D array of unique class labels found in y (e.g., array([0, 1]))
        self.n_classes_ = None            # int: Number of unique classes, equal to len(self.classes_)
        self.n_nodes_ = None              # int: Total count of all nodes (internal decision nodes + leaf nodes) created
        self.depth_ = None                # int: The actual maximum depth reached by the tree after fitting
        self.feature_importances_ = None  # np.ndarray of shape (n_features_,): Normalized importance score of each feature based on impurity decrease


    # =========================
    # Training
    # =========================
    def _weighted_entropy(self, y_left, y_right):
        """
        Calculate the weighted entropy after splitting a node into two children.
        """
        total_samples = len(y_left) + len(y_right)
        if total_samples == 0:
            return 0.0
        
        left_weight = len(y_left) / total_samples
        right_weight = len(y_right) / total_samples
        
        left_entropy = self._entropy(y_left)
        right_entropy = self._entropy(y_right)
        
        H_weighted = (left_weight * left_entropy) + (right_weight * right_entropy)
        
        return H_weighted
        

    def _weighted_gini(self, y_left, y_right):
        """
        Calculate the weighted Gini impurity after splitting a node into two children.
        """
        total_samples = len(y_left) + len(y_right)
        if total_samples == 0:
            return 0.0

        left_weight = len(y_left) / total_samples
        right_weight = len(y_right) / total_samples
        
        left_gini = self._gini(y_left)
        right_gini = self._gini(y_right)
        
        Gini_weighted = (left_weight * left_gini) + (right_weight * right_gini)
        
        return Gini_weighted
    

    def _entropy(self, y):
        """
        Calculate the entropy of the given target labels.
        """        
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        return float(-np.sum(probabilities * np.log2(probabilities)))
    

    def _gini(self, y):
        """
        Calculate the Gini impurity of the given target labels.
        """        
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        return float(1 - np.sum(probabilities ** 2))
        

    def _information_gain(self, y, y_left, y_right):
        """
        Calculate the information gain produced by a split.
        """
        impurity_parent = self._calculate_impurity(y)
        
        criterion = self.criterion.lower()
        
        if criterion == "entropy":
            weighted_child_impurity = self._weighted_entropy(y_left, y_right)
        elif criterion == "gini":
            weighted_child_impurity = self._weighted_gini(y_left, y_right)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")         
           
        information_gain = impurity_parent - weighted_child_impurity
        
        return information_gain


    def fit(self, X, y):
        """
        Train the Decision Tree using the given training data.
        """        
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be the same.")
        
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        self.root_ = self._build_tree(X, y, depth=0)
        
        return self
        

    def _build_tree(self, X, y, depth):
        """
        Recursively build a subtree from the given training samples.
        """
        # 1. Check whether this node should become a leaf.
        if self._is_stopping_condition(X, y, depth):
            return self._create_leaf(y)

        # 2. Find the best feature and threshold.
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return self._create_leaf(y)
        
        # 3. Split the dataset using the selected feature and threshold.        
        X_left, y_left, X_right, y_right = self._split(X, y, best_feature, best_threshold)

        # 4. Build the left subtree.
        left = self._build_tree(X_left, y_left, depth + 1)

        # 5. Build the right subtree.
        right = self._build_tree(X_right, y_right, depth + 1)

        # 6. Create the current internal node.
        node = Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left,
            right=right,
            samples=len(y),
            impurity=self._calculate_impurity(y),
            class_counts=np.array([np.sum(y == c) for c in self.classes_])
        )
                    
        return node


    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold for splitting the current node.
        """
        _, n_features = X.shape
        
        best_gain = -float("inf")
        best_feature = None
        best_threshold = None
        
        for feature_index in range(n_features):
            feature = X[:, feature_index]
            
            thresholds = self._get_candidate_thresholds(feature)
            
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature_index, threshold)
                
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                gain = self._calculate_information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold
            

    def _split(self, X, y, feature_index, threshold):
        """
        Split the dataset into left and right child nodes using a
        feature and threshold.
        """        
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        return X_left, y_left, X_right, y_right

        
    def _get_candidate_thresholds(self, feature_values):
        """
        Generate candidate thresholds for a single feature.
        """        
        unique_values = np.unique(feature_values)
        candidate_thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        return candidate_thresholds


    def _calculate_impurity(self, y):
        """
        Calculate the impurity of a node according to the selected criterion.
        """
        if self.criterion.lower() == "entropy":
            return self._entropy(y)
        elif self.criterion.lower() == "gini":
            return self._gini(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")


    def _calculate_information_gain(self, parent_y, left_y, right_y):
        """
        Calculate the information gain produced by a split.
        """
        parent_impurity = self._calculate_impurity(parent_y)
        criterion = self.criterion.lower()
        
        if criterion == "entropy":
            weighted_child_impurity = self._weighted_entropy(left_y, right_y)
        elif criterion == "gini":   
            weighted_child_impurity = self._weighted_gini(left_y, right_y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
        
        information_gain = parent_impurity - weighted_child_impurity
        
        return information_gain


    def _is_stopping_condition(self, X, y, depth):
        """
        Check whether the current node should become a leaf.
        """
        if len(set(y)) <= 1:
            return True
        if len(y) < self.min_samples_split:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True

        return False
        

    def _create_leaf(self, y):
        """
        Create a leaf node using the target labels of the current node.
        """        
        class_counts = np.array([np.sum(y == c) for c in self.classes_])
        predicted_class = self.classes_[np.argmax(class_counts)]
        
        leaf_node = Node(
            value=predicted_class,
            samples=len(y),
            impurity=self._calculate_impurity(y),
            class_counts=class_counts
        )
        
        return leaf_node


    def _predict_one(self, x):
        """
        Predict the class of a single sample.
        """
        if self.root_ is None:
            raise RuntimeError("This DecisionTree instance is not fitted yet.")
        
        node = self.root_
        
        while not node.is_leaf():
            feature_value = x[node.feature_index]
            if feature_value <= node.threshold:
                node = node.left
            else:
                node = node.right
                
        return node.value
    
    
    def predict(self, X):
        """
        Predict the class of each sample in X.
        """        
        predictions = np.array([self._predict_one(x) for x in X])
        
        return predictions
