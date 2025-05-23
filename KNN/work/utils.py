import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    # assert len(real_labels) == len(predicted_labels)
    # raise NotImplementedError
    l = len(real_labels)
    tp = fp = fn = 0
    for i in range(l):
        if real_labels[i]==1 and predicted_labels[i]==1:
            tp += 1
        elif real_labels[i]==0 and predicted_labels[i]==1:
            fp+=1
        elif real_labels[i]==1 and predicted_labels[i]==0:
            fn+=1
    deno = 2*tp+fp+fn
    return 2*tp/deno if deno!=0.0 else 0.0



class Distances:



    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        # raise NotImplementedError
        l = len(point1)
        cur_sum = 0
        for i in range(l):
            cur_sum += (abs(point1[i]-point2[i]))**3

        return cur_sum**(1/3)


    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        l = len(point1)
        cur_sum = 0
        for i in range(l):
            cur_sum += (point1[i] - point2[i]) ** 2

        return cur_sum ** 0.5




    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """

        def norm(X):
            cur_sum = 0
            for x in X:
                cur_sum += x ** 2

            return cur_sum ** 0.5

        def dot_p(X1, X2):
            l = len(X1)
            cur_sum = 0
            for i in range(l):
                cur_sum += X1[i] * X2[i]

            return cur_sum
        norm_1,norm_2 = norm(point1),norm(point2)
        dot_product = dot_p(point1,point2)

        if norm_1 == 0 or norm_2==0:
            return 1.0

        return 1-dot_product/(norm_1*norm_2)



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        best_f1 = -1
        priority = {'euclidean': 3, 'minkowski': 2, 'cosine_dist': 1}
        for name, func in distance_funcs.items():
            for k in range(1,30,2):
                knn_model = KNN(k, func)
                knn_model.train(x_train, y_train)
                pred = knn_model.predict(x_val)
                f1=f1_score(y_val, pred)
                # if self.best_k is None or f1>best_f1:
                #     self.best_k=k
                #     self.best_distance_function = name
                #     self.best_model = knn_model
                #     best_f1=f1
                # elif f1==best_f1:
                #     if name=="euclidean" and self.best_distance_function!=name:
                #         self.best_k = k
                #         self.best_distance_function = name
                #         self.best_model = knn_model
                #     elif name=="minkowski" and self.best_distance_function=="cosine_dist":
                #         self.best_k = k
                #         self.best_distance_function = name
                #         self.best_model = knn_model
                #
                # elif f1==best_f1 and self.best_distance_function==name and self.best_k>k:
                #     self.best_k=k
                #     self.best_model=knn_model
                update=False
                if f1>best_f1:
                    update = True
                elif f1 == best_f1:
                    if priority[name]>priority[self.best_distance_function]:
                        update = True
                    elif priority[name]==priority[self.best_distance_function] and self.best_k>k:
                        update = True
                    else:
                        update = False
                else:
                    update = False

                if update:
                    best_f1 = f1
                    self.best_k=k
                    self.best_distance_function=name
                    self.best_model=knn_model



    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        best_f1 = -1

        scaler_priority = {'min_max_scale': 2, 'normalize': 1}
        distance_priority = {'euclidean': 3, 'minkowski': 2, 'cosine_dist': 1}

        for scale_name, scale_func in scaling_classes.items():
            scaler = scale_func()
            scaled_x_train = scaler(x_train)
            scaled_x_val = scaler(x_val)

            for name, func in distance_funcs.items():
                for k in range(1, 30, 2):
                    knn_model = KNN(k, func)
                    knn_model.train(scaled_x_train, y_train)
                    pred = knn_model.predict(scaled_x_val)
                    f1 = f1_score(y_val, pred)

                    update = False
                    if f1 > best_f1:
                        update = True
                    elif f1 == best_f1:
                        current_scaler_priority = scaler_priority.get(self.best_scaler, 0)
                        current_distance_priority = distance_priority.get(self.best_distance_function, 0)

                        if scaler_priority[scale_name] > current_scaler_priority:
                            update = True
                        elif scaler_priority[scale_name] == current_scaler_priority:
                            if distance_priority[name] > current_distance_priority:
                                update = True
                            elif distance_priority[name] == current_distance_priority:
                                if self.best_k is None or k < self.best_k:
                                    update = True

                    if update:
                        best_f1 = f1
                        self.best_k = k
                        self.best_distance_function = name
                        self.best_model = knn_model
                        self.best_scaler = scale_name

    # def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
    #     """
    #     This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them.
	#
    #     :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
    #     :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
    #     :param x_train: List[List[int]] training data set to train your KNN model
    #     :param y_train: List[int] train labels to train your KNN model
    #     :param x_val: List[List[int]] validation data
    #     :param y_val: List[int] validation labels
    #
    #     Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
    #
    #     NOTE: When there is a tie, choose the model based on the following priorities:
    #     First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
    #     """
    #
    #     # You need to assign the final values to these variables
    #     self.best_k = None
    #     self.best_distance_function = None
    #     self.best_scaler = None
    #     self.best_model = None
    #     best_f1 = -1
    #     priority = {'euclidean': 3, 'minkowski': 2, 'cosine_dist': 1}
    #     for scale_name,scale_func in scaling_classes.items():
    #         scaler = scale_func()
    #         scaled_x_train = scaler(x_train)
    #         scaled_x_val = scaler(x_val)
    #
    #         for name, func in distance_funcs.items():
    #             for k in range(1, 30, 2):
    #                 knn_model = KNN(k, func)
    #                 knn_model.train(scaled_x_train, y_train)
    #                 pred = knn_model.predict(scaled_x_val)
    #                 f1 = f1_score(y_val, pred)
    #                 update = False
    #                 if f1 > best_f1:
    #                     update = True
    #                 elif f1 == best_f1:
    #                     if scale_name=="min_max_scale" and self.best_scaler=="normalize":
    #                         update=True
    #                     elif priority[name] > priority[self.best_distance_function]:
    #                         update = True
    #                     elif priority[name] == priority[self.best_distance_function] and self.best_k > k:
    #                         update = True
    #                     else:
    #                         update = False
    #                 elif f1<best_f1:
    #                     update = False
    #
    #                 if update:
    #                     best_f1 = f1
    #                     self.best_k = k
    #                     self.best_distance_function = name
    #                     self.best_model = knn_model
    #                     self.best_scaler = scale_name


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def norm(self,X):
        cur_sum = 0
        for x in X:
            cur_sum+=x**2

        return cur_sum**0.5
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        scaled_f=[]
        for f in features:
            norm=self.norm(f)
            if norm == 0:
                new_f = [0 for _ in f]
            else:
                new_f = [x / norm for x in f]
            scaled_f.append(new_f)

        return scaled_f


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        l = len(features)
        l_f = len(features[0])
        new_features = [[0]*l_f for _ in range(l)]
        for i in range(l_f):
            max_f=float('-inf')
            min_f=float('inf')
            for j in range(l):
                max_f=max(max_f,features[j][i])
                min_f=min(min_f,features[j][i])
            max_min = max_f-min_f
            for j in range(l):
                # print("max_min",max_min)
                if max_min != 0:
                    new_features[j][i] = (features[j][i]-min_f)/max_min
                else:
                    new_features[j][i]=0.0

        return new_features

