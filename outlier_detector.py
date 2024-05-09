

class OutlierDetector:
    def __init__(self,X,y,special_model, evaluate_model,special_scoring,evaluate_scoring) -> None:
        """
        special_model: a model that can be trained via `fit`, have method `predict` and is used for determining if some sample is outlier
        evaluate_model: a model that can be trained via `fit`, have method `predict` and is used for determining global model error on dataset
        """
        self.X=X
        self.y=y
        self.special_model=special_model
        self.evaluate_model=evaluate_model
        self.special_scoring = special_scoring
        self.evaluate_scoring = evaluate_scoring