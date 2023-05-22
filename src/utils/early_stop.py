

class EarlyStop(object):
    
    def __init__(self,patience=2,delta=0):

        self.patience=patience
        self.delta=delta

        self.best_score=None
        self.counter=0
        self.early_stop=False
    
    def __call__(self,val_loss):
        
        if self.best_score is None:
            self.best_score=val_loss
        elif val_loss>self.best_score+self.delta:
            self.counter+=1
            if self.counter>=self.patience:
                self.early_stop=True
        else:
            self.best_score=val_loss
            self.counter=0
        