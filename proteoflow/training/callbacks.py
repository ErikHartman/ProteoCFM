import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, cooldown=0, save_best_state=True, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.save_best_state = save_best_state
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = np.inf
        self.best_state = None
        self.cooldown_counter = 0
        
    def __call__(self, loss, model):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
            
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            if self.save_best_state:
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"New best loss: {loss:.4f}")
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.save_best_state and self.best_state is not None:
                    model.load_state_dict(self.best_state)
                    if self.verbose:
                        print("Restored best model state")
                return True
            return False
    
    def reset(self):
        self.counter = 0
        self.best_loss = np.inf
        self.best_state = None
        self.cooldown_counter = self.cooldown
