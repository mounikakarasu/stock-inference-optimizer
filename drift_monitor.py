import numpy as np
import time

class DriftDetector:
    def __init__(self, threshold=0.05, window_size=20):
        self.threshold = threshold
        self.window_size = window_size
        self.errors = []

    def update(self, y_true, y_pred):
        #absolute error
        error = abs(y_true - y_pred)
        self.errors.append(error)
        
        #historyupdation
        if len(self.errors) > self.window_size:
            self.errors.pop(0)
        
        return error

    def check_drift(self):
        if len(self.errors) < self.window_size:
            return False, 0.0
        
        #rolling mean error
        avg_error = np.mean(self.errors)
        
        #drift detection
        if avg_error > self.threshold:
            return True, avg_error
        
        return False, avg_error

#simulation
if __name__ == "__main__":
    print(f"{'Day':<5} | {'True Value':<10} | {'Pred':<10} | {'Error':<10} | {'Status'}")
    print("-" * 65)

    detector = DriftDetector(threshold=0.10) # Set sensitivity
    
    
    for day in range(1, 51):
        #datageneration
        # normal market: true value is around 0.5
        if day < 35:
            true_val = np.random.normal(0.5, 0.05)
            pred_val = true_val + np.random.normal(0, 0.02) # Good prediction
        else:
            #marketcrash: True value spikes to 0.9
            #yet model still predicts around 0.5
            true_val = np.random.normal(0.9, 0.05) 
            pred_val = 0.5 + np.random.normal(0, 0.02) 

        #monitor
        current_error = detector.update(true_val, pred_val)
        drift_detected, rolling_error = detector.check_drift()

        #logging
        status = "OK"
        if drift_detected:
            status = f"DRIFT DETECTED! (Avg Err: {rolling_error:.4f})"
        
        print(f"{day:<5} | {true_val:.4f}     | {pred_val:.4f}     | {current_error:.4f}     | {status}")
        
        if drift_detected:
            print("-" * 65)
            print(">>> CRITICAL ALERT: Model performance degraded.")
            print(">>> ACTION: Triggering automated retraining pipeline...")
            break
        
        time.sleep(0.05) # Simulate real-time delay