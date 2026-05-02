class ScalerInterface:
    """Abstract base so Docker and K8s scalers are interchangeable."""
    
    def scale_up(self):
        raise NotImplementedError
        
    def scale_down(self):
        raise NotImplementedError
        
    def hold(self):
        print("Holding — no scaling action taken.")
        return True