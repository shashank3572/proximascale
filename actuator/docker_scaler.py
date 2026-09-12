import docker
import uuid
from actuator.scaler_interface import ScalerInterface

class DockerActuator(ScalerInterface):
    def __init__(self, config):
        self.config = config
        self.image = self.config['scaling_rules']['target_service_image']
        self.prefix = "ai_sysadmin_worker_"
        
        # Let main.py handle the connection errors!
        self.client = docker.from_env()
        print(f"Actuator: Ensuring image {self.image} is available...")
        self.client.images.pull(self.image)

    def get_workers(self):
        if not self.client:
            return []
        return self.client.containers.list(filters={"name": self.prefix})

    def scale_up(self):
        if not self.client:
            return False
            
        workers = self.get_workers()
        if len(workers) < self.config['scaling_rules']['max_containers']:
            # Generate a unique 6-character string to avoid name collisions
            unique_id = uuid.uuid4().hex[:6]
            new_name = f"{self.prefix}{unique_id}"
            
            print(f"⚙️ Actuator: Spinning up new container -> {new_name}")
            self.client.containers.run(self.image, name=new_name, detach=True)
            return True
            
        print("⚠️ Actuator: Max container limit reached. Cannot scale up.")
        return False

    def scale_down(self):
        if not self.client:
            return False
            
        workers = self.get_workers()
        if len(workers) > self.config['scaling_rules']['min_containers']:
            # Grab any active worker from the list safely
            target = workers[0]
            print(f"🛑 Actuator: Stopping and removing -> {target.name}")
            target.stop()
            target.remove()
            return True
            
        print("⚠️ Actuator: Minimum container limit reached. Cannot scale down.")
        return False