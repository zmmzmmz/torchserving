import torch
from ts.torch_handler.base_handler import BaseHandler


class GroundedSAMHandler(BaseHandler):
    def __init__(self):
        # call superclass initializer
        super().__init__()

    def initialize(self, context):
        # call superclass initialization method
        self.manifest = context.manifest
        properties = context.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        self.initialized = True

    def handle(self, data, context):
        with torch.no_grad():
            responses = []
            responses.append({"success": True, "data": "test"})
            return responses
