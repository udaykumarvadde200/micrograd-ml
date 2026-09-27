class Momentum:

    def __init__(
        self,
        parameters,
        lr=0.01,
        momentum=0.9
    ):

        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum

        self.velocity = {
            id(p): 0.0
            for p in parameters
        }

    def step(self):

        for p in self.parameters:

            pid = id(p)

            self.velocity[pid] = (
                self.momentum
                * self.velocity[pid]
                - self.lr
                * p.grad
            )

            p.data += self.velocity[pid]

    def zero_grad(self):

        for p in self.parameters:

            p.grad = 0