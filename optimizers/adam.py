class Adam:

    def __init__(
        self,
        parameters,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):

        self.parameters = parameters

        self.lr = lr

        self.beta1 = beta1
        self.beta2 = beta2

        self.eps = eps

        self.m = {
            id(p): 0.0
            for p in parameters
        }

        self.v = {
            id(p): 0.0
            for p in parameters
        }

        self.t = 0

    def step(self):

        self.t += 1

        for p in self.parameters:

            pid = id(p)

            self.m[pid] = (
                self.beta1 * self.m[pid]
                + (1 - self.beta1) * p.grad
            )

            self.v[pid] = (
                self.beta2 * self.v[pid]
                + (1 - self.beta2)
                * (p.grad ** 2)
            )

            m_hat = (
                self.m[pid]
                / (1 - self.beta1 ** self.t)
            )

            v_hat = (
                self.v[pid]
                / (1 - self.beta2 ** self.t)
            )

            p.data -= (
                self.lr
                * m_hat
                / (
                    v_hat ** 0.5
                    + self.eps
                )
            )

    def zero_grad(self):

        for p in self.parameters:

            p.grad = 0