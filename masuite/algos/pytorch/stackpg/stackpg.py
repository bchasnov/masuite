import torch
from torch._C import dtype
import torch.autograd as autograd
from masuite.algos.pytorch import SimplePG

class StackPG(SimplePG):
    def __init__(self, agents, n_players: int, batch_size: int):
        assert n_players == 2, "StackPG only works for two players"
        super().__init__(agents, n_players, batch_size)
        self._reset_batch_info()
    
    @staticmethod
    def _vectorize(grads):
        """Given an iterable of gradients, flatten them into a single
        continuous vector
        
        Keyword arguments:
        grads -- iterable of gradients to vectorize
        
        returns -- single flattened vector of all gradients"""
        return torch.cat([g.contiguous().view(-1) for g in grads])
    

    @staticmethod
    def _resize(params, grads):
        new_grads = []
        index = 0
        for p in params:
            new_grads.append(grads[index:index+p.numel()].reshape(p.shape))
        if index != grad.numel():
            raise ValueError("Gradient size mismatch")
        return new_grads
    

    def compute_loss_p(self, obs, act2):
        logp2 = self.agents[0]._get_policy().log_prob(act2)
        return (logp2).mean()


    def compute_loss12(self, obs, act1, act2, weights):
        logp1 = self.agents[0]._get_policy(obs).log_prop(act1)
        logp2 = self.agents[1]._get_policy(obs).log_prob(act2)
        return (logp1 * logp2 * weights)

    
    def conjugate_gradient_stac_pg(self, vec, params, b, vec2, x=None,
        nsteps=10, residual_tol=1e-18, reg=0, device=torch.device("cpu")):

        if x is None:
            x = torch.zeros(b.shape, device=device)
        
        _Ax = autograd.grad(vec, params, grad_outputs=x, retain_graph=True)
        Ax = self._vectorize(_Ax)
        Ax += vec * torch.dot(vec2, x)
        Ax += reg * x

        r = b.clone().detach()-Ax
        p = r.clone.detach()
        rsold = torch.dot(r.view(-1), r.view(-1))

        for itr in range(nsteps):
            _Ap = autograd.grad(vec, params, grad_outputs=p, retain_graph=True)
            Ap = self._vectorize(_Ap)
            Ap += reg * p

            alpha = rsold / torch.dot(p.view(-1), Ap.view(-1))
            x.data.add_(alpha * p)
            r.data.add_(-alpha * Ap)
            rsnew = torch.dot(r.view(-1), r.view(-1))
            if rsnew < residual_tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return x, itr + 1

    
    def _step(self, obs, acts):
        info = dict()
        f1 = self._compute_loss(obs, acts[0], self.batch_weights[0])
        f2 = -self._compute_loss(obs, acts[1], self.batch_weights[1])
        info["loss"] = (f1, f2)
        p1, p2 = self.agents[0]._get_params, self.agents[1]._get_params
        D1f1 = autograd.grad(f1, p1(), create_graph=True)
        D1f1_vec = self._vectorize(D1f1)
        D2f2 = autograd.grad(f2, p2(), create_graph=True)
        D2f2_vec = self._vectorize(D2f2)
        D2f1_vec = -D2f2_vec.clone()

        f_p = self.compute_loss_p(
            obs=torch.as_tensor(obs, dtype=torch.float32),
            act2=torch.as_tensor(acts[1], dtype=torch.int32)
        )

        D2p = autograd.grad(f_p, p2(), create_graph=True)
        D2p_vec = self._vectorize(D2p)

        x, _ = self.conjugate_gradient_stac_pg(
            vec=D2f2_vec,
            params=p2(),
            vec2=D2f1_vec.detach(),
            x=D2p_vec.detach(),
        )

        f2_surro = self.compute_loss12(
            obs=torch.as_tensor(obs, dtype=torch.float32),
            act1=torch.as_tensor(acts[1], dtype=torch.int32,
            act2=torch.as_tensor(acts[1], dtype=torch.int32),
            weights=torch.as_tensor(self.batch_lens[0], dtype=torch.float32))
        )

        D2f2_surro = autograd.grad(f2_surro, p2(), create_graph=True)
        D2f2_surro_vec = self._vectorize(D2f2_surro)

        _Avec = autograd.grad(D2f2_surro_vec, p2(), create_graph=True)
        # TODO: device
        grad_imp = torch.cat(
            [g.contiguous().view(-1) if g is not None else torch.Tensor([0]) for g in _Avec]
        )
        grad_stac = D1f1_vec.detach() - grad_imp
        grads = [self._resize(p1(), grad_stac), self._resize(p2(), D2f2_vec.detach())]

        return grads, info
