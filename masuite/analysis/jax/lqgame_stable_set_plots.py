import jax.numpy as jnp
from jax import vmap, lax

def lyap_iter(A,B,Q,R,N=10):
    def f(P,_):
        return (A.T@P@A - (A.T@P@B)@jnp.linalg.solve(R+B.T@P@B, B.T@P@A) + Q,_)
    return lax.scan(f, Q, jnp.arange(N))[0]

def lyap_dare(A,B,Q,R):
    ATinv = jnp.linalg.inv(A).T
    XX = jnp.hstack([A+B@jnp.linalg.solve(R, B.T@jnp.linalg.solve(A.T, Q)), \
                    -B@jnp.linalg.solve(R, B.T@ATinv)])
    YY = jnp.hstack([-jnp.linalg.solve(A.T, Q), ATinv])
    ZZ = jnp.vstack([XX, YY])
    _, UU = jnp.linalg.eig(ZZ)
    U1, U2 = UU[:N,N:], UU[N:,N:]
    P = jnp.linalg.solve(U1.T, U2.T)
    return jnp.real((P+jnp.conj(P).T)/2)
    
def roa(K0, L0, lr1, lr2, N=9):
    def f(K,_):
        K0,L0 = K
        X0 = lyap_dare(A-B2 @ L0, B1, Q - L0.T @ R2 @ L0, R1)
        W0 = lyap_dare(A-B1 @ K0, B2, -Q - K0.T @ R1 @ K0, R2)
        gd_k = 2 * ( R1 @ K0 - B1.T @ (-W0) @ (A-B1@K0 - B2@L0) )
        gd_l = 2 * ( -R2 @ L0 - B2.T @ X0 @ (A-B1@K0 - B2@L0) )
        K1 = K0 + lr1*gd_k
        L1 = L0+lr2*gd_l
        
        return (K1, L1),  jnp.max(jnp.abs(jnp.linalg.eigvals(A-B1@K1-B2@L1)))
    
    return lax.scan(f, (K0, L0), jnp.arange(N))[1][-1]
    

ones = jnp.ones(N)
def specrad(e1, e2, P):
    K = P['K']
    L = P['L']
    Kdir = P['Kdir']
    Ldir = P['Ldir']
    KK = K - e1*Kdir
    LL = L + e2*Ldir
    
    X0 = lyap_dare(A-B1@KK-B2@LL, B, Q-LL.T@R2@LL+KK.T@R1@KK, R)
    cost = np.real(jnp.trace(1/2*(X0+jnp.conj(X0).T)@initial_state))
    
    eigs = jnp.linalg.eigvals(A-B1@KK -B2@LL)
    sp = jnp.max(jnp.abs(eigs))
    controllability =  jnp.linalg.matrix_rank(
        jnp.hstack([((A-B2@LL)**k)@B1 for k in range(N)])) 
    
    #return controllability
    return roa(KK, LL, 0.005, 0.005)
    return cost*(sp<=1)/(sp<=1)
#, sp*(sp<=1)/(sp<=1)


def main(grads, profiles):
    e1 = 0.0
    e2 = 0.0
    noise = 0.1

    np.random.seed(0)

    _ = vmap(specrad, (0,None,None))
    specradfn = vmap(_, (None,0,None))

    lyap_dare(A,B,Q,R)

    roa(Kinitial,Linitial,0.005,0.005)

    limax = 40
    limin = -40
    nlvl = 21
    count = 0

    cx = 5
    cy = 5
    linx = jnp.linspace(-cx,cx,8)
    liny = jnp.linspace(-cy,cy,8)
    XX, YY = np.meshgrid(linx,liny)

    for (Kdir,Ldir), (_K,_L) in zip(grads, profiles):
        #scale = np.sqrt(np.linalg.norm(Kdir)**2 + np.linalg.norm(Ldir)**2)
        Kdir = Kdir/np.linalg.norm(Kdir)
        Ldir = Ldir/np.linalg.norm(Ldir)
        P = dict(K=_K, L=_L, Kdir=Kdir, Ldir=Ldir)
        out = specradfn(linx,liny,P)
    #    limax = np.maximum(np.abs(np.nanmax(out)), np.abs(np.nanmin(out)))
    #    limin = -limax
        out = jnp.maximum(jnp.minimum(out,limax),limin)
        plt.figure(figsize=(5,5))
        img = plt.contourf(XX, YY, out, cmap='jet', levels=np.linspace(limin,limax,nlvl))
        plt.colorbar(img)
        plt.plot(0,0,'+k', ms=20)
        plt.title('Iteration: {}'.format(count))
        plt.savefig('lqr-simgrad-{}.png'.format(count))
        plt.show()
        count += 1