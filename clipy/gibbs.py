from . import *

from . import lkl 

if not hasjax:
  def cond(pred, true_fun, false_fun, operand):
    if pred:
      return true_fun(operand)
    else:
      return false_fun(operand)
else:
  from jax import lax
  cond = lax.cond

class gibbs_lkl(lkl._clik_cmb):
  
  def __init__(self,lkl,**options):
    import astropy.io.fits as pf
    import os.path as osp
    super().__init__(lkl,**options)
    self.delta_l = lkl["delta_l"]
    assert(lkl["version"]==3)

    sigma_file = pf.open(osp.join(lkl._name,"_external","sigma.fits"))
    lmin_in = sigma_file[0].header["LMIN"]
    lmax_in = sigma_file[0].header["LMAX"]
    self.nbin = sigma_file[0].header["NBIN"]
    self.nl = self.lmax+1-self.lmin
    cl2x_in = sigma_file[0].data
    assert cl2x_in.shape[-1]==self.nbin
    assert cl2x_in.shape[1]==lmax_in-lmin_in+1
    assert cl2x_in.shape[0]==3
    
    
    mu_in = sigma_file[1].data
    assert mu_in.shape[0]==lmax_in-lmin_in+1
    
    mu_sigma_in = sigma_file[3].data
    assert mu_sigma_in.shape[0]==lmax_in-lmin_in+1
    
    cov_in = sigma_file[2].data
    assert cov_in.shape[0]==lmax_in-lmin_in+1
    assert cov_in.shape[1]==lmax_in-lmin_in+1

    
    self.cl2x = jnp.array(cl2x_in[:,self.lmin-lmin_in:self.lmax+1-lmin_in]*1.,dtype=jnp64)
    self.mu = jnp.array(mu_in[self.lmin-lmin_in:self.lmax+1-lmin_in]*1.,dtype=jnp64)
    self.mu_sigma = jnp.array(mu_sigma_in[self.lmin-lmin_in:self.lmax+1-lmin_in]*1.,dtype=jnp64)
    self.cov = cov_in[self.lmin-lmin_in:self.lmax+1-lmin_in,:][:,self.lmin-lmin_in:self.lmax+1-lmin_in]*1.
    
    
    # bandlimit cov
    for i in range(self.nl):
      for j in range(self.nl):
        if nm.abs(i-j)>self.delta_l:
          self.cov[i,j]=0.

    prior1 = [self.cl2x[0,i,nm.max(nm.where(nm.abs(self.cl2x[1]+5)<1e-4,nm.indices((self.nl,1000))[1]+1,0),1)[i]+2] for i in range(self.nl)]
    prior2 = [self.cl2x[0,i,nm.min(nm.where(nm.abs(self.cl2x[1]-5)<1e-4,nm.indices((self.nl,1000))[1]+1,10000),1)[i]-4] for i in range(self.nl)]
    self.prior = jnp.array([prior1,prior2],dtype=jnp64)
    
    
    self.cov = jnp64(nm.linalg.inv(self.cov))

    self.offset = jnp64(0.)
    self.offset = self.internal_lkl(self.mu_sigma)
    
  def internal_lkl(self,vec):
    return cond(jnp.logical_or(jnp.any(vec<self.prior[0]),jnp.any(vec>self.prior[1])),
                lambda x: jnp64(-1e30),
                self._sure_lkl,
                vec)

  def _sure_lkl(self,vec):
    x,dxdCl = self.splint_gauss_and_deriv(vec)
    delta = x-self.mu 
    lkl = jnp64(-.5) * (delta).T @ self.cov @ delta
    
    lkl += jnp.sum(jnp.log(dxdCl))
    
    return lkl-self.offset

  def _searchsorted(self,A,v):
    #return jnp.searchsorted(A,v)
    return jnp.sum(jnp.where(A<v,jnp.ones(len(A),dtype=jnp.int64),jnp.zeros(len(A),dtype=jnp.int64)))
  def splint_gauss_and_deriv(self,cls):
    import bisect
    n   = self.nbin
    nl  = self.cl2x.shape[1]
    i=0
    klo1 = self._searchsorted(self.cl2x[0,i],cls[i])
    klo_ = jnp.array([self._searchsorted(self.cl2x[0,i],cls[i]) for i in range(nl)])
    klo__ = jnp.where(klo_>n-2,jnp.ones(nl,dtype=klo_.dtype)*(n-2),klo_)
    klo___ = jnp.where(klo__<1,jnp.ones(nl,dtype=klo_.dtype),klo__) - 1
    #klo = jnp.array([max(min(jnp.searchsorted(self.cl2x[0,i],cls[i]),n-2),1) for i in range(nl)])-1
    #print(klo-klo___)
    klo = klo___
    khi = klo+1
    dia_cl2_1_khi = self.cl2x[1,:,khi].diagonal()
    dia_cl2_2_khi = self.cl2x[2,:,khi].diagonal()
    dia_cl2_1_klo = self.cl2x[1,:,klo].diagonal()
    dia_cl2_2_klo = self.cl2x[2,:,klo].diagonal()
    h   = (self.cl2x[0,:,khi] - self.cl2x[0,:,klo]).diagonal()
    a   = (self.cl2x[0,:,khi].diagonal() - cls) / h
    b   = (cls - self.cl2x[0,:,klo].diagonal()) / h
    gauss = a*dia_cl2_1_klo + b*dia_cl2_1_khi + ((a**3-a)*dia_cl2_2_klo + (b**3-b)*dia_cl2_2_khi)*(h**2)/jnp64(6.)
    deriv_gauss = (dia_cl2_1_khi - dia_cl2_1_klo) / h - (jnp64(3.) * a**2 - jnp64(1.)) / jnp64(6.) * h * dia_cl2_2_klo + (jnp64(3.) * b**2 - jnp64(1.)) / jnp64(6.) * h * dia_cl2_2_khi
    return gauss, deriv_gauss

  def __call__(self,cls,nuisance_dict,chi2=False):
    cls = self._calib(cls,nuisance_dict)
    cls = cls[0,self.lmin:]*self.llp1
    r = self.internal_lkl(cls)
    if chi2:
      return -2*r
    return r

