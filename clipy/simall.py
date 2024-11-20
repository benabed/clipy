from . import *

from . import smica_component as smcmp

from . import lkl 

class simall_lkl(lkl._clik_lkl):

  def __init__(self,lkl,**options):
    super().__init__(lkl,**options)
  
    self.nell = len(self.ell)
    self.llp1 = self.ell*(self.ell+1)/2./nm.pi
    
    if self.has_cl[1]:
      self.nstepsEE = lkl["nstepsEE"]
      self.stepEE = lkl["stepEE"]
      self.probEE = jnp.reshape(jnp.array(lkl["probEE"]),(self.nell,self.nstepsEE))
    if self.has_cl[2]:
      self.nstepsBB = lkl["nstepsBB"]
      self.stepBB = lkl["stepBB"]
      self.probBB = jnp.reshape(jnp.array(lkl["probBB"]),(self.nell,self.nstepsBB))
    if self.has_cl[3]:
      self.nstepsTE = lkl["nstepsTE"]
      self.stepTE = lkl["stepTE"]
      self.probTE = jnp.reshape(jnp.array(lkl["probTE"]),(self.nell,self.nstepsTE))

  def __call__(self,cls,nuisance_dict,chi2=False):
    cls = self._calib(cls,nuisance_dict)
    res = 0
    if self.has_cl[1]:
      dl = jnp.astype(jnp.floor(cls[1,self.lmin:]*self.llp1/self.stepEE),jnp.int32)
      res += jnp.sum(jnp.take(self.probEE,dl,axis=1).diagonal())
    if self.has_cl[2]:
      dl = jnp.astype(jnp.floor(cls[2,self.lmin:]*self.llp1/self.stepBB),jnp.int32)
      res += jnp.sum(jnp.take(self.probBB,dl,axis=1).diagonal())
    if self.has_cl[3]:
      dl = jnp.astype(jnp.floor(cls[3,self.lmin:]*self.llp1/self.stepTE),jnp.int32)
      res += jnp.sum(jnp.take(self.probTE,dl,axis=1).diagonal())

    return res
