from . import *

from . import lkl 

import re

class cmbonly_lkl(lkl._clik_cmb):

  def _i_crop(self,bin_min_tt=None,bin_max_tt=None,bin_min_te=None,bin_max_te=None,bin_min_ee=None,bin_max_ee=None):
    if bin_min_tt is not None:
      self.bin_min_tt = bin_min_tt
    if bin_max_tt is not None:
      self.bin_max_tt = bin_max_tt
    if bin_min_te is not None:
      self.bin_min_te = bin_min_te
    if bin_max_te is not None:
      self.bin_max_te = bin_max_te
    if bin_min_ee is not None:
      self.bin_min_ee = bin_min_ee
    if bin_max_ee is not None:
      self.bin_max_ee = bin_max_ee
    #print ("crop to ",self.bin_min_tt,self.bin_max_tt,self.bin_min_te,self.bin_max_te,self.bin_min_ee,self.bin_max_ee)

    slices = []
    fslices = []

    self.bns = {}
    
    if not self.use_tt:
      self.bin_max_tt = self.bin_min_tt-1
    else:
      tt_slice = slice(self.bin_min_tt-1,self.bin_max_tt)
      slices.append(tt_slice)
      ftt_slice = slice(None,self.bin_max_tt-self.bin_min_tt+1)
      fslices.append(ftt_slice)
      self.bns["tt"] = self.bns_0[self.bin_min_tt-1:self.bin_max_tt]      
    if not self.use_te:
      self.bin_max_te = self.bin_min_te-1
    else:
      te_slice = slice(self.nbintt+self.bin_min_te-1,self.nbintt+self.bin_max_te)
      slices.append(te_slice)
      fte_slice = slice(self.bin_max_tt-self.bin_min_tt+1,self.bin_max_tt-self.bin_min_tt+1+self.bin_max_te-self.bin_min_te+1)
      fslices.append(fte_slice)
      self.bns["te"] = self.bns_0[self.bin_min_te-1:self.bin_max_te]      
    if not self.use_ee:
      self.bin_max_ee = self.bin_min_ee-1
    else:
      ee_slice = slice(self.nbintt+self.nbinte+self.bin_min_ee-1,self.nbintt+self.nbinte+self.bin_max_ee)
      slices.append(ee_slice)
      fee_slice = slice(self.bin_max_tt-self.bin_min_tt+1+self.bin_max_te-self.bin_min_te+1,self.bin_max_tt-self.bin_min_tt+1+self.bin_max_te-self.bin_min_te+1+self.bin_max_ee-self.bin_min_ee+1)
      fslices.append(fee_slice)
      self.bns["ee"] = self.bns_0[self.bin_min_ee-1:self.bin_max_ee]      
    
    # select data vector
    self.X_data = jnp.concatenate([jnp.array(self.X_data_0[sl],dtype=jnp64) for sl in slices])
    
    # select covmat
    self.bin_no = self.bin_max_tt-self.bin_min_tt+1 + self.bin_max_te-self.bin_min_te+1 + self.bin_max_ee-self.bin_min_ee+1
    fisher = nm.zeros((self.bin_no,self.bin_no))
    for i in range(len(slices)):
      for j in range(len(slices)):
        fisher[fslices[i],fslices[j]] = self.covmat_0[slices[i],slices[j]]
   
    # invert cov
    self.inv_cov = jnp.array(nm.linalg.inv(fisher),dtype=jnp64)
    
  def __init__(self,lkl,**options):
    import astropy.io.fits as pf
    import os.path as osp
    super().__init__(lkl,**options)

    ver = lkl["cmbonly_version"]
    assert ver>=18,"cmbonly plik <v17 not supported anymore"
    external_path = osp.join(lkl._name,"_external")

    self.varpar = ["A_planck"]

    self.use_tt = self.has_cl[0]
    self.use_ee = self.has_cl[1]
    self.use_te = self.has_cl[3]
    self.bin_min_tt = 1
    self.bin_max_tt = 215
    self.bin_min_te = 1
    self.bin_max_te = 199
    self.bin_min_ee = 1
    self.bin_max_ee = 199
    if "bin_min_tt" in lkl:
      self.bin_min_tt = lkl["bin_min_tt"]
      self.bin_max_tt = lkl["bin_max_tt"]
      self.bin_min_te = lkl["bin_min_te"]
      self.bin_max_te = lkl["bin_max_te"]
      self.bin_min_ee = lkl["bin_min_ee"]
      self.bin_max_ee = lkl["bin_max_ee"]

    self.nbintt = 215
    self.nbinte = 199
    self.nbinee = 199
    self.plmax  = 2508
    self.plmin  = 30
    self.nbin   = 613 
    
    like_file = osp.join(external_path,'cl_cmb_plik_v%d.dat'%ver)
    cov_file = osp.join(external_path,'c_matrix_plik_v%d.dat'%ver)
    blmin_file = osp.join(external_path,'blmin.dat')
    blmax_file = osp.join(external_path,'blmax.dat')
    binw_file  = osp.join(external_path,'bweight.dat')

    likedata = nm.loadtxt(like_file)
    #self.bval = jnp.array(likedata[:,0])
    self.X_data_0 = jnp.array(likedata[:,1],dtype=jnp64)
    #self.X_sig = jnp.array(likedata[:,2])

    covmat = cldf.forfile(cov_file).read("%df64"%(self.nbin*self.nbin))
    covmat.shape=(self.nbin,self.nbin)
    self.covmat_0 = covmat + covmat.T - nm.diag(covmat.diagonal())
    
    self.blmin = nm.loadtxt(blmin_file).astype(nm.int64)
    self.blmax = nm.loadtxt(blmax_file).astype(nm.int64)
    self.bin_w = jnp.array(nm.loadtxt(binw_file),dtype=jnp64)
    self.bns_0 = nm.zeros((self.nbintt,(self.plmax+1-self.plmin)),dtype=jnp64)
    for i in range(self.nbintt):
      self.bns_0[i,self.blmin[i]:self.blmax[i]+1] = jnp64(self.bin_w[self.blmin[i]:self.blmax[i]+1])
    
    self._X_model = self._X_model_numpy
    if hasjax:
      self._X_model = self._X_model_jax

    self._i_crop()

  def _crop(self,lkl,**options):
    if "crop" in options:
      r_bin_min_tt = self.bin_min_tt
      r_bin_max_tt = self.bin_max_tt
      r_bin_min_te = self.bin_min_te
      r_bin_max_te = self.bin_max_te
      r_bin_min_ee = self.bin_min_ee
      r_bin_max_ee = self.bin_max_ee
      regex = re.compile("(?i)^(no|only|crop|notch)?\\s*([T|E|B][T|E|B])(\\s+(\\d+)x(\\d+))?(\\s+(-?\\d+)?\\s+(-?\\d+)?)?(?:\\s+(strict|lax|half))?")
      crop_cmd = options["crop"]
      if isinstance(crop_cmd,(tuple,list)):
        crop_cmd = "\n".join(crop_cmd)
      crop_cmd = crop_cmd.split("\n")
      for cro in crop_cmd:
        m = regex.match(cro.strip())
        ell = nm.arange(self.plmin,self.plmax+1)
        if m:
          only = False
          no = False
          if m.group(1) is not None:
            if m.group(1).lower()=="only":
              only = True
            if m.group(1).lower() == ("no"):
              no = True
          strictness = 1
          if m.group(9) is not None:
            if m.group(9).lower()=="lax":
              strictness = 0
            elif m.group(9).lower()=="strict":
              strictness = 1
            else:
              strictness = 0.5
      
          kind = ["TEB".index(k) for k in m.group(2).upper()]
          if kind[1]<kind[0]:
            kind = (kind[1],kind[0])
          kind = "TEB"[kind[0]]+"TEB"[kind[1]]
          if m.group(6) is not None:
            cmin = int(m.group(7))
            cmax = int(m.group(8))
          else:
            cmin = self.plmin
            cmax = self.plmax
          if cmin<0:
            cmin=0
          if cmax<0:
            cmax=lmax
          lmsk = (ell>=cmin) * (ell<=cmax)
          lmsk = self.bns_0 @ lmsk
          if strictness<1:
            lmsk = lmsk>strictness
          else:
            lmsk = nm.abs(lmsk-1)<1e-8
          ok = nm.arange(len(lmsk))[lmsk]
          if kind=="TT":
            if no or nm.sum(ok)==0:
              r_bin_max_tt = r_bin_min_tt -1 
            else:
              r_bin_min_tt = max(r_bin_min_tt,ok[0]+1)
              r_bin_max_tt = min(r_bin_max_tt,ok[-1]+1)
              if only:
                r_bin_max_te = r_bin_min_te -1 
                r_bin_max_ee = r_bin_min_ee -1

          if kind=="TE":
            if no or nm.sum(ok)==0:
              r_bin_max_te = r_bin_min_te -1 
            else:
              r_bin_min_te = max(r_bin_min_te,ok[0]+1)
              r_bin_max_te = min(r_bin_max_te,ok[-1]+1)
              if only:
                r_bin_max_tt = r_bin_min_tt -1 
                r_bin_max_ee = r_bin_min_ee -1
            
          if kind=="EE":
            if no or nm.sum(ok)==0:
              r_bin_max_ee = r_bin_min_ee -1 
            else:
              r_bin_min_ee = max(r_bin_min_ee,ok[0]+1)
              r_bin_max_ee = min(r_bin_max_ee,ok[-1]+1)
              if only:
                r_bin_max_te = r_bin_min_te -1 
                r_bin_max_tt = r_bin_min_tt -1
            

      print("before crop : ",end="")
      if self.use_tt and self.bin_min_tt<=self.bin_max_tt:
        print("TT %d -> %d, "%(self.blmin[self.bin_min_tt-1]+self.plmin,self.blmax[self.bin_max_tt-1]+self.plmin),end="")
      else:
        print("TT disabled, ",end="")
      if self.use_te and self.bin_min_te<=self.bin_max_te:
        print("TE %d -> %d, "%(self.blmin[self.bin_min_te-1]+self.plmin,self.blmax[self.bin_max_te-1]+self.plmin),end="")
      else:
        print("TE disabled, ",end="")
      if self.use_ee and self.bin_min_ee<=self.bin_max_ee:
        print("EE %d -> %d"%(self.blmin[self.bin_min_ee-1]+self.plmin,self.blmax[self.bin_max_ee-1]+self.plmin),end="\n")
      else:
        print("EE disabled",end="\n")
      print("after crop  : ",end="")
      if self.use_tt and r_bin_min_tt<=r_bin_max_tt:
        print("TT %d -> %d, "%(self.blmin[r_bin_min_tt-1]+self.plmin,self.blmax[r_bin_max_tt-1]+self.plmin),end="")
      else:
        print("TT disabled, ",end="")
      if self.use_te and r_bin_min_te<=r_bin_max_te:
        print("TE %d -> %d, "%(self.blmin[r_bin_min_te-1]+self.plmin,self.blmax[r_bin_max_te-1]+self.plmin),end="")
      else:
        print("TE disabled, ",end="")
      if self.use_ee and r_bin_min_ee<=r_bin_max_ee:
        print("EE %d -> %d"%(self.blmin[r_bin_min_ee-1]+self.plmin,self.blmax[r_bin_max_ee-1]+self.plmin),end="\n")
      else:
        print("EE disabled",end="\n")

      self._i_crop(r_bin_min_tt,r_bin_max_tt,r_bin_min_te,r_bin_max_te,r_bin_min_ee,r_bin_max_ee)
    if hasjax:
      self._X_model_jax = jit(self._X_model_jax,static_argnums=(0,))
      self.__call__ = jit(self.__call__,static_argnums=(0,))
    
  def _X_model_jax(self,cls,nuisance_dict):
    X_model = jnp.zeros(self.bin_no,dtype=jnp64)
    cls = jnp64(cls)
    off = 0
    if self.use_tt:
      X_model = X_model.at[0:self.bin_max_tt-self.bin_min_tt+1].set(self.bns["tt"] @ cls[0,self.plmin:])
      off+=self.bin_max_tt-self.bin_min_tt+1
    if self.use_te:
      X_model = X_model.at[off:off+self.bin_max_te-self.bin_min_te+1].set(self.bns["te"] @ cls[3,self.plmin:])
      off+=self.bin_max_te-self.bin_min_te+1
    if self.use_ee:
      X_model = X_model.at[off:off+self.bin_max_ee-self.bin_min_ee+1].set(self.bns["ee"] @ cls[1,self.plmin:])
    X_model = X_model/jnp64(nuisance_dict["A_planck"]**2)
    return X_model

  def _X_model_numpy(self,cls,nuisance_dict):
    X_model = nm.zeros(self.bin_no)
    off = 0
    if self.use_tt:
      X_model[0:self.bin_max_tt-self.bin_min_tt+1] = (self.bns["tt"] @ cls[0,self.plmin:])
      off+=self.bin_max_tt-self.bin_min_tt+1
    if self.use_te:
      X_model[off:off+self.bin_max_te-self.bin_min_te+1] = (self.bns["te"] @ cls[3,self.plmin:])
      off+=self.bin_max_te-self.bin_min_te+1
    if self.use_ee:
      X_model[off:off+self.bin_max_ee-self.bin_min_ee+1] = (self.bns["ee"] @ cls[1,self.plmin:])
    X_model = X_model/(nuisance_dict["A_planck"]**2)
    return X_model

  def __call__(self,cls,nuisance_dict,chi2_mode=False):
    X_model = self._X_model(cls,nuisance_dict)
    Y = jnp64(self.X_data - X_model)
    return jnp64(-.5) * jnp.dot(Y,jnp.dot(self.inv_cov,Y))


