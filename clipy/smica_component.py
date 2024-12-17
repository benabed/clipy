
from . import *



def components_from_file(grp):
  lmax = grp["lmax"]
  lmin = grp["lmin"]
  has_cl = grp["has_cl"]
  nT = grp["m_channel_T"]
  nP = grp["m_channel_P"]
  cmp = []
  for icmp in range(1,grp["n_component"]):
    typ = grp["component_%d/component_type"%icmp]
    cmp += [globals()["smica_%s"%typ](grp["component_%d"%icmp],lmin,lmax,has_cl,nT,nP)]
  return cmp

def try_float(x):
  try:
    return jnp64(x)
  except:
    return x

class smica_component:
  def __init__(self,hgrp,lmin,lmax,has_cl,nT,nP,mul=False):
    self.component_type = hgrp["component_type"]
    if "lmin" in hgrp:
      self.lmin = hgrp["lmin"]
      self.lmax = hgrp["lmax"]
      if lmin!=self.lmin: raise clik_emul_error("lmin mismatch")
      if lmax!=self.lmax: raise clik_emul_error("lmax mismatch")
    else:
      self.lmin=lmin
      self.lmax=lmax

    self.ell = jnp.arange(self.lmin,self.lmax+1)
    self.llp1 = self.ell*(self.ell+1)/jnp.array(2*jnp.pi,dtype=jnp64)

    self.has_cl = has_cl
    self.nT = nT
    self.nP = nP
    self.mul = mul
    self.m = nT*self.has_cl[0]+nP*(self.has_cl[1]+self.has_cl[2])
    
    self.defaults = {}
    self.rename = {}
    self.defdir = {}
    self.pars_remove =[]

    if "keys" in hgrp:
      self.varpar = [v for v in split_cldf_namelist(hgrp["keys"])]
      default = [v for v in hgrp["defaults"].replace("\0"," ").split()]
      value = [try_float(v) for v in hgrp["values"].replace("\0"," ").split() ]
      self.defdir = dict(list(zip(default,value)))
      self.frq = hgrp["dfreq"][:self.m]
      try:
        rename_from = [v for v in hgrp["rename_from"].replace("\0"," ").split()]
        rename_to = [v for v in hgrp["rename_to"].replace("\0"," ").split()]
        self.rename = dict(list(zip(rename_from,rename_to)))
      except Exception as e:
        self.rename = {}
      try:
        self.color = hgrp["color"][:]
      except Exception as e:
        self.color = jnp.array([1.]*len(self.frq),dtype=jnp64)    
      self.color = jnp.array(self.color.astype(jnp64))
      try:
        self.data = hgrp["template"][:]
        self.data = jnp.array(self.data.astype(jnp64),dtype=jnp64)
      except Exception as e:
        self.data = None
    
  def set_defaults(self,defaults):
    #print(self.defdir,self.rename)
    for k in self.defdir:
      nk = k
      if k in self.rename:
        nk = self.rename[k]

      defaults[nk] = self.defdir[k]
    return defaults

  def _build_pars(self,pars):
    if isinstance(pars,(list,tuple,nm.ndarray,jnp.ndarray)):
      if isinstance(pars[0],(list,tuple,nm.ndarray,jnp.ndarray)):
        return [self._build_pars(p) for p in pars]
    cur = {}
    cur.update(self.defaults)
    if isinstance(pars,dict):
      for k in pars:
        nk = k
        if k in self.rename:
          nk = self.rename[k]
        cur[nk] = pars[k]
    else:
      for i,k in enumerate(self.varpar):
        nk = k
        if k in self.rename:
          nk = self.rename[k]
        cur[nk] = pars[i]
    for r in self.pars_remove:
      if r in cur:
        del(cur[r])
    return cur

  @partial(jit, static_argnums=(0,))
  def compute_component(self,pars,shape=None):
    #print(pars)
    cur = self._build_pars(pars)
    #print(cur)
    if isinstance(cur,list):
      return jnp.array([self._compute_component(c,shape) for c in cur],dtype=jnp64)
    else:
      return self._compute_component(cur,shape)

  def bins_mnp(self,cls,bins=None):
    assert hasjax==False
    if bins == None:
      return cls
    if isinstance(bins,jnp.ndarray) and len(bins.shape)==2:
      nb = bins.shape[0]/jnp.sum(self.has_cl)
      ls = self.lmax+1-self.lmin
      ncls = nm.zeros((nb,self.m,self.m))
      for i in range(self.m):
        ncls[:,i,i] = bins[:nb,:ls] @ cls[:,i,i]
        for j in range(i,self.m):
          j+=i
          ncls[:,i,j] = bins[:nb,:ls] @ cls[:,i,j]
          ncls[:,j,i] = ncls[:,i,j]
      return jnp.array(ncls,dtype=jnp64)
    else:
      blmin = bins[0] 
      blmax = bins[1]
      b_ws = bins[2]
      nb = int(len(blmin)/jnp.sum(self.has_cl))
      ncls = nm.zeros((nb,self.m,self.m))
      for b in range(nb):
        ncls[b] = jnp.sum(cls[blmin[b]:blmax[b]+1] * (b_ws[blmin[b]:blmax[b]+1][:,jnp.newaxis,jnp.newaxis]),axis=0)
      return jnp.array(ncls,dtype=jnp64)
  
  def bins(self,cls,bins=None):
    if bins == None:
      return cls
    if isinstance(bins,jnp.ndarray) and len(bins.shape)==2:
      nb = bins.shape[0]/jnp.sum(self.has_cl)
      ls = self.lmax+1-self.lmin
      ncls = nm.zeros((nb,self.m,self.m))
      for i in range(self.m):
        ncls[:,i,i] = bins[:nb,:ls] @ cls[:,i,i]
        for j in range(i,self.m):
          j+=i
          ncls[:,i,j] = bins[:nb,:ls] @ cls[:,i,j]
          ncls[:,j,i] = ncls[:,i,j]
      return jnp.array(ncls,dtype=jnp64)
    else:
      blmin = bins[0] 
      blmax = bins[1]
      b_ws = bins[2]
      nb = int(len(blmin)/nm.sum(self.has_cl))
      ncls = nm.zeros((nb,self.m,self.m))
      for b in range(nb):
        ncls[b] = jnp.sum(cls[blmin[b]:blmax[b]+1] * (b_ws[blmin[b]:blmax[b]+1][:,jnp.newaxis,jnp.newaxis]),axis=0)
      return jnp.array(ncls,dtype=jnp64)

  @partial(jit, static_argnums=(0,))
  def apply(self,pars,cls,bins=None):
    if self.mul:
      cls_cmp = self.compute_component(pars,shape=cls.shape)
      return cls*cls_cmp
    else:
      cls_cmp = self.compute_component(pars)
      if bins is not None:
        cls_cmp = jnp.tensordot(bins,cls_cmp,axes=(1,0))
      return cls+cls_cmp

_frq = {100:0,143:1,217:2,353:3}

class smica_gcib(smica_component):
  def __init__(self,hgrp,lmin,lmax,has_cl,nT,nP):
    super(smica_gcib,self).__init__(hgrp,lmin,lmax,has_cl,nT,nP)

    defaults = {}
    defaults["gib_index"]     = -1.3
    defaults["gib_index_ref"] = -1.3
    defaults["gib_muK_MJ-2sr_100"] = 4096.68168783/1e6
    defaults["gib_muK_MJ-2sr_143"] = 2690.05218701/1e6
    defaults["gib_muK_MJ-2sr_217"] = 2067.43988919/1e6
    defaults["gib_muK_MJ-2sr_353"] = 3478.86588972/1e6
    defaults["gib_rigid"] = 217
    defaults["gib_l_pivot"] = 3000

    defaults = self.set_defaults(defaults)
    self.defaults = defaults

    base_template = self.data[:(10001*4*4)]*1
    base_template = jnp.reshape(base_template,(10001,4,4))

    conv = jnp.array([jnp64(defaults["gib_muK_MJ-2sr_%d"%(int(f))]) for f in self.frq],dtype=jnp64)
    A_names = []
    A_rigid = ""
    assert defaults["gib_rigid"]

    A_rigid = "A_gib_%d"%int(defaults["gib_rigid"])
    idx_rigid = [int(f) for f in self.frq].index(int(defaults["gib_rigid"]))
    idx0_rigid = _frq[int(defaults["gib_rigid"])]
    defaults[A_rigid] = 70

    l_pivot = int(self.defaults["gib_l_pivot"])
    self.l_pivot = l_pivot
    self.index_ref = jnp64(self.defaults["gib_index_ref"])
    llp1_pivot = jnp64(l_pivot*(l_pivot+1)/2./jnp.pi)
    self.template = nm.zeros((self.lmax-self.lmin+1,self.m,self.m))
    self.A0 = nm.zeros((self.m,self.m))
    for i,frq0 in enumerate(self.frq):
      idx0 = _frq[int(frq0)]
      self.template[:,i,i] = base_template[self.lmin:self.lmax+1,idx0,idx0]
      self.A0[i,i] = jnp64(1.)/(base_template[l_pivot,idx0_rigid,idx0_rigid]) *conv[i]*conv[i]/conv[idx_rigid]/conv[idx_rigid]/llp1_pivot
      if self.color is not None:
        self.template[:,i,i] = jnp64(self.template[:,i,i])*jnp64(self.color[i]*self.color[i])
      for j,frq1 in enumerate(self.frq[i+1:]):
        j+=i+1
        idx1 = _frq[int(frq1)]
        self.template[:,i,j] = base_template[self.lmin:self.lmax+1,idx0,idx1]
        self.template[:,j,i] = base_template[self.lmin:self.lmax+1,idx1,idx0]
        if self.color is not None:
          self.template[:,i,j] = jnp64(self.template[:,i,j])* jnp64(self.color[i]*self.color[j])
          self.template[:,j,i] = jnp64(self.template[:,j,i])* jnp64(self.color[i]*self.color[j])
        self.A0[i,j] = jnp64(1.)/(base_template[l_pivot,idx0_rigid,idx0_rigid]) *conv[i]*conv[j]/conv[idx_rigid]/conv[idx_rigid]/llp1_pivot
        self.A0[j,i] = jnp64(1.)/(base_template[l_pivot,idx0_rigid,idx0_rigid]) *conv[i]*conv[j]/conv[idx_rigid]/conv[idx_rigid]/llp1_pivot
          
    self.template = jnp.array(self.template,dtype=jnp64)
    self.A0 = jnp.array(self.A0,dtype=jnp64)
    self.A_names = A_rigid
    self.pars_remove = ["gib_rigid","gibsz_rigid"]
    
  @partial(jit, static_argnums=(0,))
  def _compute_component(self,pars,shape=None):
    index = jnp.array(pars["gib_index"],dtype=jnp64)    
    l_pivot = int(self.defaults["gib_l_pivot"])
    res = (self.ell[:,jnp.newaxis,jnp.newaxis]*jnp64(1./l_pivot))**(index-self.index_ref) * self.template * (pars[self.A_names]) * self.A0[jnp.newaxis,:,:]
    return res

def sz_spectrum(nu, nu0):
  # This gives the SZ spectrum in \delta_T (CMB)
  # normalized at nu0
  x0 = nu0/56.78
  x  = nu /56.78
  res = jnp64(2.0)-jnp64(x/2.0)/jnp.tanh(jnp64(x/2.0))
  res0 = jnp64(2.0)-jnp64(x0/2.0)/jnp.tanh(jnp64(x0/2.0))
  return (res/res0)


class smica_gibXsz(smica_component):
  def __init__(self,hgrp,lmin,lmax,has_cl,nT,nP):
    super(smica_gibXsz,self).__init__(hgrp,lmin,lmax,has_cl,nT,nP)
    defaults = {}
    defaults["sz_color_143_to_143"]=0.975
    defaults["sz_color_100_to_143"]=0.981
    defaults["gibXsz_100_to_217"]=0.022
    defaults["gibXsz_143_to_217"]=0.094
    defaults["gibXsz_353_to_217"]=46.8
    defaults["no_szxcib_100"]=1

    defaults = self.set_defaults(defaults)

    if int(defaults["no_szxcib_100"]):
      defaults["gibXsz_100_to_217"] = 0

    self.defaults=defaults
    base_template = jnp.concatenate((jnp.array([0.,0.],dtype=jnp64),self.data))
    self.template = nm.zeros((self.lmax-self.lmin+1,self.m,self.m))
    
    self.nrm_szXcib = nm.zeros((self.m,self.m))
    for i,frq0 in enumerate(self.frq):
      self.template[:,i,i] = base_template[self.lmin:self.lmax+1] * self.color[i] * self.color[i] / self.llp1
      self.nrm_szXcib[i,i] = jnp64(defaults.get("sz_color_%d_to_143"%(int(frq0)),1.)*defaults.get("gibXsz_%d_to_217"%(int(frq0)),1.))*sz_spectrum(frq0,143.)
      for j,frq1 in enumerate(self.frq[i+1:]):
        j+=i+1
        self.template[:,i,j] = base_template[self.lmin:self.lmax+1] * self.color[i] * self.color[j] / self.llp1
        self.template[:,j,i] = self.template[:,i,j]
        self.nrm_szXcib[i,j] = jnp64(defaults.get("sz_color_%d_to_143"%(int(frq0)),1.)*defaults.get("gibXsz_%d_to_217"%(int(frq1)),1.))*sz_spectrum(frq0,143.)
        self.nrm_szXcib[j,i] = jnp64(defaults.get("sz_color_%d_to_143"%(int(frq1)),1.)*defaults.get("gibXsz_%d_to_217"%(int(frq0)),1.))*sz_spectrum(frq1,143.)
    self.nrm_szXcib = nm.sqrt(self.nrm_szXcib) + nm.sqrt(self.nrm_szXcib.T)
    self.nrm_szXcib = jnp.array(self.nrm_szXcib,dtype=jnp64)

    self.pars_remove = ["cib_rigid","cibsz_rigid"]
    
    self.template = jnp.array(self.template,dtype=jnp64)
    
  @partial(jit, static_argnums=(0,))
  def _compute_component(self,pars,shape=None):
    a_cib = jnp64(pars["A_cib_217"])
    #a_cib=43.
    a_sz = jnp64(pars["A_sz"])
    xi_sz_cib =  jnp64(pars["xi_sz_cib"])
    A = - xi_sz_cib * (a_sz*a_cib)**jnp64(.5) * self.nrm_szXcib
    return self.template * A[jnp.newaxis,:,:]



class smica_sz(smica_component):
  def __init__(self,hgrp,lmin,lmax,has_cl,nT,nP):
    super(smica_sz,self).__init__(hgrp,lmin,lmax,has_cl,nT,nP)
    defaults = {}
    defaults["sz_color_143_to_143"]=0.975
    defaults["sz_color_100_to_143"]=0.981

    defaults = self.set_defaults(defaults)
    self.defaults = defaults
      
    base_template = jnp.concatenate((jnp.array([0,0],dtype=jnp64),self.data))
    self.template = nm.zeros((self.lmax-self.lmin+1,self.m,self.m))
    
    self.nrm_szXcib = jnp.zeros((self.m,self.m),dtype=jnp64)
    for i,frq0 in enumerate(self.frq):
      self.template[:,i,i] = base_template[self.lmin:self.lmax+1] * self.color[i] * self.color[i] / self.llp1 * jnp64(defaults.get("sz_color_%d_to_143"%(int(frq0)),1.)) *jnp64(defaults.get("sz_color_%d_to_143"%(int(frq0)),1.))*sz_spectrum(frq0,143.)*sz_spectrum(frq0,143.)
      for j,frq1 in enumerate(self.frq[i+1:]):
        j+=i+1
        self.template[:,i,j] = base_template[self.lmin:self.lmax+1] * self.color[i] * self.color[j] / self.llp1 * jnp64(defaults.get("sz_color_%d_to_143"%(int(frq0)),1.)) *jnp64(defaults.get("sz_color_%d_to_143"%(int(frq1)),1.))*sz_spectrum(frq0,143.)*sz_spectrum(frq1,143.)
        self.template[:,j,i] = base_template[self.lmin:self.lmax+1] * self.color[i] * self.color[j] / self.llp1 * jnp64(defaults.get("sz_color_%d_to_143"%(int(frq0)),1.)) *jnp64(defaults.get("sz_color_%d_to_143"%(int(frq1)),1.))*sz_spectrum(frq0,143.)*sz_spectrum(frq1,143.)
    self.template = jnp.array(self.template,dtype=jnp64)
    self.pars_remove = ["cib_rigid","cibsz_rigid","sz_color_143_to_143","sz_color_100_to_143"]
    

  @partial(jit, static_argnums=(0,))
  def _compute_component(self,pars,shape=None):
    return self.template * (pars["A_sz"])
  
class smica_ksz(smica_component):
  def __init__(self,hgrp,lmin,lmax,has_cl,nT,nP):
    super(smica_ksz,self).__init__(hgrp,lmin,lmax,has_cl,nT,nP)
    defaults = {}

    defaults = self.set_defaults(defaults)
    self.defaults = defaults

    self.template = nm.zeros((self.lmax-self.lmin+1,self.m,self.m))
    self.template[:,:3,:3] = (self.data[self.lmin:self.lmax+1]/self.data[3000]/self.llp1)[:,jnp.newaxis,jnp.newaxis] * jnp.outer(self.color,self.color)[jnp.newaxis,:,:]
    self.template = jnp.array(self.template,dtype=jnp64)
    self.pars_remove = ["cib_rigid","cibsz_rigid"]
    
  @partial(jit, static_argnums=(0,))
  def _compute_component(self,pars,shape=None):
    return self.template * (pars["ksz_norm"])
  
class smica_pointsource(smica_component):
  def __init__(self,hgrp,lmin,lmax,has_cl,nT,nP):
    super(smica_pointsource,self).__init__(hgrp,lmin,lmax,has_cl,nT,nP)
    defaults = {}
    defaults["ps_l_pivot"]=3000
    for i,frq0 in enumerate(self.frq):
      for frq1 in self.frq[i:]:
        defaults["ps_A_%d_%d"%(int(frq0),int(frq1))]=0
    defaults = self.set_defaults(defaults)
    self.defaults = defaults
    self.pars_remove = ["cib_rigid","cibsz_rigid"]
    self._compute_component = self._compute_component_mnp
    if hasjax:
      self._compute_component = self._compute_component_jax


  @partial(jit, static_argnums=(0,))    
  def _compute_component_jax(self,pars,shape=None):
    A = jnp.zeros((self.m,self.m),dtype=jnp64)
    l_pivot = (pars["ps_l_pivot"])
    nrm = jnp64(l_pivot*(l_pivot+1)/2/jnp.pi)

    for i,frq0 in enumerate(self.frq):
      for j,frq1 in enumerate(self.frq[i:]):
        j+=i
        A = A.at[i,j].set( (pars["ps_A_%d_%d"%(int(frq0),int(frq1))]/nrm*self.color[i]*self.color[j]))
        A = A.at[j,i].set( A[i,j])
    return jnp.ones((self.lmax-self.lmin+1,self.m,self.m),dtype=jnp64)*A[jnp.newaxis,:,:]
  
  def _compute_component_mnp(self,pars,shape=None):
    assert hasjax==False

    A = nm.zeros((self.m,self.m))
    l_pivot = int(pars["ps_l_pivot"])
    nrm = jnp64(l_pivot*(l_pivot+1)/2/jnp.pi)

    for i,frq0 in enumerate(self.frq):
      for j,frq1 in enumerate(self.frq[i:]):
        j+=i
        A[i,j] = jnp64(pars["ps_A_%d_%d"%(int(frq0),int(frq1))]/nrm*self.color[i]*self.color[j])
        A[j,i] = A[i,j]
    return jnp.ones((self.lmax-self.lmin+1,self.m,self.m))*A[jnp.newaxis,:,:]

class smica_cnoise(smica_component):
  def __init__(self,hgrp,lmin,lmax,has_cl,nT,nP):
    super(smica_cnoise,self).__init__(hgrp,lmin,lmax,has_cl,nT,nP)
    defaults = {}
    defaults["cnoise_l_pivot"]=2000
    defaults["cnoise_abs"]=0

    for i,frq0 in enumerate(self.frq):
      teb0 = 0
      if i>=self.nT*self.has_cl[0]:
        teb0+=1
      if i>=self.nT*self.has_cl[0]+self.nP*self.has_cl[1]:
        teb0+=1
      for j,frq1 in enumerate(self.frq[i:]):
        j+=i
        teb1 = 0
        if j>=self.nT*self.has_cl[0]:
          teb1+=1
        if j>=self.nT*self.has_cl[0]+self.nP*self.has_cl[1]:
          teb1+=1
        cfrq0 = frq0
        cfrq1 = frq1
        cteb0 = teb0
        cteb1 = teb1
        if cteb0>cteb1:
          cfrq0 = frq1
          cfrq1 = frq0
          cteb0 = teb1
          cteb1 = teb0
        if cfrq0>cfrq1:
          cfrq1,cfrq0 = cfrq0,cfrq1
        defaults["A_cnoise_%d_%d_%s%s"%(int(cfrq0),int(cfrq1),"TEB"[cteb0],"TEB"[cteb1])]=0.
        defaults["A_cnoise_%d_%d_%s%s"%(int(cfrq0),int(cfrq1),"TEB"[cteb0],"TEB"[cteb1])]=0.
    
    defaults = self.set_defaults(defaults)
    # check that amplitudes are not strings... Jaxerie....
    import re
    for p in defaults:
      if re.match("A_cnoise_[0-9]+_[0-9]+_[TEB]+",p):
        defaults[p] = jnp64(defaults[p])
    self.defaults = defaults
    base_template = jnp.reshape(self.data,(3001,12,12))
    self.template = nm.zeros((self.lmax-self.lmin+1,self.m,self.m))
    
    for i,frq0 in enumerate(self.frq):
      teb0 = 0
      if i>=self.nT*self.has_cl[0]:
        teb0+=1
      if i>=self.nT*self.has_cl[0]+self.nP*has_cl[1]:
        teb0+=1
      for j,frq1 in enumerate(self.frq[i:]):
        j+=i
        #print(i,j,self.frq)
        teb1 = 0
        if j>=self.nT*self.has_cl[0]:
          teb1+=1
        if j>=self.nT*self.has_cl[0]+self.nP*has_cl[1]:
          teb1+=1
        self.template[:,i,j] = base_template[self.lmin:self.lmax+1,_frq[int(frq0)]+4*teb0,_frq[int(frq1)]+4*teb1]*self.color[i]*self.color[j]
        self.template[:,j,i] = self.template[:,i,j]
    self.template = jnp.array(self.template,dtype=jnp64)
    if  int(self.defaults["cnoise_abs"]):
      l_pivot = int(self.defaults["cnoise_l_pivot"])
      self.template/= self.template[l_pivot-self.lmin][jnp.newaxis,:,:] * l_pivot * (l_pivot + 1)/jnp64(2.*jnp.pi)
    self.pars_remove = ["cib_rigid","cibsz_rigid","cnoise_abs","cnoise_l_pivot"]
    self._compute_component = self._compute_component_mnp
    if hasjax:
      self._compute_component = self._compute_component_jax

  @partial(jit, static_argnums=(0,))
  def _compute_component_jax(self,pars,shape=None):
    A = jnp.zeros((self.m,self.m),dtype=jnp64)
    for i,frq0 in enumerate(self.frq):
      teb0 = 0
      if i>=self.nT*self.has_cl[0]:
        teb0+=1
      if i>=self.nT*self.has_cl[0]+self.nP*self.has_cl[1]:
        teb0+=1
      for j,frq1 in enumerate(self.frq[i:]):
        j+=i
        teb1 = 0
        if j>=self.nT*self.has_cl[0]:
          teb1+=1
        if j>=self.nT*self.has_cl[0]+self.nP*self.has_cl[1]:
          teb1+=1
        cfrq0 = frq0
        cfrq1 = frq1
        cteb0 = teb0
        cteb1 = teb1
        if cteb0>cteb1:
          cfrq0 = frq1
          cfrq1 = frq0
          cteb0 = teb1
          cteb1 = teb0
        if cfrq0>cfrq1:
          cfrq1,cfrq0 = cfrq0,cfrq1
        A = A.at[i,j].set((pars["A_cnoise_%d_%d_%s%s"%(int(cfrq0),int(cfrq1),"TEB"[cteb0],"TEB"[cteb1])]))
        A = A.at[j,i].set((pars["A_cnoise_%d_%d_%s%s"%(int(cfrq0),int(cfrq1),"TEB"[cteb0],"TEB"[cteb1])]))
    return self.template * A[jnp.newaxis,:,:]

  def _compute_component_mnp(self,pars,shape=None):
    assert hasjax==False
    A = nm.zeros((self.m,self.m))
    for i,frq0 in enumerate(self.frq):
      teb0 = 0
      if i>=self.nT*self.has_cl[0]:
        teb0+=1
      if i>=self.nT*self.has_cl[0]+self.nP*self.has_cl[1]:
        teb0+=1
      for j,frq1 in enumerate(self.frq[i:]):
        j+=i
        teb1 = 0
        if j>=self.nT*self.has_cl[0]:
          teb1+=1
        if j>=self.nT*self.has_cl[0]+self.nP*self.has_cl[1]:
          teb1+=1
        cfrq0 = frq0
        cfrq1 = frq1
        cteb0 = teb0
        cteb1 = teb1
        if cteb0>cteb1:
          cfrq0 = frq1
          cfrq1 = frq0
          cteb0 = teb1
          cteb1 = teb0
        if cfrq0>cfrq1:
          cfrq1,cfrq0 = cfrq0,cfrq1
        A[i,j] = jnp64(pars["A_cnoise_%d_%d_%s%s"%(int(cfrq0),int(cfrq1),"TEB"[cteb0],"TEB"[cteb1])])
        A[j,i] = jnp64(pars["A_cnoise_%d_%d_%s%s"%(int(cfrq0),int(cfrq1),"TEB"[cteb0],"TEB"[cteb1])])
    return self.template * A[jnp.newaxis,:,:]

class smica_powerlaw_free_emissivity_XX(smica_component):
  _kind = ["TT","EE","BB","TE","TB","EB"]

  def __init__(self,hgrp,lmin,lmax,has_cl,nT,nP):
    super(smica_powerlaw_free_emissivity_XX,self).__init__(hgrp,lmin,lmax,has_cl,nT,nP)
    defaults = {}
    defaults["pwfe_XX_l_pivot"]=500
    defaults["pwfe_XX_l2_norm"]=1
    defaults["pwfe_XX_index"]=0
    defaults = self.set_defaults(defaults)
    self.defaults = defaults
    #print (defaults)
    kind = int(defaults["pwfe_XX_kind"])

    Alist = {}
    match self._kind[kind]:
      case "TT":
        bi = 0
        bj = 0
        ei = self.nT
        ej = self.nT

      case "EE":
        bi = self.has_cl[0]*self.nT
        ei = self.has_cl[0]*self.nT+self.nP
        bj = self.has_cl[0]*self.nT
        ej = self.has_cl[0]*self.nT+self.nP
      case "BB":
        bi = self.has_cl[0]*self.nT,self.has_cl[0]*self.nT+self.has_cl[1]*self.nP
        ei = self.has_cl[0]*self.nT,self.has_cl[0]*self.nT+self.has_cl[1]*self.nP+self.nP
        bj = self.has_cl[0]*self.nT,self.has_cl[0]*self.nT+self.has_cl[1]*self.nP
        ej = self.has_cl[0]*self.nT,self.has_cl[0]*self.nT+self.has_cl[1]*self.nP+self.nP
      case "TE":
        bi = 0
        ei = self.nT
        bj = self.nT
        ej = self.nT+self.nP

      case "TB":
        bi = 0
        ei = self.nT
        bj = self.nT+self.nP*self.has_cl[1]
        ej = self.nT+self.nP+self.nP*self.has_cl[1]
      case "EB":
        bi = self.nT*self.has_cl[0]
        ei = self.nP+self.nT*self.has_cl[0]
        bj = self.nT*self.has_cl[0]+self.nP+self.nP*self.has_cl[1]
        ej = self.nT*self.has_cl[0]+self.nP+self.nP+self.nP*self.has_cl[1]

    for i,frq0 in enumerate(self.frq[bi:ei]):
      for j,frq1 in enumerate(self.frq[bj:ej]):
        if frq0==frq1:
          Alist["pwfe_XX_A_%d"%(int(frq0))]=[(i+bi,j+bj),(j+bj,i+bi)]
        if frq0<frq1 and kind<3:
          Alist["pwfe_XX_A_%d_%d"%(int(frq0),int(frq1))]=[(i+bi,j+bj),(j+bj,i+bi)]
        if frq0>frq1 and kind<3:
          Alist["pwfe_XX_A_%d_%d"%(int(frq1),int(frq0))]=[(i+bi,j+bj),(j+bj,i+bi)]
        if frq0<frq1 and kind>2:
          Alist["pwfe_XX_A_%d_%d"%(int(frq0),int(frq1))]=[(i+bi,j+bj),(j+bj,i+bi)]
          if i+bj<ej and j+bi<ei:
            Alist["pwfe_XX_A_%d_%d"%(int(frq0),int(frq1))]+=[(i+bj,j+bi),(j+bi,i+bj)]
    self.Alist = Alist
    self.l_pivot = int(defaults["pwfe_XX_l_pivot"])
    if int(defaults["pwfe_XX_l2_norm"]):
      nrm = self.l_pivot*(self.l_pivot+1)/2/jnp.pi
    else:
      nrm = 1
    self.nrm = jnp64(nrm)

    self.pars_remove = ["cib_rigid","cibsz_rigid","pwfe_XX_l2_norm","pwfe_XX_l_pivot","pwfe_XX_kind"]
    self._compute_component = self._compute_component_mnp
    if hasjax:
      self._compute_component = self._compute_component_jax

  @partial(jit, static_argnums=(0,))
  def _compute_component_jax(self,pars,shape=None):
    index = (pars["pwfe_XX_index"])

    tmpl = (self.ell/jnp64(self.l_pivot)) ** index / self.nrm

    A = jnp.zeros((self.m,self.m),dtype=jnp64)
    for k in self.Alist:
      v = (pars[k])
      for i,j in self.Alist[k]:
        A = A.at[i,j].set(v * self.color[i] * self.color[j])
    
    return tmpl[:,jnp.newaxis,jnp.newaxis] * A[jnp.newaxis,:,:]
  def _compute_component_mnp(self,pars,shape=None):
    assert hasjax==False
    index = jnp64(pars["pwfe_XX_index"])

    tmpl = (self.ell/jnp64(self.l_pivot)) ** index / self.nrm

    A = nm.zeros((self.m,self.m))
    for k in self.Alist:
      v = jnp64(pars[k])
      for i,j in self.Alist[k]:
        A[i,j] = v * self.color[i] * self.color[j]
    
    return tmpl[:,jnp.newaxis,jnp.newaxis] * A[jnp.newaxis,:,:]
    



class smica_calTP(smica_component):
  def calfunc(self,x):
    return jnp.exp(x)

  def __init__(self,hgrp,lmin,lmax,has_cl,nT,nP):
    super(smica_calTP,self).__init__(hgrp,lmin,lmax,has_cl,nT,nP,mul=True)
    names = split_cldf_namelist(hgrp["names"])
    self.im = hgrp["im"].astype(jnp.int32)
    if "w" in hgrp:
      self.w = jnp.array(hgrp["w"],dtype=jnp64)
      self.other = hgrp["other"].astype(jnp.int32)
      self.w = jnp.reshape(self.w,(self.m,self.m,2))
      self.other.shape=(self.m,self.m,2)
      
    else: 
      self.w = nm.zeros((self.m,self.m,2))
      self.w[:,:] = [1,0]
      self.other =  nm.indices((m,m)).T[:,:,::-1]

    self.varpar = names
    self._compute_component = self._compute_component_mnp
    if hasjax:
      self._compute_component = self._compute_component_jax

  @partial(jit, static_argnums=(0,))
  def _compute_component_jax(self,pars,shape=None):
    vec = jnp.ones(self.m,dtype=jnp64)
    vals = jnp.array([(pars[k]) for k in self.varpar],dtype=jnp64)
    evals = self.calfunc(vals)
    #print(self.im)
    #print(self.im.dtype)
    vec = vec.at[self.im[self.im<self.nT]].set(evals[self.im<self.nT])
    vec = vec.at[self.im[self.im>=self.nT]].set(evals[self.im>=self.nT])
    if self.has_cl[2]:
      vec = vec.at[self.im[self.im>=self.nT]+self.nP].set(evals[self.im>=self.nT])
    gmat = jnp.outer(vec,vec)
    gmat_prime = gmat[self.other[:,:,0],self.other[:,:,1]]
    gmat = self.w[:,:,0] * gmat + self.w[:,:,1] * gmat_prime
    res = jnp.array(gmat[jnp.newaxis,:,:],dtype=jnp64)
    #if shape==None:
    #  shape = (self.lmax-self.lmin+1,self.m,self.m)
    #res = jnp.ones(shape)*res
    return res

  def _compute_component_mnp(self,pars,shape=None):
    assert hasjax==False
    vec = jnp.ones(self.m)
    vals = jnp.array([jnp64(pars[k]) for k in self.varpar],dtype=jnp64)
    evals = self.calfunc(vals)
    vec[self.im[self.im<self.nT]] = (evals[self.im<self.nT])
    vec[self.im[self.im>=self.nT]] = (evals[self.im>=self.nT])
    if self.has_cl[2]:
      vec = vec.at[self.im[self.im>=self.nT]+self.nP].set(evals[self.im>=self.nT])
    gmat = jnp.outer(vec,vec)
    gmat_prime = gmat[self.other[:,:,0],self.other[:,:,1]]
    gmat = self.w[:,:,0] * gmat + self.w[:,:,1] * gmat_prime
    res = jnp.array(gmat[jnp.newaxis,:,:],dtype=jnp64)
    if shape==None:
      shape = (self.lmax-self.lmin+1,self.m,self.m)
    res = jnp.ones(shape)*res
    return res

class smica_icalTP(smica_calTP):
  def calfunc(self,x):
    return 1./jnp.sqrt(x)
  
class smica_totcal(smica_component):
  def __init__(self,hgrp,lmin,lmax,has_cl,nT,nP):
    super(smica_totcal,self).__init__(hgrp,lmin,lmax,has_cl,nT,nP,mul=True)
    names = split_cldf_namelist(hgrp["calname"])
    self.varpar = names
    self._compute_component = self._compute_component_mnp
    if hasjax:
      self._compute_component = self._compute_component_jax


  @partial(jit, static_argnums=(0,))
  def _compute_component_jax(self,pars,shape=None):
    #print("---totcal---")
    res = jnp64(1.)/jnp64((pars[self.varpar[0]]))**jnp64(2)
    #print("---totcal---")
    
    #if shape==None:
    #  shape = (self.lmax-self.lmin+1,self.m,self.m)
    #res = jnp.ones(shape)*res
    return res

  def _compute_component_mnp(self,pars,shape=None):
    assert hasjax==False
    res = 1./jnp64(pars[self.varpar[0]])**2
    if shape==None:
      shape = (self.lmax-self.lmin+1,self.m,self.m)
    res = jnp.ones(shape)*res
    return res

class smica_totcalP(smica_component):
  def __init__(self,hgrp,lmin,lmax,has_cl,nT,nP):
    super(smica_totcalP,self).__init__(hgrp,lmin,lmax,has_cl,nT,nP,mul=True)
    names = split_cldf_namelist(hgrp["calnameP"])
    self.varpar = names
    self._compute_component = self._compute_component_mnp
    if hasjax:
      self._compute_component = self._compute_component_jax
  
  @partial(jit, static_argnums=(0,))
  def _compute_component_jax(self,pars,shape=None):
    calP = jnp64(1.)/jnp64(pars[self.varpar[0]])
    res = jnp.ones((self.m,self.m),dtype=jnp64)
    res = res.at[:self.nT,self.nT:].set(calP)
    res = res.at[self.nT:,:self.nT].set(calP)
    #res = res.at[self.nT:,self.nT:].set(calP**2)
    #if shape==None:
    #  shape = (self.lmax-self.lmin+1,self.m,self.m)
    #res = jnp.ones(shape)*res
    return res[nm.newaxis,:,:]

  def _compute_component_mnp(self,pars,shape=None):
    assert hasjax==False
    calP = 1./jnp64(pars[self.varpar[0]])
    res = jnp.ones((self.m,self.m))
    res[:self.nT,self.nT:]=(calP)
    res[self.nT:,:self.nT]=(calP)
    res[self.nT:,self.nT:]=(calP**2)
    if shape==None:
      shape = (self.lmax-self.lmin+1,self.m,self.m)
    res = jnp.ones(shape)*res
    return res

