from . import *

from . import smica_component as smcmp

from . import lkl 

class smica_lkl(lkl._clik_cmb):
  
  def __init__(self,lkl,**options):
    super().__init__(lkl,**options)

    self.hascl = lkl["has_cl"]
    self.lmin = int(lkl["lmin"])
    self.lmax = int(lkl["lmax"])
    self.ell = jnp.arange(self.lmin,self.lmax+1)
    
    self.mt = lkl["m_channel_T"]*self.hascl[0]
    self.me = lkl["m_channel_P"]*self.hascl[1]
    self.mb = lkl["m_channel_P"]*self.hascl[2]
    self.m = self.mt+self.me+self.mb
  
    self.nb = int(lkl["nbins"]/self.hascl.sum())
    self.rq_shape = (self.nb,self.m,self.m)

    # binning details
    self.blmin = jnp.array(lkl["bin_lmin"])
    self.blmax = jnp.array(lkl["bin_lmax"])
    self.b_ws =  jnp.array(lkl["bin_ws"],dtype=jnp64)
    
    # the binning matrix is also simply obtained this way (but using it is slower than using the binning details, 'cause it's full of zeros)
    if self.lkl["nbins"]==len(self.ell)*self.has_cl.sum():
      #unbinned !
      self.bins = None
      self.lm = self.ell
      self.bns = jnp.identity(self.lkl["nbins"],dtype=jnp64)
      self.bns_0 = None
    else:
      shape = (self.lkl["nbins"],len(self.ell)*self.has_cl.sum())
      bns = nm.zeros(shape)
      lc = 0
      for i in range(shape[0]):
        bsz = self.blmax[i]-self.blmin[i]+1
        bns[i,self.blmin[i]:self.blmax[i]+1] = self.b_ws[lc:lc+bsz]
        lc+=bsz
      self.bns = jnp.array(bns,dtype=jnp64)
      self.bns_0 = jnp.array(bns[:self.nb,:self.lmax+1-self.lmin],dtype=jnp64)
      #compute the binned ells
      self.lm = jnp.dot(self.bns[:self.nb,:self.lmax+1-self.lmin],self.ell)
      self.bins = (self.blmin,self.blmax,self.b_ws)

    self.acmb = lkl["A_cmb"]

    
    self.rqh = jnp.reshape(jnp.array(self.lkl["Rq_hat"],dtype=jnp64),self.rq_shape)
    
    self.nc = lkl["n_component"]
    self.cmp = smcmp.components_from_file(lkl)

    
    if self.lkl["criterion"]!="gauss":
      raise clik_emul_error("only gaussian likelihoods supported")

    self.ord = self.lkl["criterion_gauss_ordering"]
    self.msk = self.lkl["criterion_gauss_mask"]
    self.msk.shape=(-1,self.m,self.m)
    m2 = self.m*self.m
    self.oo =  self.get_ordering(False)
    self.rqh_f = self.rqh.flatten()[self.oo]
    
    #self.rq_idx = nm.zeros([lmax+1-lmin,self.rq_shape[1],self.rq_shape[1]])
    #cls_idx = nm.arange(6*(lmax+1))
    #cls_idx.shape=(6,-1)
    ##ICICICICICI
    #self.rq_idx[:,:mt,:mt] = ((cls_idx[0,lmin:lmax+1])[:,nm.newaxis,nm.newaxis])[:,:mt,:mt]
    #self.rq_idx[:,mt:mt+me,mt:mt+me] = ((cls_idx[1,lmin:lmax+1])[:,nm.newaxis,nm.newaxis])[:,mt:mt+me,mt:mt+me]
    #self.rq_idx[:,mt+me:mt+me+mb,mt+me:mt+me+mb] = ((cls_idx[2,lmin:lmax+1])[:,nm.newaxis,nm.newaxis])[:,mt+me:mt+me+mb,mt+me:mt+me+mb]
    #self.rq_idx[:,:mt,mt:mt+me] = ((cls_idx[3,lmin:lmax+1])[:,nm.newaxis,nm.newaxis])[:,:mt,mt:mt+me]
    #self.rq_idx[:,mt:mt+me,:mt] = ((cls_idx[3,lmin:lmax+1])[:,nm.newaxis,nm.newaxis])[:,mt:mt+me,:mt]
    #self.rq_idx[:,:mt,mt+me:mt+me+mb] = ((cls_idx[4,lmin:lmax+1])[:,nm.newaxis,nm.newaxis])[:,:mt,mt+me:mt+me+mb]
    #self.rq_idx[:,mt+me:mt+me+mb,:mt] = ((cls_idx[4,lmin:lmax+1])[:,nm.newaxis,nm.newaxis])[:,mt+me:mt+me+mb,:mt]
    #self.rq_idx[:,mt:mt+me,mt+me:mt+me+mb] = ((cls_idx[5,lmin:lmax+1])[:,nm.newaxis,nm.newaxis])[:,mt:mt+me,mt+me:mt+me+mb]
    #self.rq_idx[:,mt+me:mt+me+mb,mt:mt+me] = ((cls_idx[5,lmin:lmax+1])[:,nm.newaxis,nm.newaxis])[:,mt+me:mt+me+mb,mt:mt+me]

     
    self.siginv = jnp.array(self.lkl["criterion_gauss_mat"],dtype=jnp64)
    self.siginv = jnp.reshape(self.siginv,(len(self.oo),len(self.oo)))

    self.varpar = []
    for c in self.cmp:
      self.varpar += [p for p in c.varpar if p not in self.varpar]


    self._get_cmb_rq = self._get_cmb_rq_nmp
    if hasjax:
      self._get_cmb_rq = self._get_cmb_rq_jax

    self._parlen = len(self.varpar) + self.nd

 
  def get_ordering(self,jac=True,omsk=None):
    if omsk is None:
      msk = self.msk
    else:
      msk = omsk
    ord = self.ord
    m2 = self.m*self.m
    msk.shape = (-1,self.m,self.m)
    ordr =  jnp.concatenate([jnp.arange(self.nb)[msk[:,iv,jv]==1]*m2+iv*self.m+jv for iv,jv in zip(ord[::2],ord[1::2])])
    if jac==True:
      Jt = []
      ps = ordr//m2
      ii = (ordr%m2)//self.m
      jj = (ordr%m2)%self.m
      ii_T = ii < self.me
      jj_T = jj < self.me
      ii_E = (ii >= self.me) * (ii < self.mb)
      jj_E = (jj >= self.me) * (jj < self.mb)
      ii_B = (ii >= self.mb) * (ii < self.m)
      jj_B = (jj >= self.mb) * (jj < self.m)

      if self.hascl[0]:
        jac = jnp.array([(ps==i) * ii_T * jj_T for i in range(self.nb)],dtype=jnp.int)
        Jt += [jac]
      if self.hascl[1]:
        jac = jnp.array([(ps==i) * ii_E * jj_E for i in range(self.nb)],dtype=jnp.int)
        Jt += [jac]
      if self.hascl[2]:
        jac = jnp.array([(ps==i) * ii_B * jj_B for i in range(self.nb)],dtype=jnp.int)
        Jt += [jac]
      if self.hascl[3]:
        TE = (ii_T * jj_E) + (ii_E * jj_T) 
        jac = jnp.array([(ps==i) * TE for i in range(self.nb)],dtype=jnp.int)
        Jt += [jac]
      if self.hascl[4]:
        TB = (ii_T * jj_B) + (ii_B * jj_T) 
        jac = jnp.array([(ps==i) * TB for i in range(self.nb)],dtype=jnp.int)
        Jt += [jac]
      if self.hascl[5]:
        EB = (ii_E * jj_B) + (ii_B * jj_E) 
        jac = jnp.array([(ps==i) * EB for i in range(self.nb)],dtype=jnp.int)
        Jt += [jac]
    
      return ordr,jnp.concatenate(Jt)
    
    return ordr
  
  def spec_order(self):
    order_string =""
    msk = self.msk
    msk.shape = (-1,self.m,self.m)

    if self.hascl[0]:
      "TT"



  def get_model_rq(self,cls,nuisance_dict,bin=True,max_cmp=-1):
    # get the calib
    max_cmp = max_cmp if max_cmp!=-1 else len(self.cmp)
    rq = self._get_cmb_rq(cls,bin)
    bins = self.bns_0 if bin else None
    for c in self.cmp[:max_cmp]:
      rq = c.apply(nuisance_dict,rq,bins)
    return rq
    
  def get_nuisance_rq(self,cls,nuisance_dict,bin=True):
    if bin:
      rq = jnp.zeros((self.nb,self.m,self.m))
    else:
      rq = jnp.zeros((self.lmax-self.lmin+1,self.m,self.m))
    bins=None
    if bin:
      bins = self.bins
    for c in self.cmp:
      c.apply(nuisance_dict,rq,bins)
    return rq

  def _get_cmb_rq_nmp(self,cls, bin):
    # get the cls

    if bin:
      rq = jnp.zeros((self.nb,self.m,self.m))
    else:
      rq = jnp.zeros((self.lmax-self.lmin+1,self.m,self.m))
    nb = self.nb
    m = self.m
    mt = self.mt
    me = self.me
    mb = self.mb
    blmin = self.blmin
    blmax = self.blmax
    b_ws = self.b_ws
    lmin = self.lmin
    lmax = self.lmax
    # bin it (and order it in)
    if bin:
      for b in range(nb):
        if mt:
          rq[b,:mt,:mt]+=jnp.sum(cls[0,lmin+blmin[b]:lmin+blmax[b]+1]*b_ws[blmin[b]:blmax[b]+1])
          if me:
            rq[b,:mt,mt:mt+me]+=jnp.sum(cls[3,lmin+blmin[b]:lmin+blmax[b]+1]*b_ws[blmin[b]:blmax[b]+1])
            rq[b,mt:mt+me,:mt]+=jnp.sum(cls[3,lmin+blmin[b]:lmin+blmax[b]+1]*b_ws[blmin[b]:blmax[b]+1])
          if mb:
            rq[b,:mt,mt+me:mb+mt+me]+=jnp.sum(cls[4,lmin+blmin[b]:lmin+blmax[b]+1]*b_ws[blmin[b]:blmax[b]+1])
            rq[b,mt+me:mb+mt+me,:mt]+=jnp.sum(cls[4,lmin+blmin[b]:lmin+blmax[b]+1]*b_ws[blmin[b]:blmax[b]+1])
        if me:
          rq[b,mt:mt+me,mt:mt+me]+=jnp.sum(cls[1,lmin+blmin[b]:lmin+blmax[b]+1]*b_ws[blmin[b]:blmax[b]+1])
          if mb:
            rq[b,mt:mt+me,mt+me:mb+mt+me]+=jnp.sum(cls[5,lmin+blmin[b]:lmin+blmax[b]+1]*b_ws[blmin[b]:blmax[b]+1])
            rq[b,mt+me:mb+mt+me,mt:mt+me]+=jnp.sum(cls[5,lmin+blmin[b]:lmin+blmax[b]+1]*b_ws[blmin[b]:blmax[b]+1])
        if mb:
          rq[b,mt+me:mt+me+mb,mt+me:mb+mt+me]+=jnp.sum(cls[2,lmin+blmin[b]:lmin+blmax[b]+1]*b_ws[blmin[b]:blmax[b]+1])
    else:
      if mt:
        rq[:,:mt,:mt]=cls[0,lmin:lmax+1]
        if me:
          rq[b,:mt,mt:mt+me]=cls[3,lmin:lmax+1]
          rq[b,mt:mt+me,:mt]=cls[3,lmin:lmax+1]
        if mb:
          rq[b,:mt,mt+me:mb+mt+me]=cls[4,lmin:lmax+1]
          rq[b,mt+me:mb+mt+me,:mt]=cls[4,lmin:lmax+1]
      if me:
        rq[b,mt:mt+me,mt:mt+me]=cls[1,lmin:lmax+1]
        if mb:
          rq[b,mt:mt+me,mt+me:mb+mt+me]=cls[5,lmin:lmax+1]
          rq[b,mt+me:mb+mt+me,mt:mt+me]=cls[5,lmin:lmax+1]
      if mb:
        rq[b,mt+me:mt+me+mb,mt+me:mb+mt+me]=cls[2,lmin:lmax+1]
    return rq

  @partial(jit, static_argnums=(0,))
  def _get_cmb_rq_jax_bin(self,cls):
    # get the cls

    rq = jnp.zeros((self.nb,self.m,self.m), dtype=jnp64)
    nb = self.nb
    m = self.m
    mt = self.mt
    me = self.me
    mb = self.mb
    bns_0 = self.bns_0
    lmin = self.lmin
    lmax = self.lmax
    nl = lmax+1-lmin
    # bin it (and order it in)
    #TT
    rq = rq.at[:,:mt,:mt].set(((bns_0[:,:nl*(mt!=0)] @ cls[0,lmin*(mt!=0):(lmax+1)*(mt!=0)])[:,nm.newaxis,nm.newaxis])[:,:mt,:mt])
    #TE
    rq = rq.at[:,:mt,mt:mt+me].set(((bns_0[:,:nl*(mt!=0)*(me!=0)] @ cls[3,lmin*(mt!=0)*(me!=0):(lmax+1)*(mt!=0)*(me!=0)])[:,nm.newaxis,nm.newaxis])[:,:mt,:me])
    rq = rq.at[:,mt:mt+me,:mt].set(((bns_0[:,:nl*(mt!=0)*(me!=0)] @ cls[3,lmin*(mt!=0)*(me!=0):(lmax+1)*(mt!=0)*(me!=0)])[:,nm.newaxis,nm.newaxis])[:,:mt,:me])
    #TB
    rq = rq.at[:,:mt,mt+me:mb+mt+me].set(((bns_0[:,:nl*(mt!=0)*(mb!=0)] @ cls[4,lmin*(mt!=0)*(mb!=0):(lmax+1)*(mt!=0)*(mb!=0)])[:,nm.newaxis,nm.newaxis])[:,:mt,:mb])
    rq = rq.at[:,mt+me:mb+mt+me,:mt].set(((bns_0[:,:nl*(mt!=0)*(mb!=0)] @ cls[4,lmin*(mt!=0)*(mb!=0):(lmax+1)*(mt!=0)*(mb!=0)])[:,nm.newaxis,nm.newaxis])[:,:mt,:mb])
    #EE
    rq = rq.at[:,mt:mt+me,mt:mt+me].set(((bns_0[:,:nl*(me!=0)] @ cls[1,lmin*(me!=0):(lmax+1)*(me!=0)])[:,nm.newaxis,nm.newaxis])[:,:me,:me])
    #EB
    rq = rq.at[:,mt:mt+me,mt+me:mb+mt+me].set(((bns_0[:,:nl*(mt!=0)*(mb!=0)] @ cls[5,lmin*(mt!=0)*(mb!=0):(lmax+1)*(mt!=0)*(mb!=0)])[:,nm.newaxis,nm.newaxis])[:,:me,:mb])
    rq = rq.at[:,mt+me:mb+mt+me,mt:mt+me].set(((bns_0[:,:nl*(mt!=0)*(mb!=0)] @ cls[5,lmin*(mt!=0)*(mb!=0):(lmax+1)*(mt!=0)*(mb!=0)])[:,nm.newaxis,nm.newaxis])[:,:me,:mb])
    #BB
    rq = rq.at[:,mt+me:mt+me+mb,mt+me:mb+mt+me].set(((bns_0[:,:nl*(mb!=0)] @ cls[2,lmin*(me!=0):(lmax+1)*(mb!=0)])[:,nm.newaxis,nm.newaxis])[:,:mb,:mb])
    return rq


  @partial(jit, static_argnums=(0,))
  def _get_cmb_rq_jax_unbin(self,cls):
    # get the cls

    rq = jnp.zeros((self.lmax-self.lmin+1,self.m,self.m), dtype=jnp64)
    nb = self.nb
    m = self.m
    mt = self.mt
    me = self.me
    mb = self.mb
    blmin = self.blmin
    blmax = self.blmax
    b_ws = self.b_ws
    lmin = self.lmin
    lmax = self.lmax
    #TT
    rq = rq.at[:,:mt,:mt].set((( cls[0,lmin*(mt!=0):(lmax+1)*(mt!=0)])[:,nm.newaxis,nm.newaxis])[:,:mt,:mt])
    #TE
    rq = rq.at[:,:mt,mt:mt+me].set((( cls[3,lmin*(mt!=0)*(me!=0):(lmax+1)*(mt!=0)*(me!=0)])[:,nm.newaxis,nm.newaxis])[:,:mt,:me])
    rq = rq.at[:,mt:mt+me,:mt].set((( cls[3,lmin*(mt!=0)*(me!=0):(lmax+1)*(mt!=0)*(me!=0)])[:,nm.newaxis,nm.newaxis])[:,:mt,:me])
    #TB
    rq = rq.at[:,:mt,mt+me:mb+mt+me].set((( cls[4,lmin*(mt!=0)*(mb!=0):(lmax+1)*(mt!=0)*(mb!=0)])[:,nm.newaxis,nm.newaxis])[:,:mt,:mb])
    rq = rq.at[:,mt+me:mb+mt+me,:mt].set((( cls[4,lmin*(mt!=0)*(mb!=0):(lmax+1)*(mt!=0)*(mb!=0)])[:,nm.newaxis,nm.newaxis])[:,:mt,:mb])
    #EE
    rq = rq.at[:,mt:mt+me,mt:mt+me].set((( cls[1,lmin*(me!=0):(lmax+1)*(me!=0)])[:,nm.newaxis,nm.newaxis])[:,:me,:me])
    #EB
    rq = rq.at[:,mt:mt+me,mt+me:mb+mt+me].set((( cls[5,lmin*(mt!=0)*(mb!=0):(lmax+1)*(mt!=0)*(mb!=0)])[:,nm.newaxis,nm.newaxis])[:,:me,:mb])
    rq = rq.at[:,mt+me:mb+mt+me,mt:mt+me].set((( cls[5,lmin*(mt!=0)*(mb!=0):(lmax+1)*(mt!=0)*(mb!=0)])[:,nm.newaxis,nm.newaxis])[:,:me,:mb])
    #BB
    rq = rq.at[:,mt+me:mt+me+mb,mt+me:mb+mt+me].set((( cls[2,lmin*(me!=0):(lmax+1)*(mb!=0)])[:,nm.newaxis,nm.newaxis])[:,:mb,:mb])
    
    return rq

  def _get_cmb_rq_jax(self,cls,bin):
    if bin:
      return self._get_cmb_rq_jax_bin(cls)
    else:
      return self._get_cmb_rq_jax_unbin(cls)


  @partial(jit, static_argnums=(0,))
  def __call__(self,cls,nuisance_dict,chi2_mode=False):
    #for k in nuisance_dict:
    #  print(k,nuisance_dict[k],nm.dtype(nuisance_dict[k]))
    cls=self._calib(cls,nuisance_dict)
    rq = self.get_model_rq(cls,nuisance_dict, False if self.bins is None else True)
    delta = self.rqh_f - rq.flatten()[self.oo]
    lkl = -.5 * delta.T @ (self.siginv @ delta)
    return lkl

  def _crop(self,lkl,**options):
    if options.get("crop",""):
      crop_cmd = options["crop"]
      if isinstance(crop_cmd,(tuple,list)):
        crop_cmd = "\n".join(crop_cmd)
      crop_cmd = crop_cmd.split("\n")
      msk = translate_crop(crop_cmd[0],self.mt,self.me,self.hascl,[100,143,217],self.lmin,self.lmax,self.bns_0)
      for cro in crop_cmd[1:]:
        msk *= translate_crop(cro,self.mt,self.me,self.hascl,[100,143,217],self.lmin,self.lmax,self.bns_0)
      rmsk = msk.flatten()[self.oo]
      noo = self.oo[rmsk]
      nrqh_f = self.rqh_f[rmsk]
      print("inverse can be slow")
      sig = nm.linalg.inv(self.siginv)
      nsig = sig[rmsk,:][:,rmsk]
      self.sig = nsig
      print("inverse can be slow")
      nsiginv = nm.linalg.inv(nsig)
      print("before crop")
      self.print_lranges()
      self.oo = noo
      self.siginv = nsiginv
      self.rqh_f = nrqh_f
      print("after crop")
      self.print_lranges()
    #super()._init(candl,**options)
      
  def candl_init(self,candl,**options):
    if options.get("all_priors",False):
      options["baseline_priors"] = True
      options["SZ_priors"] = True
    if options.get("baseline_priors",False):
      candl.set_priors(_priors,std=True)
    if options.get("SZ_priors",False):
      candl.set_priors({("A_sz","ksz_norm"):("linear combination",(1.6,1.),9.5,3)},std=True)
    super().candl_init(candl,**options)
    
  @property
  def data_bandpowers(self):
    return self.rqh_f

  @property
  def covariance(self):
    if self.sig is not None:
      return self.sig*1.
    else:
      self.sig = jnp.linalg.inv(self.siginv)
      return self.sig*1.

  @property
  def spec_order(self):
    ns = nm.array([self.mt,self.me,self.mb])
    sn = nm.array([0,ns[0],ns[0]+ns[1]])
    frq = [str(f) for f in [100,143,217]]
    ntot = nm.sum(ns)
    a = (self.oo%(ntot*ntot))//ntot
    b = self.oo%ntot
    cr = []
    for i,j in zip(a,b):
      if len(cr)==0 or cr[-1][0]!=i or cr[-1][1]!=j:
        cr += [(i,j)]
    so = []
    for i,j in cr:
      if i<sn[1]:
        ka="T"
        fa = frq[i]
      elif i<sn[2]:
        ka="E"
        fa = frq[i-ns[0]]
      else:
        ka="B"
        fa = frq[i-ns[1]]
      if j<sn[1]:
        kb="T"
        fb = frq[j]
      elif j<sn[2]:
        kb="E"
        fb = frq[j-ns[0]]
      else:
        kb="B"
        fb = frq[j-ns[1]]
      so += [ka+kb+" %sx%s"%(fa,fb)]
    return so

  @property
  def spec_types(self):
    so = self.spec_order
    return [s.split()[0] for s in so]

  @property
  def bins_start_ix(self):
    ns = nm.array([self.mt,self.me,self.mb])
    ntot = nm.sum(ns)
    a = (self.oo%(ntot*ntot))//ntot
    b = self.oo%ntot
    p = a*ntot+b
    return nm.concatenate([[0],nm.arange(len(p)-1)[p[1:]-p[:-1]!=0]+1])
  
  @property
  def bins_stop_ix(self):
    return nm.concatenate([self.bins_start_ix[1:],[len(self.oo)]])
  
  @property
  def effective_ells(self):
    lq = self.rqh*0
    lq[:]=self.lm[:,nm.newaxis,nm.newaxis]
    return lq[self.oo]

  def get_lranges(self):
    ns = nm.array([self.mt,self.me,self.mb])
    sn = nm.array([0,ns[0],ns[0]+ns[1]])
    ntot = nm.sum(ns)
    frq = [str(f) for f in [100,143,217]]
    nl = self.nb
    rqmin = nm.zeros((nl,ntot,ntot))
    rqmax = nm.zeros((nl,ntot,ntot))
    rqmsk = nm.zeros((nl,ntot,ntot))
    if self.bins is not None:
      rqmin[:] = self.bin_lmin[:nl,nm.newaxis,nm.newaxis]
      rqmax[:] = self.bin_lmax[:nl,nm.newaxis,nm.newaxis]
    else:
      rqmin[:] = nm.arange(self.lmax+1-self.lmin)[:nl,nm.newaxis,nm.newaxis]
      rqmax[:] = nm.arange(self.lmax+1-self.lmin)[:nl,nm.newaxis,nm.newaxis]  
    rqmsk.flat[nm.array(self.oo)] = 1
    
    res = []

    for i in range(ntot):
      if i<sn[1]:
        ka="T"
        fa = frq[i]
      elif i<sn[2]:
        ka="E"
        fa = frq[i-sn[1]]
      else:
        ka="B"
        fa = frq[i-sn[2]]
      for j in range(ntot):
        if nm.sum(rqmsk[:,i,j])==0:
          continue
        if j<sn[1]:
          kb="T"
          fb = frq[j]
        elif j<sn[2]:
          kb="E"
          fb = frq[j-sn[1]]
        else:
          kb="B"
          fb = frq[j-sn[2]]
        li = ka+kb+" %sx%s "%(fa,fb)
        cures = [li,ka+kb,(fa,fb),(i,j)]
        pos = []
        t=0
        up=False
        print(cures)
        while t<nl:
          if rqmsk[t,i,j]==1 and not up:
            pos +=[[[t,t],[rqmin[t,i,j]+self.lmin,rqmin[t,i,j]+self.lmin]]]
            up = True
          if rqmsk[t,i,j]==0 and up:
            print(pos,pos[-1])
            pos[-1][0][-1]=t
            pos[-1][1][-1]=rqmax[t-1,i,j]+self.lmin
            up=False
          t +=1
        if up:
          print("ARGL")
        cures += pos
        res += [cures]
    return res


  def print_lranges(self):
    upstring = u'\u005F'+"/"+u'\u203E'
    downstring = u'\u203E'+"\\"+u'\u005F'
    top = u'\u203E'*2+u"\uFE49"+u'\u203E'*2
    bottom = u'\u005F'*2+u"\uFE4D"+u'\u005F'*2
    ns = nm.array([self.mt,self.me,self.mb])
    sn = nm.array([0,ns[0],ns[0]+ns[1]])
    ntot = nm.sum(ns)
    frq = [str(f) for f in [100,143,217]]
    nl = self.nb
    rqmin = nm.zeros((nl,ntot,ntot))
    rqmax = nm.zeros((nl,ntot,ntot))
    rqmsk = nm.zeros((nl,ntot,ntot))
    if self.bins is not None:
      rqmin[:] = self.bin_lmin[:nl,nm.newaxis,nm.newaxis]
      rqmax[:] = self.bin_lmax[:nl,nm.newaxis,nm.newaxis]
    else:
      rqmin[:] = nm.arange(self.lmax+1-self.lmin)[:nl,nm.newaxis,nm.newaxis]
      rqmax[:] = nm.arange(self.lmax+1-self.lmin)[:nl,nm.newaxis,nm.newaxis]  
    rqmsk.flat[nm.array(self.oo)] = 1
    for i in range(ntot):
      if i<sn[1]:
        ka="T"
        fa = frq[i]
      elif i<sn[2]:
        ka="E"
        fa = frq[i-sn[1]]
      else:
        ka="B"
        fa = frq[i-sn[2]]
      for j in range(ntot):
        if nm.sum(rqmsk[:,i,j])==0:
          continue
        if j<sn[1]:
          kb="T"
          fb = frq[j]
        elif j<sn[2]:
          kb="E"
          fb = frq[j-sn[1]]
        else:
          kb="B"
          fb = frq[j-sn[2]]
        li = ka+kb+" %sx%s "%(fa,fb)
        
        t=0
        up=False
        while t<nl:
          if rqmsk[t,i,j]==1 and not up:
            li += upstring + "%d"%(rqmin[t,i,j]+self.lmin)+top
            up = True
          if rqmsk[t,i,j]==0 and up:
            li += "%d"%(rqmax[t-1,i,j]+self.lmin)+downstring+bottom
            up=False
          t +=1
        if up:
          li += "%d"%(rqmax[-1,i,j]+self.lmin)+downstring
        else:
          li = li[:-len(bottom)]
        print(li)



  _cosmomc_names = {
    "A_cib_217" : "acib217",
    "cib_index" : "ncib",
    "xi_sz_cib" : "xi",
    "A_sz" : "asz143",
    "ps_A_100_100" : "aps100",
    "ps_A_143_143" : "aps143",
    "ps_A_143_217" : "aps143217",
    "ps_A_217_217" : "aps217",
    "ksz_norm" : "aksz",
    "gal545_A_100" : "kgal100",
    "gal545_A_143" : "kgal143",
    "gal545_A_143_217" : "kgal143217",
    "gal545_A_217" : "kgal217",
    "galf_EE_A_100" : "galfEE100",
    "galf_EE_A_100_143" : "galfEE100143",
    "galf_EE_A_100_217" : "galfEE100217",
    "galf_EE_A_143" : "galfEE143",
    "galf_EE_A_143_217" : "galfEE143217",
    "galf_EE_A_217" : "galfEE217",
    "galf_EE_index" : "galfEEindex",
    "galf_TE_A_100" : "galfTE100",
    "galf_TE_A_100_143" : "galfTE100143",
    "galf_TE_A_100_217" : "galfTE100217",
    "galf_TE_A_143" : "galfTE143",
    "galf_TE_A_143_217" : "galfTE143217",
    "galf_TE_A_217" : "galfTE217",
    "galf_TE_index" : "galfTEindex",
    "A_cnoise_e2e_100_100_EE" : "Acnoisee2e100",
    "A_cnoise_e2e_143_143_EE" : "Acnoisee2e143",
    "A_cnoise_e2e_217_217_EE" : "Acnoisee2e217",
    "A_sbpx_100_100_TT" : "Asbpx100TT",
    "A_sbpx_143_143_TT" : "Asbpx143TT",
    "A_sbpx_143_217_TT" : "Asbpx143217TT",
    "A_sbpx_217_217_TT" : "Asbpx217TT",
    "A_sbpx_100_100_EE" : "Asbpx100EE",
    "A_sbpx_100_143_EE" : "Asbpx100143EE",
    "A_sbpx_100_217_EE" : "Asbpx100217EE",
    "A_sbpx_143_143_EE" : "Asbpx143EE",
    "A_sbpx_143_217_EE" : "Asbpx143217EE",
    "A_sbpx_217_217_EE" : "Asbpx217EE",
    "calib_100T" : "cal0",
    "calib_217T" : "cal2",
    "calib_100P" : "calEE0",
    "calib_143P" : "calEE1",
    "calib_217P" : "calEE2",
    "A_pol" : "calPol",
    "A_planck" : "calPlanck",
  }

_priors = {
  "cib_index":-1.3, 
  "gal545_A_100"      : (8.6  , 2),
  "gal545_A_143"      : (10.6 , 2),
  "gal545_A_143_217"  : (23.5 , 8.5),
  "gal545_A_217"      : (91.9 , 20),

  "galf_EE_A_100"      : 0.055,
  "galf_EE_A_100_143"  : 0.040,
  "galf_EE_A_100_217"  : 0.094,
  "galf_EE_A_143"      : 0.086,
  "galf_EE_A_143_217"  : 0.21 ,
  "galf_EE_A_217"      : 0.70 ,

  "galf_EE_index"      : -2.4,

  "galf_TE_A_100"     : (0.13  , 0.042),
  "galf_TE_A_100_143" : (0.13  , 0.036),
  "galf_TE_A_100_217" : (0.46  , 0.09),
  "galf_TE_A_143"     : (0.207 , 0.072),
  "galf_TE_A_143_217" : (0.69  , 0.09 ),
  "galf_TE_A_217"     : (1.938 , 0.54),

  "galf_TE_index"     : -2.4,

  "calib_100T" : (1.0002  , 0.0007),
  "calib_217T" : (0.99805 , 0.00065),
  "calib_100P" : 1.021,
  "calib_143P" : 0.966,
  "calib_217P" : 1.040,
  "A_pol" : 1.0,

  "A_cnoise_e2e_100_100_EE" : 1,
  "A_cnoise_e2e_143_143_EE" : 1,
  "A_cnoise_e2e_217_217_EE" : 1,
  "A_sbpx_100_100_TT"       : 1,
  "A_sbpx_143_143_TT"       : 1,
  "A_sbpx_143_217_TT"       : 1,
  "A_sbpx_217_217_TT"       : 1,
  "A_sbpx_100_100_EE"       : 1,
  "A_sbpx_100_143_EE"       : 1,
  "A_sbpx_100_217_EE"       : 1,
  "A_sbpx_143_143_EE"       : 1,
  "A_sbpx_143_217_EE"       : 1,
  "A_sbpx_217_217_EE"       : 1,
}

def translate_crop(crop_cmd,mT,mP,hascl,frq,lmin,lmax,bins=None):
  ns = nm.array([mT*hascl[0],mP*hascl[1],mP*hascl[2]])
  sn = nm.array([0,ns[0],ns[0]+ns[1]])
  ntot = nm.sum(ns)
  frq = [str(f) for f in frq]
  import re
  regex = re.compile("(?i)^(no|only|crop|notch)?\\s*([T|E|B][T|E|B])(\\s+(\\d+)x(\\d+))?(\\s+(-?\\d+)?\\s+(-?\\d+)?)?(?:\\s+(strict|lax|half))?")
  m = regex.match(crop_cmd.strip())
  ell = nm.arange(lmin,lmax+1)
  if m:
    only = False
    notch = False
    if m.group(1) is not None:
      if m.group(1).lower()=="only":
        only = True
      if m.group(1).lower() in ("notch","no"):
        notch = True
    strictness = 1
    if m.group(9) is not None:
      if m.group(9).lower()=="lax":
        strictness = 0
      elif m.group(9).lower()=="strict":
        strictness = 1
      else:
        strictness = 0.5

    kind = ["TEB".index(k) for k in m.group(2).upper()]
    if kind[0]!=kind[1]:
      kind = ((kind[0],kind[1]),(kind[1],kind[0]))
    else:
      kind = (kind,)
    if m.group(3) is not None:
      fc = ((frq.index(m.group(4)),frq.index(m.group(5))),(frq.index(m.group(5)),frq.index(m.group(4))))
    else:
      fc = ()
      for f in range(len(frq)):
        for g in range(len(frq)):
          fc = fc + ((f,g),)
    if m.group(6) is not None:
      cmin = int(m.group(7))
      cmax = int(m.group(8))
    else:
      cmin = 0
      cmax = lmax
    if cmin<0:
      cmin=0
    if cmax<0:
      cmax=lmax
    lmsk = (ell>=cmin) * (ell<=cmax)
    
    if bins is not None:
      lmsk = bins @ lmsk
      
    if notch:
      lmsk = 1-lmsk
      #strictness = 1-strictness
    if strictness<1:
      lmsk = lmsk>strictness
    else:
      lmsk = nm.abs(lmsk-1)<1e-5
    msk = nm.ones((len(lmsk),ntot,ntot),dtype=nm.int8)*2
   
    for kn in kind:
       for f in fc:
        if f[0]>ns[kn[0]] or f[1]>ns[kn[1]]:
          #out of range, ignore
          continue
        msk[:,sn[kn[0]]+f[0],sn[kn[1]]+f[1]]=lmsk
    if only:
      msk[msk==2]=0
    msk = msk>0
    return msk



