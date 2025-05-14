# popular clik likelihoods are here

# Try to import JAX, fall back to numpy if needed

from . import *
import importlib

_supported = {"smica":"smica","gibbs_gauss":"gibbs","simall":"simall","bflike_smw":"bflike","plik_cmbonly":"cmbonly"}

import inspect

from collections import OrderedDict
import copy

@partial(jit, static_argnums=(0,))
def cls_fromcosmomc(cls):
  lmax = cls.shape[1]+1
  ncls = jnp.zeros((6,self.lmax+1),dtype=jnp64)
  llp1 = jnp.arange(2,self.lmax+1)/2./jnp.pi
  ncls = ncls.at[0,2:].set(cls[1,:self.lmax+1-2]/llp1)
  ncls = ncls.at[1,2:].set(cls[3,:self.lmax+1-2]/llp1)
  ncls = ncls.at[2,2:].set(cls[4,:self.lmax+1-2]/llp1)
  ncls = ncls.at[3,2:].set(cls[2,:self.lmax+1-2]/llp1)
  return ncls

class _clik_common:
  @partial(jit, static_argnums=(0,))
  def prior(self,nuisance_dict):
    lkl = 0
    for p in self._prior:
      if isinstance(p,tuple):
        vl = jnp.array([nuisance_dict[pp] for pp in p],dtype=jnp64)
      else:
        vl = jnp.array(nuisance_dict[p],dtype=jnp64)
      #print(p,vl,self._prior[p](vl))
      lkl += self._prior[p](vl)
    return lkl

  @partial(jit, static_argnums=(0,3))
  def __call__(self,cls,nuisance_dict={},chi2_mode=False):
    if cls.shape[-1]==self.parlen and len(cls.shape)==2:
      return jnp.array([self(c,nuisance_dict) for c in cls],dtype=jnp64)
    cls,nuisance_dict = self.normalize(cls,nuisance_dict)
    tot_dict = nuisance_dict|self._default
    for old,new in self.rename_dict.items():
      tot_dict[old] = tot_dict[new]
      del(tot_dict[new])
    lkl = self._internal(cls,tot_dict,chi2_mode)
    if not chi2_mode:
      lkl += self.prior(nuisance_dict)
    return lkl

  @partial(jit, static_argnums=(0,))
  def normalize_jax(self,cls,nuisance_dict={}):
    if (len(cls.shape)==1 or cls.shape[-1]==self.parlen):
      nuisance_dict = dict(zip(self.extra_parameter_names,cls[-len(self.extra_parameter_names):]))|nuisance_dict
      ncls = jnp.zeros((len(self.lmax),nm.max(self.lmax)+1),dtype=jnp64)
      off = 0
      for i in range(len(self.lmax)):
        if self.lmax[i]!=-1:
          ncls = ncls.at[i].set(cls[off:off+self.lmax[i]+1])
          off += self.lmax[i]+1
      return ncls,nuisance_dict
    return cls,nuisance_dict

  def normalize_mnp(self,cls,nuisance_dict={}):
    if (len(cls.shape)==1 or cls.shape[-1]==self.parlen):
      nuisance_dict = dict(zip(self.extra_parameter_names,cls[-len(self.extra_parameter_names):]))|nuisance_dict
      ncls = jnp.zeros((len(self.lmax),nm.max(self.lmax)+1),dtype=jnp64)
      off = 0
      for i in range(len(self.lmax)):
        if self.lmax[i]!=-1:
          ncls[i] = (cls[off:off+self.lmax[i]+1])
          off += self.lmax[i]+1
      return ncls,nuisance_dict
    return cls,nuisance_dict

  def normalize_clik(self,cls,nuisance_dict={}):
    if (len(cls.shape)==1 or cls.shape[-1]==self.parlen):
      ncls = nm.array(cls*1.)
      ncls[-len(self.extra_parameter_names):] = [nuisance_dict[p] for p in self.extra_parameter_names]
      return ncls
    else:
      ncls = nm.zeros(self.parlen)
      off = 0
      for i in range(len(self.lmax)):
        if self.lmax[i]!=-1:
          ncls[off:off+self.lmax[i]+1] = cls[i] 
          off += self.lmax[i]+1
      ncls[-len(self.extra_parameter_names):] = [nuisance_dict[p] for p in self.extra_parameter_names]
      return ncls       


  @property
  def default_par(self):
    return self._default_par

  @property
  def has_cl(self):
    return self.get_has_cl()
  def get_has_cl (self):
    return [1 if l!=-1 else 0 for l in self.lmax]

  def get_options(self):
    pass

  def get_lmax(self):
    return self._lmax.copy()

  def rename(self,rename_dict):
    self.rename_dict = rename_dict
    _default = OrderedDict()
    for k in self._default:
      v = self._default[k]
      _default[rename_dict.get(k,k)] = v
    self._default = _default
    _prior = OrderedDict()
    for k in self._prior:
      v = self._prior[k]
      if isinstance(k,tuple):
        k = tuple([rename_dict.get(ki,ki) for ki in k])
      else:
        k = rename_dict.get(k,k)
      _prior[k] = v
    self._prior = _prior

  def set_priors(self,priors,**options):
    for k,v in priors.items():
      if isinstance(k,tuple):
        # joint priors !
        # first check that all the parameters are there
        ig = False
        for kk in k:
          if kk not in self.extra_parameter_names:
            print("ignore prior on ",k)
            ig = True
            break 
        if ig:
          continue
        self._prior[k] = generate_prior_function(v,**options)
        continue
      if k not in self.extra_parameter_names:
        print("ignore prior on ",k)
        continue
      if isinstance(v,(int,float)):
        # default value, make sure to remove it from the default vector from the selfcheck !
        w = -len(self.extra_parameter_names)+self.extra_parameter_names.index(k)
        if self._default_par is not None:
          self._default_par=nm.concatenate([self._default_par[:w],self._default_par[w+1:]])
        self._parlen-=1
        self._default[k]=v
        
      else:
        self._prior[k]=generate_prior_function(v,**options)

  @property 
  def lmax(self):
    return self.get_lmax()

  def get_extra_parameter_names(self,rename=True):
    ext = [v for v in self.varpar if v not in self._default]
    if rename:
      return [self.rename_dict.get(old,old) for old in ext if self.rename_dict.get(old,old) not in self._default]
    return v


  @property 
  def extra_parameter_names(self):
    return self.get_extra_parameter_names()

  @property 
  def parlen(self):
    return nm.array(self._parlen)

  def _init_def_priors(self,clkl,radical):
    self._default =OrderedDict()
  
    if "default" in clkl[radical]:
      names = clkl[radical+"/default/name"].replace("\0"," ").split()
      loc = clkl[radical+"/default/loc"]
      for n,l in zip(names,loc):
        self._default[n] = l
  
    self._prior =OrderedDict()
  
    if "prior" in clkl[radical]:
      names = clkl[radical+"/prior/name"].replace("\0"," ").split()
      loc = clkl[radical+"/prior/loc"]
      var = clkl[radical+"/prior/var"]
      for n,l,v in zip(names,loc,var):
        self._prior[n] = (l,v)
  
    self.rename_dict={}

    # jax or numpy :
    if hasjax:
      self.normalize = self.normalize_jax
    else:
      self.normalize = self.normalize_mnp

    self._parlen = nm.sum(self._lmax+1)+len(self.extra_parameter_names)
    self._default_par = None

    print("----\n%s"%version());  # noqa: F405



class clik(_clik_common):

  @property
  def init_filename(self):
    return self._init_filename
  
  @property
  def init_options(self):
    return copy.deepcopy(self._init_options)
  
  def __init__(self,filename,**options):
    self._init_filename = filename

    self._init_options = copy.deepcopy(options)
    
    clkl = cldf.open(filename)
    if "clik" in clkl:
      self.__init__cmb(filename,**options)
    elif "clik_lensing" in clkl:
      self.__init__lensing(filename,**options)
    else:
      raise clik_emul_error("unsupported likelihood type %s"%filename)
  
  def __init__cmb(self,filename,**options):

    clkl = cldf.open(filename)

    if clkl["clik/n_lkl_object"]!=1:
      raise clik_emul_error("only one likelihood object supported in clik_emul")

    lkl = clkl["clik/lkl_0"]

    self._lmax = clkl["clik/lmax"]


    lkl_type = lkl["lkl_type"]

    if lkl_type not in _supported:
      raise clik_emul_error("unsupported likelihood type %s"%lkl_type)

    if _supported[lkl_type]: 
      lkl_type = _supported[lkl_type]

    try:
      md = importlib.import_module("."+lkl_type,__package__)
      self._internal = getattr(md,"%s_lkl"%lkl_type)(lkl,**options)
    except ImportError as e:
      raise clik_emul_error("could not import likelihood %s"%lkl_type)
    except AttributeError as e:
      raise clik_emul_error("could not find likelihood %s"%lkl_type)

    self._init_def_priors(clkl,"clik")
  
    if "check_param" in clkl["clik"]:
      par = clkl["clik"]["check_param"]
      self._default_par = jnp.array(par,dtype=jnp64)
      res = jnp64(clkl["clik"]["check_value"])
      res2 = self(jnp.array(par,dtype=jnp64))
  
      print("Checking likelihood '%s' on test data. got %g expected %g (diff %g)"%(filename,res2,res,res-res2))

    print("----")

    self._internal._post_init(clkl,**options)

    # at this stage I can jit a few things
    if hasjax:
      #self.__call__ = jit(self.__call__,static_argnums=(-1,))
      #self.prior = jit(self.prior)
      #self.normalize_jax = jit(self.normalize_jax,static_argnums=(0,))
      pass

  def __init__lensing(self,filename,**options):

    clkl = cldf.open(filename)
    assert clkl["clik_lensing/itype"]==4

    _lmax = clkl["clik_lensing/lmax"]
    hascl = clkl["clik_lensing/hascl"]
    lmax = [-1]*7
    lmax[0] = _lmax
    for i in range(6):
      if hascl[i]:
        lmax[i+1] = _lmax
    self._lmax = nm.array(lmax)

    self.nlt = nm.sum(self.lmax)+7
    self.nbins = clkl["clik_lensing/nbins"]
    self.pp_hat = jnp64(clkl["clik_lensing/pp_hat"])
    assert self.pp_hat.shape[0]==self.nbins
    self.bins = jnp64(clkl["clik_lensing/bins"])
    self.bins = jnp.reshape(self.bins,(self.nbins,-1))
    #assert self.bins.shape[0]==self.nbins+1
    self.siginv = jnp64(clkl["clik_lensing/siginv"])
    self.siginv = jnp.reshape(self.siginv,[self.nbins,self.nbins])

    self.cors=None

    if "cors" in clkl["clik_lensing"]:
      self.cors = jnp64(clkl["clik_lensing/cors"])
      self.cors = jnp.reshape(self.cors,[self.nbins,self.nlt])
      
    self.cl_fid = jnp64(clkl["clik_lensing/cl_fid"])
    assert self.cl_fid.shape[0]==self.nlt
    #self.cl_fid = jnp.reshape(self.cl_fid,(-1,self.lmax[0]+1))

    self.renorm = clkl["clik_lensing/renorm"]
    self.ren1 = clkl["clik_lensing/ren1"]

    self.varpar=[]
    if clkl["clik_lensing/has_calib"]:
      self.varpar=["A_planck"]

    if "cor0" in clkl["clik_lensing"]:
      self.cor0 = jnp64(clkl["clik_lensing/cor0"])
      assert self.cor0.shape[0]==self.nbins
    else:
      self.cor0 = 0
    
    self._m_llp1_2 = jnp64((nm.arange(_lmax+1)*(nm.arange(_lmax+1)+1.))**2/2./nm.pi)
    self._m_llp1 = jnp.tile(jnp64(nm.arange(_lmax+1)*(nm.arange(_lmax+1)+1.)/2./nm.pi),nm.sum(self.has_cl)-1)

    self._internal = _clik_lensing(self)
    self._init_def_priors(clkl,"clik_lensing")
    self._default_par = jnp.concatenate([self.cl_fid,jnp64([1.])])
    selftest = self(self._default_par)
    if "check" in clkl:
      cv = clkl["clik_lensing/check"]
      print("Checking likelihood '%s' on test data. got %g expected %g (diff %g)"%(filename,selftest,cv,cv-selftest))
    else:
      print("Checking likelihood '%s' on test data. got %g"%(filename,selftest))
    print("----")

  def __getattr__(self,name):
    return getattr(self._internal,name)


  def candl_init(self,**options):
    self._internal.candl_init(self,**options)

clik_lensing = clik

class _clik_cmb:
  def __init__(self,lkl,**options):
    self.lkl = lkl
    self.unit = lkl["unit"]
    self.has_cl = lkl["has_cl"]
    if "lmax" in lkl:
      self.lmin = int(lkl["lmin"])
      self.lmax = int(lkl["lmax"])
      self.ell = jnp.arange(self.lmin,self.lmax+1)
    else:
      self.ell = lkl["ell"]
    self.llp1 = self.ell*(self.ell+1)/jnp.array(2*jnp.pi,dtype=jnp64)
    self.nell = len(self.ell)
    self.lmaxs = [self.lmax if v else -1 for v in self.has_cl]

    self.nd = len(self.ell)*self.has_cl.sum()

    wl = None
    if "wl" in lkl:
      wl = lkl["wl"]

    if "nbins" in lkl:
      self.nbins = lkl["nbins"]
      if "bins" in lkl:
        self.bins = lkl["bins"]
        self.bins.shape = (self.nbins,self.nd)
        self.bin_pack = False
      else:
        self.bin_pack = True
        self.bin_ws = lkl["bin_ws"]
        self.bin_lmin = lkl["bin_lmin"]
        self.bin_lmax = lkl["bin_lmax"]

    if "free_calib" in lkl:
      self.free_calib = lkl["free_calib"]
      self.varpar = [self.free_calib]
    else:
      self.free_calib = None

  def _calib(self,cls,nuisance_dict):
    if self.free_calib is None:
      return cls
    return cls/jnp64(nuisance_dict[self.free_calib])**2

  def _post_init(self,lkl,**options):
    if "crop" in options:
      self._crop(lkl,**options)

  def _crop(self,candl,**options):
    print("crop is not implemented for this likelihood")

  def candl_init(self,candl,**options):
    # add prior on A_planck
    if options.get("all_priors",False):
      options["A_planck_prior"] = True
    if options.get("A_planck_prior",False):
      candl.set_priors({"A_planck":(1.,0.0025)},std=True)
    
    # rename
    if options.get("cosmomc_names",False):
      candl.rename(self._cosmomc_names)

    # data_selection
    if "data_selection" in options:
      crop_cmd=[]
      for cmd in options["data_selection"]:
        # use a regex rather than cutting the cmd like lennart
        import re
        regex = re.compile("(?i)^([T|E|B][T|E|B])?(\\s*(\\d+)x(\\d+))?(\\s*(?:(?:ell)|l)(<|>)(\\d+))?\\s*(remove|only)")
        m = regex.match(cmd.strip())
        if m[8]=="remove":
          crop_order = "no "
        else:
          crop_order = "only"
        
        if m[1]:
          spec = [m[1]]
        else:
          spec = ["TT","TE","EE"]

        trail=""

        if m[2] is not None and m[2].strip():
          trail += m[2].strip()
        
        if m[5]:
          if m[6]=="<":
            trail += " -1 "+m[7]
          else:
            trail += " "+m[7]+" -1"

        trail += " strict"
        trail = " "+trail.strip()
        for spc in spec:
          crop_cmd += [crop_order+" "+spc+trail]
      options["crop"] = crop_cmd
      self._crop(candl,**options)

  _cosmomc_names = {
  "A_planck" : "calPlanck",
  }

class _clik_lensing(_clik_cmb):
  def __init__(self,celf):
    self.celf=celf

  
  def __call__(self,cls,nuisance_dict,chi2_mode=False):
    calib=1.
    if len(self.celf.extra_parameter_names)==1:
      calib = nuisance_dict["A_planck"]
      #print(calib,nuisance_dict)
      calib = 1./(calib*calib);
    calib = jnp64(calib)
    
    vls = jnp.concatenate([cls[i] for i in range(7) if self.celf.lmax[i]!=-1])
    
    bcls = jnp.dot(self.celf.bins,cls[0]*self.celf._m_llp1_2) - self.celf.cor0
    
    if self.celf.cors is not None:
      fpars = vls
      if self.celf.renorm==0:
        fpars = self.celf.cl_fid
      fphi = cls[0]
      if self.celf.ren1 == 0:
        fphi = self.celf.cl_fid[:self.lmax[0]+1]
      curfid = jnp.concatenate([fphi*self.celf._m_llp1_2,fpars[self.celf.lmax[0]+1:]*calib*self.celf._m_llp1])
      bcls = bcls + self.celf.cors @ curfid

    delta = self.celf.pp_hat - bcls

    lkl = -.5* delta @ (self.celf.siginv @ delta)

    return lkl

class clik_candl(clik):
  def __init__(self,filename,**options):
    # I should check if filename appens to be a yaml file and if so change filename to be the value of some magic value...
    # for now I defer to the regular clik init
    super().__init__(filename,**options)
        
    # this one is a special case to deal with the candl specific features. Must be implemented by each likelihoods, in clik...
    self.candl_init(**options)

    # candl expects the following 
    self.ell_min = 2
    self.ell_max = nm.max(self.lmax)
    self._llp1 = jnp64(nm.arange(2,self.ell_max+1)*(nm.arange(2,self.ell_max+1)+1.)/2./nm.pi)
    self.normalize_from_candl = self.normalize_from_candl_numpy
    if hasjax:
      self.normalize_from_candl = self.normalize_from_candl_jax

  @property
  def _dr(self):
    return ["pp","TT","EE","BB","TE","TB","EB"][7-len(self.lmax):]

  def normalize_to_candl(self,cls,nuisance_dict={}):
    cls,nuisance_dict = self.normalize(cls,nuisance_dict)
    
    return nuisance_dict | {"Dl":dict([(vv,jnp64(cls[ii,2:]*self._llp1)) for ii,vv in enumerate(self._dr)])}

  def normalize_from_candl_jax(self,params):
    cls = jnp.zeros((len(self.lmax),self.ell_max+1),dtype=jnp64)
    for ii,vv in enumerate(self._dr):
      if self.lmax[ii]>0:
        cls = cls.at[ii,2:].set(params["Dl"][vv][:self.ell_max-2+1]/self._llp1)
    return cls,params
    
  def normalize_from_candl_numpy(self,params):
    cls = jnp.zeros((len(self.lmax),self.ell_max+1),dtype=jnp64)
    for ii,vv in enumerate(self._dr):
      if self.lmax[ii]>0:
        cls [ii,2:] = (params["Dl"][vv][:self.ell_max-2+1]/self._llp1)
    return cls,params
    
  def log_like(self,params,chi2_mode=False):
    # candl provides the Cls as Dls in the params dictionnary...
    # let's deal with that.
    cls,params = self.normalize_from_candl(params)
    return self(cls,params,chi2_mode)

  def chi_square(self,params):
    cls,params = self.normalize_from_candl(params)
    return -2*self(cls,params,chi2_mode=True)

  @property 
  def required_nuisance_parameters(self):
    return self.get_extra_parameter_names()
  @property 
  def unique_spec_types(self):
    return [self._dr[i] for i,l in enumerate(self.lmax) if l>0]

  def __call__(self,cls,nuisance_dict={},chi2_mode=False):
    if isinstance(cls,dict):
      return self.log_like(cls,chi2_mode)
    return super().__call__(cls,nuisance_dict,chi2_mode)

  @property
  def data_bandpowers(self):
    current_frame = inspect.currentframe()
    fname = current_frame.f_code.co_name
    try:
      return getattr(self._internal,fname)
    except AttributeError:
      raise NotImplementedError(fname+" not implemented")
  @property
  def covariance(self):
    current_frame = inspect.currentframe()
    fname = current_frame.f_code.co_name
    try:
      return getattr(self._internal,fname)
    except AttributeError:
      raise NotImplementedError(fname+" not implemented")
  @property
  def window_functions(self):
    current_frame = inspect.currentframe()
    fname = current_frame.f_code.co_name
    try:
      return getattr(self._internal,fname)
    except AttributeError:
      raise NotImplementedError(fname+" not implemented")
  @property
  def spec_order(self):
    current_frame = inspect.currentframe()
    fname = current_frame.f_code.co_name
    try:
      return getattr(self._internal,fname)
    except AttributeError:
      raise NotImplementedError(fname+" not implemented")
  @property
  def spec_types(self):
    current_frame = inspect.currentframe()
    fname = current_frame.f_code.co_name
    try:
      return getattr(self._internal,fname)
    except AttributeError:
      raise NotImplementedError(fname+" not implemented")
  @property
  def bins_start_ix(self):
    current_frame = inspect.currentframe()
    fname = current_frame.f_code.co_name
    try:
      return getattr(self._internal,fname)
    except AttributeError:
      raise NotImplementedError(fname+" not implemented")
  @property
  def bins_stop_ix(self):
    current_frame = inspect.currentframe()
    fname = current_frame.f_code.co_name
    try:
      return getattr(self._internal,fname)
    except AttributeError:
      raise NotImplementedError(fname+" not implemented")
  @property
  def effective_ells(self):
    current_frame = inspect.currentframe()
    fname = current_frame.f_code.co_name
    try:
      return getattr(self._internal,fname)
    except AttributeError:
      raise NotImplementedError(fname+" not implemented")


  @property
  def default_par_candl(self):
    return self.normalize_to_candl(self._default_par)
  
  @property
  def default_par_clik(self):
    return self._default_par

  @property
  def default_par(self):
    return self.default_par_candl

  @property
  def data_set_file(self):
    return self.init_filename

def generate_prior_function(v,**options):
  if callable(v):
    return lambda x:v(x,**options)
  if isinstance(v,(list,tuple)):
    if len(v)==2:
      v = ("g",v[0],v[1])
    if isinstance(v[0],str) :
      if v[0].lower()=="g":
        if isinstance(v[1],(int,float)):
          mean=jnp64(v[1])
          siginv=jnp64(1./v[2])
          if options.get("std",False):
            siginv = jnp64(siginv)**2
        else:
          mean = jnp.array(v[1],dtype=jnp64)
          sig = jnp.array(v[2],dtype=jnp64)
          if len(sig)==len(mean):
            sig = jnp.diag(sig,dtype=jnp64)
            if options.get("std",False):
              siginv = jnp64(siginv)**2
          siginv = jnp.linalg.inv(sig)
        return lambda x: jnp64(-.5)*jnp.dot(jnp.dot(jnp.array(x,dtype=jnp64)-mean,siginv),jnp.array(x,dtype=jnp64)-mean)
      if v[0].lower()=="u":
        MIN = jnp.array(v[1],dtype=jnp64)
        MAX = jnp.array(v[2],dtype=jnp64)
        return lambda x: 0 if jnp.all(MIN<=jnp.array(x,dtype=jnp64)<=MAX) else -jnp.inf
      if v[0].lower()=="linear combination":
        lc = jnp.array(v[1],dtype=jnp64)
        return lambda x: -.5 * (jnp.dot(lc,x)-v[2])**2/v[3]

