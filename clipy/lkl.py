# popular clik likelihoods are here

# Try to import JAX, fall back to numpy if needed

from . import *
import importlib

_supported = {"smica":"smica","gibbs_gauss":"gibbs","simall":"simall","bflike_smw":"bflike","plik_cmbonly":"cmbonly"}


from collections import OrderedDict

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



class clik:
  def __init__(self,filename,**options):

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

    self._default =OrderedDict()
  
    if "default" in clkl["clik"]:
      names = clkl["clik/default/name"].replace("\0"," ").split()
      loc = clkl["clik/default/loc"]
      for n,l in zip(names,loc):
        self._default[n] = l
  
    self._prior =OrderedDict()
  
    if "prior" in clkl["clik"]:
      names = clkl["clik/prior/name"].replace("\0"," ").split()
      loc = clkl["clik/prior/loc"]
      var = clkl["clik/prior/var"]
      for n,l,v in zip(names,loc,var):
        self._prior[n] = (l,v)
  
    self.rename_dict={}

    # jax or numpy :
    if hasjax:
      self.normalize = self.normalize_jax
    else:
      self.normalize = self.normalize_mnp
  
    print("----\n%s"%version());  # noqa: F405
  
    self._parlen = nm.sum(self._lmax+1)+len(self.extra_parameter_names)
    self._default_par = None

    if "check_param" in clkl["clik"]:
      par = clkl["clik"]["check_param"]
      self._default_par = jnp.array(par)
      res = clkl["clik"]["check_value"]
      res2 = self(jnp.array(par))
  
      print("Checking likelihood '%s' on test data. got %g expected %g (diff %g)"%(filename,res2,res,res-res2))

    print("----")

    if hasattr(self._internal,"_post_init"):
      self._internal._post_init(clkl,**options)

    # at this stage I can jit a few things
    if hasjax:
      self.__call__ = jax.jit(self.__call__,static_argnums=(0,))
      #self.prior = jax.jit(self.prior,static_argnums=(0,))
      self.normalize_jax = jax.jit(self.normalize_jax,static_argnums=(0,))

  @property
  def default_par(self):
    return self._default_par
  
  def __getattr__(self,name):
    return getattr(self._internal,name)

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
    ext = [v for v in self._internal.varpar if v not in self._default]
    if rename:
      return [self.rename_dict.get(old,old) for old in ext if self.rename_dict.get(old,old) not in self._default]
    return v


  @property 
  def extra_parameter_names(self):
    return self.get_extra_parameter_names()

  @property 
  def parlen(self):
    return self._parlen

  #@partial(jit, static_argnums=(0,))
  def normalize_jax(self,cls,nuisance_dict={}):
    if (len(cls.shape)==1 or cls.shape[-1]==self.parlen):
      nuisance_dict = dict(zip(self.extra_parameter_names,cls[-len(self.extra_parameter_names):]))|nuisance_dict
      ncls = jnp.zeros((6,nm.max(self.lmax)+1),dtype=jnp64)
      off = 0
      for i in range(6):
        if self.lmax[i]!=-1:
          ncls = ncls.at[i].set(cls[off:off+self.lmax[i]+1])
          off += self.lmax[i]+1
      return ncls,nuisance_dict
    return cls,nuisance_dict

  def normalize_mnp(self,cls,nuisance_dict={}):
    if (len(cls.shape)==1 or cls.shape[-1]==self.parlen):
      nuisance_dict = dict(zip(self.extra_parameter_names,cls[-len(self.extra_parameter_names):]))|nuisance_dict
      ncls = jnp.zeros((6,nm.max(self.lmax)+1),dtype=jnp64)
      off = 0
      for i in range(6):
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
      for i in range(6):
        if self.lmax[i]!=-1:
          cls[off:off+self.lmax[i]+1] = ncls[i] 
          off += self.lmax[i]+1
      ncls[-len(self.extra_parameter_names):] = [nuisance_dict[p] for p in self.extra_parameter_names]
      return ncls       

  #@partial(jit, static_argnums=(0,))
  def __call__(self,cls,nuisance_dict={},chi2_mode=False):
    if cls.shape[-1]==self._parlen and len(cls.shape)==2:
      return jnp.array([self(c,nuisance_dict) for c in cls])
    cls,nuisance_dict = self.normalize(cls,nuisance_dict)
    tot_dict = nuisance_dict|self._default
    for old,new in self.rename_dict.items():
      tot_dict[old] = tot_dict[new]
      del(tot_dict[new])
    lkl = self._internal(cls,tot_dict,chi2_mode)
    if not chi2_mode:
      lkl += self.prior(nuisance_dict)
    return lkl

  #@partial(jit, static_argnums=(0,))
  def prior(self,nuisance_dict):
    lkl = 0
    for p in self._prior:
      if isinstance(p,tuple):
        vl = jnp.array([nuisance_dict[pp] for pp in p])
      else:
        vl = jnp.array(nuisance_dict[p])
      #print(p,vl,self._prior[p](vl))
      lkl += self._prior[p](vl)
    return lkl

  def candl_init(self,**options):
    self._internal.candl_init(self,**options)

class _clik_lkl:
  def __init__(self,lkl,**options):
    self.lkl = lkl
    self.unit = lkl["unit"]
    self.has_cl = lkl["has_cl"]
    if "lmax" in lkl:
      self.lmin = lkl["lmin"]
      self.lmax = lkl["lmax"]
      self.ell = jnp.arange(self.lmin,self.lmax+1)
    else:
      self.ell = lkl["ell"]
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
    return cls/float(nuisance_dict[self.free_calib])**2

  def candl_init(self,candl,**options):
    # add prior on A_planck
    if options.get("all_priors",False):
      options["A_planck_prior"] = True
    if options.get("A_planck_prior",False):
      candl.set_priors({"A_planck":(1.,0.0025)},std=True)
    
    # rename
    if options.get("cosmomc_names",False):
      candl.rename(self._cosmomc_names)
  
  _cosmomc_names = {
  "A_planck" : "calPlanck",
  }



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
    self._llp1 = nm.arange(2,self.ell_max+1)*(nm.arange(2,self.ell_max+1)+1.)/2./nm.pi


  def normalize_to_candl(self,cls,nuisance_dict={}):
    cls,nuisance_dict = self.normalize(cls,nuisance_dict)
    return nuisance_dict | {"Dl":{"TT":jnp.array(cls[0,2:]*self._llp1),"EE":jnp.array(cls[1,2:]*self._llp1),"BB":jnp.array(cls[2,2:]*self._llp1),"TE":jnp.array(cls[3,2:]*self._llp1)}}

  def normalize_from_candl(self,params):
    cls = jnp.zeros((6,self.ell_max+1))
    if self.lmax[0]>0:
      cls = cls.at[0,2:].set(params["Dl"]["TT"][:self.ell_max-2+1]/self._llp1)
    if self.lmax[1]>0:
      cls = cls.at[1,2:].set(params["Dl"]["EE"][:self.ell_max-2+1]/self._llp1)
    if self.lmax[2]>0:
      cls = cls.at[2,2:].set(params["Dl"]["BB"][:self.ell_max-2+1]/self._llp1)
    if self.lmax[3]>0:
      cls = cls.at[3,2:].set(params["Dl"]["TE"][:self.ell_max-2+1]/self._llp1)
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
    return [["TT","EE","BB","TE","TB","EB"][i] for i in range(6) if self.lmax[i]>0]

  def __call__(self,cls,nuisance_dict={},chi2_mode=False):
    if isinstance(cls,dict):
      return self.log_like(cls,chi2_mode)
    return super().__call__(cls,nuisance_dict,chi2_mode)



  @property
  def default_par_candl(self):
    return self.normalize_to_candl(self._default_par)
  
  @property
  def default_par_clik(self):
    return self._default_par

  @property
  def default_par(self):
    return self.default_par_candl


def generate_prior_function(v,**options):
  if callable(v):
    return lambda x:v(x,**options)
  if isinstance(v,(list,tuple)):
    if len(v)==2:
      v = ("g",v[0],v[1])
    if isinstance(v[0],str) :
      if v[0].lower()=="g":
        if isinstance(v[1],(int,float)):
          mean=v[1]
          siginv=1./v[2]
          if options.get("std",False):
            siginv = siginv**2
        else:
          mean = jnp.array(v[1])
          sig = jnp.array(v[2])
          if len(sig)==len(mean):
            sig = jnp.diag(sig)
            if options.get("std",False):
              siginv = siginv**2
          siginv = jnp.linalg.inv(sig)
        return lambda x: -.5*jnp.dot(jnp.dot(jnp.array(x)-mean,siginv),jnp.array(x)-mean)
      if v[0].lower()=="u":
        MIN = jnp.array(v[1])
        MAX = jnp.array(v[2])
        return lambda x: 0 if jnp.all(MIN<=jnp.array(x)<=MAX) else -jnp.inf
      if v[0].lower()=="linear combination":
        lc = jnp.array(v[1])
        return lambda x: -.5 * (jnp.dot(lc,x)-v[2])**2/v[3]

