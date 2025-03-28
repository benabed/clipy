from . import *

from . import lkl 

def read_namelist(path):
  res = {}
  with open(path) as f:
    l = f.readline()
    while bool(l.strip())==False:
      l = f.readline()
    assert l.strip()=="&inputs"
    l = f.readline()
    while bool(l.strip())==False:
      l = f.readline()
    while l.strip()!="&end":
      left,right = l.strip().split("=")
      right = right.strip("'")
      right = right.strip(".")
      res[left]=right
      l = f.readline()
      while bool(l.strip())==False:
        l = f.readline()
  return res

class bflike_lkl(lkl._clik_cmb):

  def __init__(self,lkl,**options):
    import astropy.io.fits as pf
    import os.path as osp
    super().__init__(lkl,**options)
    namelist = read_namelist(osp.join(lkl._name,"_external","params_bflike.ini"))

    self.datafile = namelist["datafile"]
    self.project_mondip = namelist["project_mondip"]=="true"
    self._internal_lmax = int(namelist["lmax"])
    self._internal_ell = nm.arange(self._internal_lmax+1)
    self.lswitch = int(namelist["lswitch"])
    self.basisfile = namelist["basisfile"]
    self.clfiducial = namelist["clfiducial"]

    data = pf.open(osp.join(lkl._name,"_external",self.datafile))
    
    self.ntemp = data[1].header["NTEMP"] 
    self.nq = data[1].header["NQ"] 
    self.nu = data[1].header["NU"] 

    self.ntot = self.ntemp +self.nq +self.nu
    self.nqu  = self.nq +self.nu
    
    self.nside = data[1].header["NSIDE"] 

    assert self._internal_lmax<=4*self.nside,"%d %d"%(self._internal_lmax,4*self.nside)
    assert 3*(self.lswitch+1)**2 <= self.ntot

    self.ndata = data[1].header["NUMDATA"] 
    self.nwrite = data[1].header["NWRITE"] 

    evec = jnp.array(data[1].data[data[1].header["TTYPE1"]],dtype=jnp64)

    _theta_phi = jnp.reshape(evec[:self.ntemp*2],(self.ntemp,2))
    Tvec = jnp.stack( (jnp.sin(_theta_phi[:,0])*jnp.cos(_theta_phi[:,1]) , 
                       jnp.sin(_theta_phi[:,0])*jnp.sin(_theta_phi[:,1]) ,
                       jnp.cos(_theta_phi[:,0])                         ) )
    mod = jnp.sqrt(jnp.sum(Tvec**2,axis=0))
    Tvec = Tvec / mod[jnp.newaxis,:]
    off = self.ntemp*2

    _theta_phi = jnp.reshape(evec[off:off+self.nq*2],(self.nq,2))
    Qvec = jnp.stack( (jnp.sin(_theta_phi[:,0])*jnp.cos(_theta_phi[:,1]) , 
                       jnp.sin(_theta_phi[:,0])*jnp.sin(_theta_phi[:,1]) ,
                       jnp.cos(_theta_phi[:,0])                         ) )
    mod = jnp.sqrt(jnp.sum(Qvec**2,axis=0))
    Qvec = Qvec / mod[jnp.newaxis,:]
    off += self.nq*2

    _theta_phi = jnp.reshape(evec[off:off+self.nu*2],(self.nu,2))
    Uvec = jnp.stack( (jnp.sin(_theta_phi[:,0])*jnp.cos(_theta_phi[:,1]) , 
                       jnp.sin(_theta_phi[:,0])*jnp.sin(_theta_phi[:,1]) ,
                       jnp.cos(_theta_phi[:,0])                         ) )
    mod = jnp.sqrt(jnp.sum(Uvec**2,axis=0))
    Uvec = Uvec / mod[jnp.newaxis,:]
    off += self.nq*2

    #beam
    self.beam = jnp.reshape(evec[off:off+4*self.nside*6+6],(6,-1))
    off += 4*self.nside*6+6

    #clnorm
    fct2 = 1./((self._internal_ell+2)*(self._internal_ell+1)*self._internal_ell*(self._internal_ell-1))
    fct = nm.sqrt(fct2)
    chgconv = (2.*self._internal_ell +1.)/2./(self._internal_ell*(self._internal_ell+1.))
    fct2_cnv = fct2*chgconv
    fct_cnv = fct*chgconv

    self.clnorm = self.beam**2*jnp.stack([chgconv.T,fct2_cnv.T,fct2_cnv.T,fct_cnv.T,fct_cnv.T,fct2_cnv.T])

    #ncvm
    self.ncvm = jnp.reshape(evec[off:off+self.ntot**2],(self.ntot,self.ntot))
    off += self.ntot**2

    #data
    self.data = jnp.reshape(evec[off:off+self.ntot*self.ndata],(self.ndata,self.ntot))

    # compute rotation angles
    cos1 = nm.zeros((self.ntot,self.ntot))
    sin1 = nm.zeros((self.ntot,self.ntot))
    cos2 = nm.zeros((self.ntot,self.ntot))
    sin2 = nm.zeros((self.ntot,self.ntot))

    for i in range(self.ntemp):
      for j in range(self.nq):
        a12,a21 = get_rotation_angle(Qvec[:,j],Tvec[:,i])
        cos1[i,j+self.ntemp] = jnp.cos(a12)
        sin1[i,j+self.ntemp] = jnp.sin(a12)
      for j in range(self.nu):
        a12,a21 = get_rotation_angle(Uvec[:,j],Tvec[:,i])
        cos1[i,j+self.ntemp+self.nq] = jnp.cos(a12)
        sin1[i,j+self.ntemp+self.nq] = jnp.sin(a12)
    
    for i in range(self.nq):
      for j in range(i,self.nq):
        a12,a21 = get_rotation_angle(Qvec[:,j],Qvec[:,i])
        cos1[i+self.ntemp,j+self.ntemp] = jnp.cos(a12)
        sin1[i+self.ntemp,j+self.ntemp] = jnp.sin(a12)
        cos2[i+self.ntemp,j+self.ntemp] = jnp.cos(a21)
        sin2[i+self.ntemp,j+self.ntemp] = jnp.sin(a21)
      for j in range(i,self.nu):
        a12,a21 = get_rotation_angle(Uvec[:,j],Qvec[:,i])
        cos1[i+self.ntemp,j+self.ntemp+self.nq] = jnp.cos(a12)
        sin1[i+self.ntemp,j+self.ntemp+self.nq] = jnp.sin(a12)
        cos2[i+self.ntemp,j+self.ntemp+self.nq] = jnp.cos(a21)
        sin2[i+self.ntemp,j+self.ntemp+self.nq] = jnp.sin(a21)
      
    for i in range(self.nu):
      for j in range(i,self.nu):
        a12,a21 = get_rotation_angle(Uvec[:,j],Uvec[:,i])
        cos1[i+self.ntemp+self.nq,j+self.ntemp+self.nq] = jnp.cos(a12)
        sin1[i+self.ntemp+self.nq,j+self.ntemp+self.nq] = jnp.sin(a12)
        cos2[i+self.ntemp+self.nq,j+self.ntemp+self.nq] = jnp.cos(a21)
        sin2[i+self.ntemp+self.nq,j+self.ntemp+self.nq] = jnp.sin(a21)


    if self.lswitch<self.lmax:
      cl_fid = jnp.zeros((self.lmax+1,6))
      try:
        cl_fidt = jnp.loadtxt(osp.join(lkl._name,"_external",self.clfiducial))
        cl_fid.at[2:,0].set(cl_fidt[1,:self.lmax+1])
        cl_fid.at[2:,1].set(cl_fidt[2,:self.lmax+1])
        if cl_fidt.shape[0]==4:
          cl_fid.at[2:,3].set(cl_fidt[3,:self.lmax+1])
        else:
          cl_fid.at[2:,3].set(cl_fidt[4,:self.lmax+1])
          cl_fid.at[2:,2].set(cl_fidt[3,:self.lmax+1])

      except :
        pass
      self.update_ncvm(cl_fid,self.lswitch+1,self.lmax,self.ncvm,self.project_mondip)
    else:
      self.update_ncvm(cl_fid,2,2,self.ncvm,self.project_mondip)
    
    # assume that the plms are in the file.
    feval = nm.zeros((self.lswitch+1,3*(2*self.lswitch+3)))
    fevec = nm.zeros((self.lswitch+1,self.ntot,3*(2*self.lswitch+3)))
    cblocks = nm.zeros((self.lswitch+1,3*(2*self.lswitch+3),3*(2*self.lswitch+3)))
    bsfile = cldf.forfile.open(osp.join(lkl._name,"_external",self.basisfile))
    for l in range(2,self.lswitch+1):
      nl = 3*(2*l+3)
      feval[l,:] = bsfile.read("f64")
      fevec[l,:,:] = bsfile.read("f64").T
      cblocks[l,:,:] = bsfile.read("f64").T
    bsfile.close()

    neigen = 0
    for l in range(2,self.lswitch+1):
       neigen = neigen +2*l+1
    neigen = neigen*3

    evec = nm.zeros((self.ntot,neingen))
#    iu = 0
#    for l in range(2,self.lswitch+1):
#       il = iu +1
#       iu = il +3*(2*l +1)-1
#       evec[:,il:iu] = fevec(l)%m
#    end do







  def update_ncvm(cls_in,l_inf,l_sup,NCM,project_mondip):
    ct0 = ct1 = 0
    if project_mondip:
      ct0 = ct1 = 1e6
    
    cls = cls_in*self.clnorm

    for i in  range(self.ntemp):
      for j in range(i,self.ntemp):
        cz = Tvec[:,j]@Tvec[:,i]
        lpl = pl(cz,l_sup)
        tt = cls[l_inf:l_sup+1,0] @ lpl[l_inf:]
        NCM.at[i,j].add(tt+ct0+pl[1]*ct1)
      for j in range(self.nq):
        cz = Qvec[:,j]@Tvec[:,i]
        lplm = plm(cz,l_sup)
        tq = - cls[l_inf:l_sup+1,3] @ lplm[l_inf:]
        tu = - cls[l_inf:l_sup+1,4] @ lplm[l_inf:]
        NCM.at[i,j+self.ntemp].add(tq*self.cos1[i,j+self.ntemp]+tu*self.sin1[i,j+self.ntemp])
      for j in range(self.nu):
        cz = Uvec[:,j]@Tvec[:,i]
        lplm = plm(cz,l_sup)
        tq = - cls[l_inf:l_sup+1,3] @ lplm[l_inf:]
        tu = - cls[l_inf:l_sup+1,4] @ lplm[l_inf:]
        NCM.at[i,j+self.ntemp+self.nq].add(-tq*self.sin1[i,j+self.ntemp+self.nq]+tu*self.cos1[i,j+self.ntemp+self.nq])
    for i in range(self.nq):
      for j in range(i,self.nq):
        cz = Qvec[:,j]@Qvec[:,i]
        lplm,lf1,lf2 = plm_and_f(cz,l_sup)
        qq = cls_in[l_inf:l_sup+1,1] @ f1[l_inf:] - cls_in[l_inf:l_sup+1,2] @ f2[l_inf:]
        uu = cls_in[l_inf:l_sup+1,2] @ f1[l_inf:] - cls_in[l_inf:l_sup+1,1] @ f2[l_inf:]
        qu = cls_in[l_inf:l_sup+1,5] @ (f1[l_inf:] + f2[l_inf:])

        c1c2 = cos1[i+self.ntemp,j+self.ntemp]*cos2[i+self.ntemp,j+self.ntemp]
        s1s2 = sin1[i+self.ntemp,j+self.ntemp]*sin2[i+self.ntemp,j+self.ntemp]
        c1s2 = cos1[i+self.ntemp,j+self.ntemp]*sin2[i+self.ntemp,j+self.ntemp]
        s1c2 = sin1[i+self.ntemp,j+self.ntemp]*cos2[i+self.ntemp,j+self.ntemp]

        NCM.at[j+self.ntemp,i+self.ntemp].add(qq*c1c2 +uu*s1s2 +qu*(c1s2+s1c2))

      for j in range(self.nu):
        cz = Uvec[:,j]@Qvec[:,i]
        lplm,lf1,lf2 = plm_and_f(cz,l_sup)

        lplm,lf1,lf2 = plm_and_f(cz,l_sup)
        qq = cls_in[l_inf:l_sup+1,1] @ f1[l_inf:] - cls_in[l_inf:l_sup+1,2] @ f2[l_inf:]
        uu = cls_in[l_inf:l_sup+1,2] @ f1[l_inf:] - cls_in[l_inf:l_sup+1,1] @ f2[l_inf:]
        qu = cls_in[l_inf:l_sup+1,5] @ (f1[l_inf:] + f2[l_inf:])

        c1c2 = cos1[i+self.ntemp,j+self.ntemp+self.nq]*cos2[i+self.ntemp,j+self.ntemp+self.nq]
        s1s2 = sin1[i+self.ntemp,j+self.ntemp+self.nq]*sin2[i+self.ntemp,j+self.ntemp+self.nq]
        c1s2 = cos1[i+self.ntemp,j+self.ntemp+self.nq]*sin2[i+self.ntemp,j+self.ntemp+self.nq]
        s1c2 = sin1[i+self.ntemp,j+self.ntemp+self.nq]*cos2[i+self.ntemp,j+self.ntemp+self.nq]

        NCM.at[j+self.ntemp,i+self.ntemp+self.nq].add(-qq*s1c2 +uu*c1s2 +qu*(c1c2-s1s2))

    for i in range(self.nu):
      for j in range(i,self.nu):
        cz = Uvec[:,j]@Qvec[:,i]
        lplm,lf1,lf2 = plm_and_f(cz,l_sup)
        qq = cls_in[l_inf:l_sup+1,1] @ f1[l_inf:] - cls_in[l_inf:l_sup+1,2] @ f2[l_inf:]
        uu = cls_in[l_inf:l_sup+1,2] @ f1[l_inf:] - cls_in[l_inf:l_sup+1,1] @ f2[l_inf:]
        qu = cls_in[l_inf:l_sup+1,5] @ (f1[l_inf:] + f2[l_inf:])

        c1c2 = cos1[i+self.ntemp+self.nq,j+self.ntemp+self.nq]*cos2[i+self.ntemp+self.nq,j+self.ntemp+self.nq]
        s1s2 = sin1[i+self.ntemp+self.nq,j+self.ntemp+self.nq]*sin2[i+self.ntemp+self.nq,j+self.ntemp+self.nq]
        c1s2 = cos1[i+self.ntemp+self.nq,j+self.ntemp+self.nq]*sin2[i+self.ntemp+self.nq,j+self.ntemp+self.nq]
        s1c2 = sin1[i+self.ntemp+self.nq,j+self.ntemp+self.nq]*cos2[i+self.ntemp+self.nq,j+self.ntemp+self.nq]

        NCM.at[j+self.ntemp+self.nq,i+self.ntemp+self.nq].add(qq*s1s2 +uu*c1c2 -qu*(c1s2+s1c2))
    return



def pl(cz,l_sup):
  pl = jnp.zeros((l_sup+1))
  pl.at[1:3].set(jnp.array([sz,1.5*cz*cz-.5],dtype=jnp64))
  for l in range(3,l_sup+1):
    pl.at[l].set((cz*(2*l -1)*pl[l-1] -(l-1)*pl[l-2])/l)
  return pl

def plm(cz,l_sup):
  plm = jnp.zeros((l_sup+1))
  plm.at[2].set(3*(1-cz*cz))
  for l in range(3,l_sup+1):
    plm.at[l].set((cz*(2*l -1)*plm[l-1] -(l+1)*plm[l-2])/(l-2))
  return plm

def plm_and_f(cz,l_sup):
  plm = jnp.zeros((l_sup+1))
  f1 = jnp.zeros((l_sup+1))
  f2 = jnp.zeros((l_sup+1))
  cz2 = cz*cz
  plm.at[2].set(3)
  f1.at[2].set(6.*(1.+cz2))
  f2.at[2].set(-12*cz)

  for l in range(3,l_sup+1):
    plm.at[l].set((cz*(2*l -1)*plm[l-1] -(l+1)*plm[l-2])/(l-2))
    f1.at[l].set(-(2*l-8 +l*(l-1)*(1. -cz2))*plm[l]+ (2*l+4)*cz*plm[l-1])
    f2.at[l].set(4.*(-(l-1)*cz*plm[l] +(l+2)*plm[l-1]))
  return plm,f1,f2

def get_rotation_angle(r1,r2):
  # returns a12, a21
  #r1 and r2 vectors
  
  #computes TWICE the rotation angle
  #between two vectors r1 and r2

  eps = jnp.pi/180./3600./100.
  zz = jnp.array([0,0,1],dtype=jnp64)
  epsilon = jnp.array( [eps,0,0],dtype=jnp64)

  r12 = jnp.cross(r1,r2)
  mod = jnp.sqrt(r12@r12)
  if mod<1.e-8:
    a12,a21 = 0,0
    return a12,a21
  r12 = r12/mod
  
  r1star = jnp.cross(zz,r1)
  r1star.at[2].set(0)
  mod = jnp.sqrt(r1star@r1star)
  if mod<1.e-8: # too close to pole
    r1star = r1star+epsilon
    mod = jnp.sqrt(r1star@r1star)
  r1star = r1star/mod

  r2star = jnp.cross(zz,r2)
  r2star.at[2].set(0)
  mod = jnp.sqrt(r2star@r2star)
  if mod<1.e-8: # too close to pole
    r2star = r2star+epsilon
    mod = jnp.sqrt(r2star@r2star)
  r2star = r2star/mod

  mod = r1star@r2star
  mod = jnp.minimum(1.,mod)
  mod = jnp.maximum(-1.,mod)

  if r12@zz>0:
    a12 = 2*jnp.arccos(mod)
  else:
    a12 = -2*jnp.arccos(mod)

  a12 = -a12

  mr12 = -r12
  mod = mr12@r2star
  mod = jnp.minimum(1.,mod)
  mod = jnp.maximum(-1.,mod)
  if r12@zz>0:
    a21 = 2*jnp.arccos(mod)
  else:
    a21 = -2*jnp.arccos(mod)
  a21 = -a21

  return a12,a21
