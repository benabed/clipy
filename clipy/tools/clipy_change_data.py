#!/usr/bin/env python3

# this is almost the same the clik one

import sys

import os

import clipy
nm=clipy.nm
jnp=clipy.jnp
jnp64=clipy.jnp64

hpy = clipy.cldf

try:
  import clik as clik
  _hasclik = True
except:
  _hasclik = False

from clipy.miniparse import miniparse

def main():
  argv=sys.argv
  if len(sys.argv)!=2:
    print("clipy version "+clipy.version())
    if _hasclik:
      print("clik version "+clik.version())
    print("usage : %s parameter_file"%(sys.argv[0]))
    sys.exit(1)

  pars = miniparse(argv[1])
  
  lklfile = pars.str.input_clik
  outfile = pars.str.output_clik
  print ("replace data in \n %s\nwith cl from %s\n"%(pars.input_clik,pars.new_data))

  print("read input clik file")
  inhf = hpy.File(lklfile)
  ty = inhf["clik/lkl_0/lkl_type"]
  
  assert ty in ["smica"],"Cannot change data for likelihood type %s"%ty
    
  inlkl = clipy.clik(lklfile)
  cls,nuisance_par = inlkl.normalize(inlkl.default_par)

  print("read new data")
  #try fits
  try:
      with hpy.pf.open(pars.str.new_data) as hdul:
        data = hdul[1].data  # Supposons que les données sont dans la première extension
        cls_new = nm.stack([data["ell"],data["TT"],data["TE"],data["EE"],data["BB"],])
  except Exception as e:  
    cls_new = nm.loadtxt(pars.str.new_data)

  rcls = nm.array(cls*jnp64(0.))
  lmax = cls.shape[1]-1
  llp1 = nm.arange(lmax+1)*nm.arange(1,lmax+2)/2./nm.pi
  ridx = [1,3,4,2]
  off = int(2-cls_new[0,0])
  for i in range(4):
    rcls[i,2:] = cls_new[ridx[i],off:off+lmax-1]/llp1[2:]
  
  first = True
  for nuis in nuisance_par:
    if nuis in pars:
      if first:
        first=False
        print("change following nuisances :")
      print("%s = %g (was %g)"%(nuis,getattr(pars.float,nuis),nuisance_par[nuis]))
      nuisance_par[nuis]=jnp64(getattr(pars.float,nuis))

  rq_new = nm.array(inlkl.get_model_rq(rcls,nuisance_par, False if inlkl.bins is None else True))

  print("create new clik file")
  if pars.bool(default=False).replace_output_clik:
    print("(remove %s if already existing)"%pars.str.output_clik)
  hpy.copyfile(lklfile,outfile,pars.bool(default=False).replace_output_clik)
  outhf = hpy.File(outfile,"r+")

  outhf["clik/lkl_0/Rq_hat"] = rq_new.flat[:]
  default_par = nm.array(inlkl.normalize_clik(rcls,nuisance_par))
  outhf["clik/check_param"] = default_par
  outhf["clik/check_value"] = 0.
  outhf.close()

  print("test ",pars.str.output_clik)
  outlkl = clipy.clik(pars.str.output_clik)
  
    



if __name__=="__main__":
  main(sys.argv)