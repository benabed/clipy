#!/usr/bin/env python3

# this is almost the same the clik one

import sys

import numpy as nm
import clipy

def main():
  main_CMB(sys.argv)

def main_CMB(argv):
  from optparse import OptionParser
  parser = OptionParser()
  parser.add_option("-p", "--par", dest="options",
                  help="add option at initialization 'TOTO = TATA'",action="append")
  parser.add_option("-e", "--extra", dest="pars",
                  help="set exta parameter to a default value 'TOTO = TATA'",action="append")
  opt,argv = parser.parse_args(argv)
  options = {}
  if opt.options is not None:
    for kv in opt.options:
      kvs = kv.split("=")
      if len(kvs)!=2:
        continue
      k = kvs[0].strip()
      v = kvs[1].strip()
      if k in options:
        if isinstance(options[k],str):
          options[k] = [options[k]]
        options[k]+=[v]
      else:
        options[k]=v
  pars={}
  if opt.pars is not None:
    for kv in opt.pars:
      kvs = kv.split("=")
      if len(kvs)!=2:
        continue
      k = kvs[0].strip()
      v = float(kvs[1].strip())
      pars[k]=v
  
  lklfile = argv[1]
  #print(options)
  lkl = clipy.clik(lklfile,**options)
  for clfile in argv[2:]:
    cls = nm.loadtxt(clfile)
    nres = lkl(cls.flat[:],pars)
    print(clfile+", ".join([k+" = %g"%pars[k] for k in pars])+" -> %g"%nres)

if __name__=="__main__":
  main()