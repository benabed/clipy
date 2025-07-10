`clipy`
======
*July 2025*

`clipy` is a pure python implementation of most of [`clik`](https://github.com/benabed/clik) with [`JAX`](https://github.com/google/jax) support (see [here](#Likelihood-files) for the list of currently supported likelihoods).

`clipy` also is natively compatible with [`candl`](https://github.com/Lbalkenhol/candl/).

While it is a direct pure python replacement, and thus fully compatible with `clik`, `clipy` introduces some new features:

- the option to modify the data content of a likelihood at initialization time (i.e. frequencies, $\ell$-ranges, temperature and/or polarization) for some likelihoods (see [below](#New-initialisation-time-features)),
- an [alternative API](#New-calling-API) to pass power spectra and nuisance parameters when computing a likelihood.

## Citing
As for all usages of the planck likelihoods, please site [Planck 2018 V. CMB power spectra and likelihood, Planck Collaboration et al. A&A 641, A5 (2020)](	https://doi.org/10.1051/0004-6361/201936386).

## Likelihood files
This package only contains the code.
Likelihood files for Planck are available on the  [Planck Legacy Archive](http://pla.esac.esa.int/pla/#cosmology). 

It is currently only compatible with the 
following files :

- high-ell
	- all the plik files (i.e. TTTEEE, as well a single spectra files, binned and unbinned) 
	- all the plik-lite files (i.e. TTTEEE and TT only)
- low-ell
	- all the simall files (EE, BB and EEBB), including the sroll2 versions from [Pagano et al. A&A 635, A99 (2020)](https://doi.org/10.1051/0004-6361/201936630) available [here](https://web.fe.infn.it/~pagano/low_ell_datasets/sroll2).
	- the commander file	
- lensing
    - all the lensing likelihood from the PR3 release as well as the PR4 release (available here -- add link)  

`clipy` also allows to perform $ell$-range and spectra selections at initialisation time for the high-ell likelihood, without having to resort to the `clik_change_lrange` tool to create a new likelihood file ([see below](#New-initialisation-time-features)).

As a reminder, the `clik` likelihood files are directories containing a fixed hierarchy of files and directory containing the data and metadata necessary to compute the likelihood of a given CMB power spectrum or spectra along with nuisance parameters given a part of the Planck data (at large or small scales, using either temperature, polarisation or both datasets, and using a given subset of frequency channels).

## Version 
The code is at version **clipy_0.15**

### History
- **clipy_0.15** (07/25) - change `fromstring` to `frombuffer` when reading the `plik_lite` likelihood files to support latest `numpy`. 
- **clipy_0.14** (04/25) - multiple corrections for crop. add init_options and init_filename, two properties that allow to reinit an already initialized likelihood (usefull for cobaya)
- **clipy_0.12** (03/25) - multiple corrections. Correct installation of the CLI tools (`clipy_print`, etc...). Add lensing likelihoods. Refactor code.
- **clipy_0.11** (01/25) - Correct `commander` which was failing on some macos jax version. Improve documentation. Correct crop bug in 32bit mode.
- **clipy_0.1** (11/24) - initial release



## Installation
`clipy` can be installed with pip:

    pip install clipy-like
    
`clipy` requires the following packages:

- numpy 
- astropy 

any recent version (i.e. after 2016!) will do.

As for `clik`, tests can be performed by running the `clipy_print` tool:

	$> clipy_print plik_rd12_HM_v22b_TTTEEE.clik
	
## About JAX
[`JAX`](https://github.com/google/jax) is a Google-developed python library that allows to compute gradient and can deploy on GPU. It is used by [`candl`](https://github.com/Lbalkenhol/candl/) to perform all sorts of nice tricks, like computing fisher matrices or quickly optimizing the likelihood (in both sence of the word, finding the best parameters, and using JIT tricks to accelerate the computation).

Similarily to `candl`, `clipy` does not require `JAX` but will use it if it is available. Beware however that on some architectures, 64bits computation is not available in `JAX`. This will translate into up to $10^{-4}$ differences in log likelihood computations (for the plik files for example, this is due to a truncation of the planck overall calibration). 

`JAX` support can be turned off by setting the environement parameter `CLIPY_NOJAX` or by setting a global variable in python before importing clipy 
	
	CLIPY_NOJAX = True
	
## Using the library

The library can be called from python by importing the `clipy` python package.
It uses the same syntax than the `clik` likelihood and in most case, code using `clik` can be made to use `clipy` instead by replacing

	import clik

by

	import clipy as clik


### Initialization

`clipy` uses the same Planck likelihood files as `clik`. It is currently compatible with only a subset of those, listed [above](#Likelihood-files). 

CMB Likelihoods are represented by instance of the `clik` objects. 

```
import clipy
CMBlkl = clipy.clik("/path/to/some/clikfile")
```

##### New initialisation time features
With `clipy`, the `plik` and `plik_lite` likelihood can be modified at initialization time. The ell-range and list of spectra can be modified. Further in the case of `plik` file, ranges of multipoles (notches) can be omited in each spectra (for example to test the effect of the removal of the ell=1450 dip).
This is done with the following syntax :


```
import clipy
CMBlkl = clipy.clik("/path/to/some/plikfile", crop=[cmd1, cmd2,...])
```

where each `cmd1`,`cmd2`, etc is a string containing a crop command. The likelihood will use the remaining cls after applying all the crop commands. 

The command can be

- `crop XX [nuxmu]lmin lmax [strict|half|lax]`:  will crop the spectra XX (i.e. TT, TE, EE) to only cover the range lmin to lmax. If the optional nuxmu argument is present, nu and mu must be one of (100,143,217) and the crop command will only affect this cross frequency spectrum. This only make sense for `plik` likelihoods. Since the `plik` and `plik_lite` likelihoods are binned, the lmin and lmax are to be translated to the closest bins. The definition of closest bins is modified by the optional last argument. If it is set to strict (default) the crop will ensure that no multipole out of the lmin-lmax range is used in any selected bin, if it is set to lax, the crop command will conserve any bins that contains multipole between lmin-lmax, and if it is set to half, any given selected bins cannot give more than 50% weight to any bin out of the lmin-lmax range. For example `"crop TT 100x143 80 1250 strict"` will restrict the likelihood computation to use only the 100x143 TT bandpowers that contains informations from modes stricly larger than $\ell=80$ and smaller than $\ell=1250$. The rest of the spectra at TT at other frequency will be left unmodified.
- `no XX [nuxmu]` will remove entirely the `XX` spectrum.  If the optional nuxmu argument is present, nu and mu must be one of (100,143,217) and the  command will only affect this cross frequency spectrum. This only make sense for `plik` likelihoods. For example `"no TE"` will exclude all TE spectra from the likelihood computation.
- `only XX [nuxmu]lmin lmax [strict|half|lax]`: will only keep `XX` spectrum between lmin and lmax and cut all the other spectra. If the optional nuxmu argument is present, nu and mu must be one of (100,143,217) and the  command will only affect this cross frequency spectrum. This only make sense for `plik` likelihoods. The last optional argument behave similarily as the `crop` command. For example `"only EE 217x217 500 800 lax"` will restrict the computation to use only the 217 GHz EE autospectrum bandpowers that contain or use modes larger than $\ell=500$ and that contain or use modes smaller than $\ell=800$. All the rest of the data is discarded.
- `notch XX [nuxmu]lmin lmax [strict|half|lax]`: will remove the multipole of the `XX` spectrum between lmin and lmax. If the optional nuxmu argument is present, nu and mu must be one of (100,143,217) and the  command will only affect this cross frequency spectrum. This only make sense for `plik` likelihoods. The last optional argument behave similarily as the `crop` command. For example `"notch TT 100x100 100 130"` will exclude from the computation the 100 Ghz TT autospectrum bandpowers between $\ell=100$ and $\ell=130$. The rest of the spectra at TT at other frequency will be left unmodified.

### Obtaining the list of spectra, lrange, nuisance parameters

The ``get_lmax`` method of the ``clik`` object returns a 6 element tuple 
containing the maximum multipole for each of the CMB spectra. Each element of the tuple correspond to the
lmax of one of the CMB spectra needed by the `clik` object.
Ordering is **TT EE BB TE TB EB**. The tuple elements are set to -1 when a given spectra is not needed.

The `lmax` property of the object also returns the result of the `get_lmax` function.

The ``get_extra_parameter_names`` method of the ``clik`` object returns a tuple
containing the names of the nuisance parameters.

The `extra_parameter_names` property of the object also returns the result of the `get_extra_parameter_names` function.


##### New features of clipy objects

A new `default_par` property of the object contains the default spectra and nuisance parameters that are used to perform the self test when initializing the likelihood.

The new `normalize` method of the object transforms the usual spectra and nuisance parameter vector used in `clik`, such as the one that is stored in the `default_par` attribute into the new spectra and nuisance dictionnary format that `clipy` objects can use to compute likelihoods. 

	default_cl_an_nuisance = CMBlkl.default_par
	cls, nuisance_dict = CMBlkl.normalize(default_cl_an_nuisance)

More details on this new call API is defined [here](#New-calling-API). The new `normalize_clik` method performs the inverse function and transforms a spectra and a nuisance.

	cls_and_nuisance_vect = CMBlkl.normalize(cls, nuisance_dict)

#### Computing a log likelihood

The ``clik`` object is callable and returns the computation of a log_likelihood.
	
	loglkl = CMBlkl(cl_and_pars)
	
As in the old `clik`, `cl_and_pars` must be 
either a one or two dimensional 
array of float. 
If the arry is one dimensional, it must have a size `ntot` given by
	
	ntot = sum(CMBlkl.lmax)+6 + len(CMBlkl.extra_parameter_names)
	
i.e. the sum of the lenght of the required Cls plus the number of extra nuisance parameters. The Cls must be arranged in the array in the order **TT EE BB TE TB EB**, and must all start at 0 and stop at lmax (included). The values of the nuisance parameters must be appended to the Cls. The Cls are Cls, not Dls!

It the arrayis two dimensional, it must have a shape ``(i,ntot)`` and the code will return `i` evaluations of the log likelihood.

##### New calling API

`clipy` introduces a new, simpler, way to compute a single log likelihood

	loglkl = CMBlkl(cls,nuisance_dict)
	
In this case, `cls` can either be a one dimentional array as described in the previous section, or a `(6,max(CMBlkl.lmax))` 2 dimensional one. In this case, `cls[0]` correspond to **TT**, `cls[1]`, **EE**, etc. following the order **TT EE BB TE TB EB**. `nuisance_dict` is a dictionnary with key the names of the nuisance parameters from `CMBlkl.extra_parameter_names` and value their respective values. `nuisance_dict` can contains elements which are not required by the likelihood. They will be ignored. 

If `cls` is a single dimensional array, it also contains nuisance parameter values. In this case, the nuisance parameters values will be updated by the content of `nuisance_dict` and the dictionnary can omit some of the nuisance parameter names to default to the value found in the `cls` array.

If `cls` is a two dimensional array with shape `(6,max(CMBlkl.lmax))`, `nuisance_dict` must contain all of the nuisance parameter names and values.

Parameter vectors fllowing the classical `clik` calling API and the new spectra and nuisance dictionnary pairs following the new API can be transformed into one another using the `normalize`  (from `clik` API to the new one) and `normalize_clik` (from the new API to the `clik` one) attribute of the clipy objects, as described [above](#New-features-of-clipy-objects).

### Lensing likelihood

Lensing likelihood works the same way as CMB likelihoods. However, the Cl arrays that have to be passed to the likelihood must of course also include the 

## Example code

The script ``clipy_example_py`` demonstrate how to use the python library. It is similar to the `clik_example_py` script from the old `clik` library.

## `candl` compatibility

Instead of following the `clik` API, the likelihood objects initialized with clipy can be made to behave like `candl` likelihood objects. To do so, instead of using the `clipy.clik` object, use the `clipy.clik_candl` one, like in the following example :

	import clipy
	candl_CMBlkl = clipy.clik_candl("/path/to/some/clikfile")

The `candl_CMBlkl` initialized above will partially implement the `candl` API. Only the following methods and attributes are currently implemented :

- `required_nuisance_parameters`: provide the list of nuisance parameters as a list of strings 
- `unique_spec_types`: provides a list of temperature and polarization spectra used in the likelihood
- `log_like(params)`: method that computes the log likelihood for a set of spectra and nuisance parameters. Following the `candl` API, `params` must be a dictionnary containing a set of key-value for each nuisance parameters. It must further contains a dictionnary index by the key `Dl`. This dictionnary contains the $D_\ell=\ell(\ell+1)C_\ell/2\pi$ for each power spectrum using the keys TT, TE, etc in upper case. This is different from the `clik` API that expects power spectra (i.e. $C_\ell$ and not $D_\ell$). Also note that all priors are applied in the computation.
- `chi_square(params)`: method that compute the $\chi^2$ for a set of spectra and nuisance parameters. `params` is defined as for the `log_like` method described above. Contrary to the `log_like` method, the priors **are not** applied.

#### Priors

`candl` expects that the priors are applied in the likelihood. This is not the `clik` and `clipy` default behavior. To emulate the `candl` behavior, the following options must be set at initialization time:

- `all_priors`: setting this option will make sure that all [recommended priors](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code) are applied `candl_CMBlkl_allpriors = clipy.clik_candl("/path/to/some/clikfile", all_priors=True)`
- `A_planck_prior`: only apply the Planck calibration prior $y_{cal} = 1 ± 0.0025$
- `baseline_priors`: for `plik` only, apply all recommended priors on the polarization efficiency, dust and polarized dust amplitude. This option does not apply the planck calibration prior or the joint SZ prior.
- `SZ_prior`: for `plik` only, apply the joint SZ linear combinaison prior  $D_{kSZ} + 1.6 D_{tSZ} = 9.5 ± 3 $.

#### Parameter names 
By default, the `clipy` initialized likelihood object will use the `clik` parameter names. CosmoMC and Cobaya have been using different names for the nuisance parameters. For the `candl` compatible likelihood objects, it is possible to rename the parameters to the CosmoMC expected ones using the option `cosmomc_names`:

	candl_CMBlkl_cosmomc = clipy.clik_candl("/path/to/some/clikfile", cosmomc_names=True)
	
	
#### Initialization time options

`clipy` introduces an [optional initialization time selection](#New-initialisation-time-features) of the likelihood content. `candl` provides a similar possibility adding a `data_selection` option to the initialization call. This option must be either a string (for a single selection order) or a list of strings (for a more complex selection) describing a list of command. This is similar to the `crop` option in `clipy`. However the [`candl` syntax](https://candl.readthedocs.io/en/latest/tutorial/usage_tips.html#data-selection) for those command is slighlty different. When initializing `candl` compatible likelihood object, an initialization time selection of the data content of the likelihood can be made either with the [`clipy` syntax](#New-initialisation-time-features) using the `crop` option, or with the `candl` one, using the `data_selection` option. For example
 
	candl_CMBlkl_crop = clipy.clik_candl("/path/to/some/clikfile", 
		data_selection=["EE 100x100 l<500 remove","TE l>1000 remove"])

	
will use data after removing the 100 GHz EE autospectra bandpowers at $\ell<500$ and discarding all the TE spectra for bandpowers at $\ell>1000$. This is the equivalent of
	
	candl_CMBlkl_crop = clipy.clik_candl("/path/to/some/clikfile", 
		crop = ["crop EE 100x100 500 -1","crop TE -1 1000"])
		








	