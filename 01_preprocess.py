import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from xmip.preprocessing import rename_cmip6
import dask
from nc_processing import calc_spatial_mean
from tqdm import tqdm


##### user inputs #####
drop_years_len = 75 # number of years at start of 2xco2 run to ignore
window_length = 20 # length of time slices which define mean-states (years)
piCO2 = 284.3 # ppm (Meishausen, 2017 DOI 10.5194/gmd-10-2057-2017)
mods_to_run = ['CanESM5', 'GISS-E2-1-H', 'HadGEM3-GC31-LL', 
               'IPSL-CM6A-LR', 'MIROC6', 'CESM2',
               'TaiESM1', 'MRI-ESM2-0', 'CNRM-CM6-1'] 
gws_mods = ['CESM2', 'TaiESM1', 'MRI-ESM2-0', 'CNRM-CM6-1']
#N.B several versions of GISS are available for the 2xCO2 run, we restrict to only version here so as not to dominate the statistics


##### functions to read in .nc files on archive to datasets #####
def read_in(dir):
    files = []
    for x in os.listdir(dir):
        files.append(dir + x)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = rename_cmip6(xr.open_mfdataset(files, use_cftime=True, engine='netcdf4'))
    return ds

def read_in_ens_mean(dirs):
    """ returns (1) the ensemble mean dataset, and (2) the number of ensemble members """
    files = []
    for dir in dirs:
        for x in os.listdir(dir):
            if '.nc' in x:
                files.append(dir + x)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = rename_cmip6(xr.open_mfdataset(files, use_cftime=True, concat_dim='ensemble',combine='nested'))
        n_ens = len(ds.ensemble) 
        ds = ds.mean(dim='ensemble')
        ds['number_ens_mems_meaned'] = n_ens
    return ds, n_ens

def read_in_ens(model, variable='pr', scenario='1pctCO2', table='Amon'):
    # note annoying exception first, 
    # which deals with the missing pr data for TaiESM1 on the main archive
    if model == 'TaiESM1' and variable == 'pr':
        ds_list = []
        file =  '/gws/nopw/j04/cpom/aduffey/Transient_baselines/1pctCO2/pr/TaiESM1/pr_Amon_TaiESM1_1pctCO2_r1i1p1f1_gn_000101-015012.nc'
        ds = rename_cmip6(xr.open_mfdataset(file))
        DS = xr.concat([ds], dim='Ensemble_member')
        ens_mems = ['r1i1p1f1']
    else:
        ens_mems = []
        for path in glob.glob('/badc/cmip6/data/CMIP6/*/*/{m}/{s}/*p1*/{t}/{v}/*/latest/'.format(
        m=model, s=scenario, t=table, v=variable)):
            ens_mems.append(path.split('/')[9])
        ds_list = []
        for es in ens_mems:
            path = '/badc/cmip6/data/CMIP6/*/*/{m}/{s}/{e}/{t}/{v}/*/latest/'.format(
                 m=model, s=scenario, e=es, t=table, v=variable)
            #files = os.listdir(path)
            ds = rename_cmip6(xr.open_mfdataset(path+'*.nc'))
            ds_list.append(ds)
        
        DS = xr.concat(ds_list, dim='Ensemble_member')
    return DS, ens_mems



##### calc CO2 arithmetic #####
Double_CO2 =  284.3*2
One_pct_CO2 = [piCO2*(1.01**x) for x in np.arange(0, 150, 1)]


##### 01 - get preindustrial temp for each model and save to dict #####
# first define function to get PI gmst:
def get_pi(model, var='tas'):
    dir_pi = glob.glob('/badc/cmip6/data/CMIP6/*/*/{m}/piControl/r1i*/Amon/{v}/*/latest/'.format(m=model, v=var))
    files_pi = os.listdir(dir_pi[0])[0:3] # don't need the full length run
    paths_pi = []
    for x in files_pi:
        paths_pi.append(dir_pi[0]+x)
    da_pi = rename_cmip6(xr.open_mfdataset(paths_pi))[var].mean(dim='x')
    da_pi = da_pi.weighted(weights=np.cos(np.deg2rad(da_pi.y))).mean(dim='y')
    
    gmst_pi = da_pi.mean(dim='time').values
    print(gmst_pi)
    return gmst_pi

gmst_pis = []
for mod in tqdm(mods_to_run):
    print(mod)
    gmst_pi = get_pi(mod)
    gmst_pis.append(gmst_pi)
gmst_pi_dict = dict(zip(mods_to_run, gmst_pis))

# repeat for preindistrial Pr
pr_pis = []
for mod in tqdm(mods_to_run):
    print(mod)
    pr_pi = get_pi(mod, var='pr')
    pr_pis.append(pr_pi)
pr_pi_dict = dict(zip(mods_to_run, pr_pis))

# get landfracs by model:
land_fracs=[]
for mod in mods_to_run:
    dir_pi = glob.glob('/badc/cmip6/data/CMIP6/*/*/{}/piControl/r1i*/fx/sftlf/*/latest/'.format(mod))
    files_pi = os.listdir(dir_pi[0])
    paths_pi = []
    for x in files_pi:
        paths_pi.append(dir_pi[0]+x)
    land_frac = rename_cmip6(xr.open_mfdataset(paths_pi))
    land_fracs.append(land_frac)
land_frac_dict = dict(zip(mods_to_run, land_fracs))

# also get gmst_pi_dicts by model for land and sea only:
def get_pi_region(model, region, land_frac=None):
    if not land_frac:
        land_frac = land_frac_dict[model] 
    dir_pi = glob.glob('/badc/cmip6/data/CMIP6/*/*/{}/piControl/r1i*/Amon/tas/*/latest/'.format(model))
    files_pi = os.listdir(dir_pi[0])[0:3] # don't need the full length run
    paths_pi = []
    for x in files_pi:
        paths_pi.append(dir_pi[0]+x)
    da_pi = rename_cmip6(xr.open_mfdataset(paths_pi)).tas
    if region == 'land':
        da_pi = da_pi.where(land_frac.sftlf>99, drop=True).mean(dim='x')
    elif region == 'ocean':
        da_pi = da_pi.where(land_frac.sftlf<1, drop=True).mean(dim='x')
    da_pi = da_pi.weighted(weights=np.cos(np.deg2rad(da_pi.y))).mean(dim='y')
    
    gmst_pi = da_pi.mean(dim='time').values
    print(gmst_pi)
    return gmst_pi

gmst_pis_land, gmst_pis_ocean = [], []
for mod in tqdm(mods_to_run):
    print(mod)
    gmst_pi_land, gmst_pi_ocean = get_pi_region(mod, 'land'), get_pi_region(mod, 'ocean')
    gmst_pis_land.append(gmst_pi_land)
    gmst_pis_ocean.append(gmst_pi_ocean)
    
gmst_pi_dict_land = dict(zip(mods_to_run, gmst_pis_land))
gmst_pi_dict_ocean = dict(zip(mods_to_run, gmst_pis_ocean))

##### 02 - set up functions for main processing #####

def get_gmst(model, scen='abrupt-2xCO2'):
    if model in gws_mods:
        dir_4x = glob.glob('/gws/nopw/j04/cpom/aduffey/Transient_baselines/abrupt-2xCO2/tas/{m}/'.format(m=model))
    else:
        dir_4x = glob.glob('/badc/cmip6/data/CMIP6/*/*/{m}/{s}/r1i*/Amon/tas/*/latest/'.format(m=model, s=scen))
    files_4x = os.listdir(dir_4x[0]) 
    paths_4x = []
    for x in files_4x:
        paths_4x.append(dir_4x[0]+x)
    da_4x = rename_cmip6(xr.open_mfdataset(paths_4x)).tas.mean(dim='x')
    da_4x = da_4x.weighted(weights=np.cos(np.deg2rad(da_4x.y))).mean(dim='y')
    da_4x = da_4x.groupby("time.year").mean(dim="time")
    da_4x = da_4x.isel(year=slice(drop_years_len,1000)) #drop first N years 
    gmst_4x = da_4x.mean(dim='year').values
    return gmst_4x
    
def get_stable_means(model, var, scen='abrupt-2xCO2', chunks=False):
    if model in gws_mods:
        dir_4x = glob.glob('/gws/nopw/j04/cpom/aduffey/Transient_baselines/abrupt-2xCO2/{v}/{m}/'.format(v=var, m=model))
    else:
        dir_4x = glob.glob('/badc/cmip6/data/CMIP6/*/*/{m}/{s}/r1i*/Amon/{v}/*/latest/'.format(m=model, s=scen, v=var))
    files_4x = os.listdir(dir_4x[0]) 
    paths_4x = []
    for x in files_4x:
        paths_4x.append(dir_4x[0]+x)
    da_4x = rename_cmip6(xr.open_mfdataset(paths_4x))[var].mean(dim='x')
    da_4x = da_4x.weighted(weights=np.cos(np.deg2rad(da_4x.y))).mean(dim='y')
    da_4x = da_4x.groupby("time.year").mean(dim="time")
    
    da_4x = da_4x.isel(year=slice(drop_years_len,1000)) #drop first n years 
    
    gmst_4x = da_4x.mean(dim='year').values
    return gmst_4x

def get_stable_scen_ds(model, var, scen='abrupt-2xCO2'):
    if model in gws_mods:
        dir_4x = glob.glob('/gws/nopw/j04/cpom/aduffey/Transient_baselines/abrupt-2xCO2/{v}/{m}/'.format(v=var, m=model))
    else:
        dir_4x = glob.glob('/badc/cmip6/data/CMIP6/*/*/{m}/{s}/r1i*/Amon/{v}/*/latest/'.format(m=model, s=scen, v=var))
    files_4x = os.listdir(dir_4x[0]) 
    paths_4x = []
    for x in files_4x:
        paths_4x.append(dir_4x[0]+x)
    da_4x = rename_cmip6(xr.open_mfdataset(paths_4x))[var].mean(dim='x')
    da_4x = da_4x.weighted(weights=np.cos(np.deg2rad(da_4x.y))).mean(dim='y')
    da_4x = da_4x.groupby("time.year").mean(dim="time")
    return da_4x

def get_var_at_crit_T(mod, var,  
                      window_length=window_length, 
                      stable_scen='abrupt-2xCO2',
                      check_plot=False):
    """ produce a spatial nc file, with map of mean values during window when 
    model and scen pass a gmst temperature threshold crit_temp
     """
    
    crit_temp = get_gmst(model=mod, scen=stable_scen)
    stable_var = get_stable_means(model=mod,var=var, scen='abrupt-2xCO2', chunks=False)
                            
    ds_tas, ens_mems = read_in_ens(model=mod, variable='tas', scenario='1pctCO2', table='Amon')
    ds_var, ens_mems = read_in_ens(model=mod, variable='pr', scenario='1pctCO2', table='Amon')
    
    ds_tas = calc_spatial_mean(ds_tas['tas'], lon_name="x", lat_name="y").groupby("time.year").mean(dim="time")
    ds_var = calc_spatial_mean(ds_var[var], lon_name="x", lat_name="y").groupby("time.year").mean(dim="time")
    
    year_0 = ds_tas.year.values[0]                       
    df = pd.DataFrame({'Year': ds_tas.year.values})

    crossing_years, crossing_tass, crossing_vars, crossing_rates, crossing_CO2s = [], [], [], [], []
    tas_ts_list, pr_ts_list = [], []
    for ens_mem in ds_tas.Ensemble_member:
        #ens_mems.append(ens_mem.values)
        ds_tas_em = ds_tas.sel(Ensemble_member=ens_mem)
        ds_var_em = ds_var.sel(Ensemble_member=ens_mem)
        #print(ds_tas_em)
        df['gmst'] = ds_tas_em.values
        #print(df['gmst'])
        df['gmst_rolling'] = df['gmst'].rolling(window=window_length, center=True).mean()
        df[var] = ds_var_em.values
        name = str(var+'_rolling')
        df[name] = df[var].rolling(window=window_length, center=True).mean()
        # also add column for rate of change of T:
        gmst_rolling_dTdt = np.diff(df['gmst_rolling'])
    
        if check_plot:
            plt.plot(df['Year'], df['gmst_rolling'])
            plt.axhline(crit_temp)
            plt.show()
    
        One_pct_CO2 = [piCO2*(1.01**x) for x in np.arange(0, len(df['Year']), 1)]
        # we update temps here to be anomalies relative to that model's PI
        crossing_year = np.interp(crit_temp, df['gmst_rolling'], df['Year'])
        
        crossing_years.append(crossing_year)
        crossing_tass.append(np.interp(crossing_year, df['Year'], df['gmst_rolling']) - gmst_pi_dict[mod])
        crossing_vars.append(np.interp(crossing_year, df['Year'], df[name]))
        crossing_rates.append(np.interp(crossing_year, df['Year'][:-1], gmst_rolling_dTdt))
        crossing_CO2s.append(np.interp(crossing_year, df['Year'], One_pct_CO2))
        tas_ts_list.append(df['gmst']- gmst_pi_dict[mod])
        pr_ts_list.append(df['pr'])
    
    out_dict = {'Scenario':scen,
                'Model':mod,
                'tas_stable':crit_temp - gmst_pi_dict[mod],
                '{}_stable'.format(var):stable_var,
                #'{}_stable_chunks'.format(var):stable_var_chunks,
                'Ensemble_members':ens_mems,
                'Year_0_1pctCO2':year_0,
                'Crossing_years':crossing_years,
                'tas_at_crossing_years':crossing_tass,
                '{}_at_crossing_years'.format(var):crossing_vars,
                'CO2_at_crossing_years':crossing_CO2s,
                'Warming_rates_at_crossing_year':crossing_rates,
                'tas_ts_1pctCO2':tas_ts_list,
                'pr_ts_1pctCO2':pr_ts_list,
                'CO2_ts_1pctCO2':One_pct_CO2
               }
                #'Warming_rate_at_crossing_year_2':crossing_rate_2}
    return out_dict, df

## main event, landsea

def get_stable_means_landsea(model, land_frac_dict, scen='abrupt-2xCO2', chunks=False):
    if model in gws_mods:
        dir = glob.glob('/gws/nopw/j04/cpom/aduffey/Transient_baselines/abrupt-2xCO2/{v}/{m}/'.format(v=var, m=model))
    else:
        dir = glob.glob('/badc/cmip6/data/CMIP6/*/*/{m}/{s}/r1i*/Amon/{v}/*/latest/'.format(m=model, s=scen, v=var))
    
    files = os.listdir(dir[0]) 
    paths = []
    for x in files:
        paths.append(dir[0]+x)
    da = rename_cmip6(xr.open_mfdataset(paths))['tas']
    da = da.groupby("time.year").mean(dim="time")
    da = da.isel(year=slice(drop_years_len,200)) #drop first n years 

    land_frac = land_frac_dict[model]
    
    da_all = da.mean(dim='x')
    da_all = da_all.weighted(weights=np.cos(np.deg2rad(da_all.y))).mean(dim='y')
    gmst_all = da_all.mean(dim='year').values
    
    da_land = da.where(land_frac.sftlf>99, drop=True).mean(dim='x')
    da_land = da_land.weighted(weights=np.cos(np.deg2rad(da_land.y))).mean(dim='y')
    gmst_land = da_land.mean(dim='year').values
    
    da_ocean = da.where(land_frac.sftlf<1, drop=True).mean(dim='x')
    da_ocean = da_ocean.weighted(weights=np.cos(np.deg2rad(da_ocean.y))).mean(dim='y')
    gmst_ocean = da_ocean.mean(dim='year').values
    
    return gmst_all, gmst_land, gmst_ocean

def get_landsea_at_crit_T(mod, land_frac_dict,
                          window_length=window_length, 
                          stable_scen='abrupt-2xCO2',
                          check_plot=False):
    
    crit_temp = get_gmst(model=mod, scen=stable_scen)
    gmst_all, gmst_land, gmst_ocean = get_stable_means_landsea(model=mod, land_frac_dict=land_frac_dict, scen='abrupt-2xCO2', chunks=False)
                              
    ds_tas, ens_mems = read_in_ens(model=mod, variable='tas', scenario='1pctCO2', table='Amon')
    ds_tas = ds_tas.groupby("time.year").mean(dim="time")
    
    land_frac = land_frac_dict[mod]
                          
    df = pd.DataFrame({'Year': ds_tas.year.values})

    crossing_years, crossing_tass, crossing_tass_land, crossing_tass_ocean, crossing_rates = [], [], [], [], []
    tas_land_ts_list, tas_ocean_ts_list, tas_global_ts_list = [], [], []
    for ens_mem in ds_tas.Ensemble_member:
        ds_tas_em = ds_tas.sel(Ensemble_member=ens_mem).tas
        
        ds_tas_em_all = ds_tas_em.mean(dim='x')
        ds_tas_em_all = ds_tas_em_all.weighted(weights=np.cos(np.deg2rad(ds_tas_em_all.y))).mean(dim='y')
        
        ds_tas_land_em = ds_tas_em.where(land_frac.sftlf>99, drop=True).mean(dim='x')
        ds_tas_land_em = ds_tas_land_em.weighted(weights=np.cos(np.deg2rad(ds_tas_land_em.y))).mean(dim='y')
        
        ds_tas_ocean_em = ds_tas_em.where(land_frac.sftlf<1, drop=True).mean(dim='x')
        ds_tas_ocean_em = ds_tas_ocean_em.weighted(weights=np.cos(np.deg2rad(ds_tas_ocean_em.y))).mean(dim='y')

    
        df['gmst'] = ds_tas_em_all.values
        df['gmst_rolling'] = df['gmst'].rolling(window=window_length, center=True).mean()   
        df['gmst_land'] = ds_tas_land_em.values
        df['gmst_ocean'] = ds_tas_ocean_em.values
        df['gmst_land_rolling'] = df['gmst_land'].rolling(window=window_length, center=True).mean()
        df['gmst_ocean_rolling'] = df['gmst_ocean'].rolling(window=window_length, center=True).mean()
        
        # also add column for rate of change of T:
        gmst_rolling_dTdt = np.diff(df['gmst_rolling'])
        
        if check_plot:
            plt.plot(df['Year'], df['gmst_rolling'])
            plt.axhline(crit_temp)
            plt.show()
    
        
        # we update temps here to be anomalies relative to that model's PI
        crossing_year = np.interp(crit_temp, df['gmst_rolling'], df['Year'])
        crossing_years.append(crossing_year)
        crossing_tass.append(np.interp(crossing_year, df['Year'], df['gmst_rolling']) - gmst_pi_dict[mod])
        crossing_tass_land.append(np.interp(crossing_year, df['Year'], df['gmst_land_rolling']) - gmst_pi_dict_land[mod])
        crossing_tass_ocean.append(np.interp(crossing_year, df['Year'], df['gmst_ocean_rolling']) - gmst_pi_dict_ocean[mod])
        crossing_rates.append(np.interp(crossing_year, df['Year'][:-1], gmst_rolling_dTdt))
        tas_land_ts_list.append(df['gmst_land']- gmst_pi_dict_land[mod])
        tas_ocean_ts_list.append(df['gmst_ocean']- gmst_pi_dict_ocean[mod])
        tas_global_ts_list.append(df['gmst'] - gmst_pi_dict[mod])

    out_dict = {'Scenario':scen,
                'Model':mod,
                'tas_stable':crit_temp - gmst_pi_dict[mod],
                'tas_land_stable':gmst_land - gmst_pi_dict_land[mod],
                'tas_ocean_stable':gmst_ocean - gmst_pi_dict_ocean[mod],
                'Ensemble_members':ens_mems,
                'Crossing_years':crossing_years,
                'tas_at_crossing_years':crossing_tass,
                'tas_land_at_crossing_years':crossing_tass_land,
                'tas_ocean_at_crossing_years':crossing_tass_ocean,
                'Warming_rates_at_crossing_year':crossing_rates,
                'tas_land_ts':tas_land_ts_list,
                'tas_ocean_ts':tas_ocean_ts_list,
                'tas_global_ts':tas_global_ts_list}
                
    return out_dict, df

##### 03 - run and save #####
print('part 1 - running pr and co2')

out_dicts = []
df_list = []
mod_list = []

scen='1pctCO2'

for model in mods_to_run:
    print(model)
    out_dict, df = get_var_at_crit_T(mod=model, var='pr')
    
    out_dicts.append(out_dict)
    df_list.append(df) 
    mod_list.append(model)
    
DF = pd.DataFrame()
for i in range(len(mods_to_run)):
    d = out_dicts[i]
    mod = d['Model']
    
    df1 = pd.DataFrame({'Model':mod,
                        'Exp.':'2xCO2',
                        #'pr':d['pr_stable_chunks'],
                        'pr':[d['pr_stable']],
                        'CO2 (ppm)':[Double_CO2]})
    df2 = pd.DataFrame({'Model':mod,
                        'Exp.':'1pctCO2',
                        'pr':d['pr_at_crossing_years'],
                        'CO2 (ppm)':d['CO2_at_crossing_years']})
            
    DF = pd.concat([DF, df1])
    DF = pd.concat([DF, df2])

unit_conversion = 86400
DF['pr (mm/day)'] = DF['pr']*unit_conversion
DF['Model_short'] = [x.split('-')[0] for x in DF['Model']]
DF.to_csv('int_outs/Pr_CO2_main_df_out.csv')

print('part 2 - running landsea warming ratio')

out_dicts_ls = []
df_list = []
mod_list = []

var = 'tas'

for model in mods_to_run:
    print(model)
    out_dict_ls, df =  get_landsea_at_crit_T(mod=model, 
                                             land_frac_dict=land_frac_dict,
                                             window_length=window_length,
                                             check_plot=False)
    
    out_dicts_ls.append(out_dict_ls)
    df_list.append(df) 
    mod_list.append(model)
    
DF_landsea = pd.DataFrame()
for i in range(len(mods_to_run)):
    d = out_dicts_ls[i]
    mod = d['Model']
    
    df1 = pd.DataFrame({'Model':mod,
                        'Exp.':'2xCO2',
                        'tas_land':[d['tas_land_stable']],
                        'tas_ocean':[d['tas_ocean_stable']]})
                        #'tas_land':d['tas_land_stable_chunks'],
                        #'tas_ocean':d['tas_ocean_stable_chunks']})
    df2 = pd.DataFrame({'Model':mod,
                        'Exp.':'1pctCO2',
                        'Warming_rate_at_crossing':d['Warming_rates_at_crossing_year'],
                        'tas_at_crossing':d['tas_at_crossing_years'],
                        'tas_land':d['tas_land_at_crossing_years'],
                        'tas_ocean':d['tas_ocean_at_crossing_years']})
            
    DF_landsea = pd.concat([DF_landsea, df1])
    DF_landsea = pd.concat([DF_landsea, df2])

DF_landsea['Model_short'] = [x.split('-')[0] for x in DF_landsea['Model']]
DF_landsea['tas_ratio'] = DF_landsea['tas_land']/DF_landsea['tas_ocean']
DF_landsea.to_csv('int_outs/landsearatio_main_out_df.csv')

#df_out = pd.merge(DF, DF_landsea, on=['Model', 'Exp.', 'Model_short'], how='left')
#df_out.to_csv('int_outs/combined_main_out_df.csv')

# also save the full time series data from the dicts:
import pickle
for d in out_dicts:
    mod = d['Model']
    print(mod)
    with open('int_outs/{}_dict.pickle'.format(mod), 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

for d in out_dicts_ls:
    mod = d['Model']
    print(mod)
    with open('int_outs/{}_LS_dict.pickle'.format(mod), 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
