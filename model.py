import hddm

# behavioral model m1-m5
def run_m1(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    m = hddm.HDDM(df)
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m



def run_m2(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    m = hddm.HDDM(df, include = ['z'])
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

def run_m3(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    m = hddm.HDDMRegressor(df,"v ~ C(coherence, Treatment('low'))" ,include=['z'],keep_regressor_trace=True, group_only_regressors = False)
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m


def run_m4(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~1+C(coherence, Treatment('low'))", 'link_func': lambda x:x}
    
    z_reg={'model': "z~1+C(prioritized, Treatment('no'))", 'link_func': lambda x:x}

    m = hddm.HDDMRegressor(df, [v_reg, z_reg] ,include=['z'],keep_regressor_trace=True, group_only_regressors = False)
 
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m


def run_m5(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~1+C(coherence, Treatment('low'))", 'link_func': lambda x:x}
    
    t_reg={'model': "t~1+C(prioritized, Treatment('no'))", 'link_func': lambda x:x}

    m = hddm.HDDMRegressor(df, [v_reg, t_reg] ,include=['z'],keep_regressor_trace=True, group_only_regressors = False)
 
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

# multiverse
# trial-wise
def slps(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~C(coherence, Treatment('low'))*slps", 'link_func': lambda x:x}
    
    t_reg={'model': "t~1+C(prioritized, Treatment('no'))", 'link_func': lambda x:x}

    m = hddm.HDDMRegressor(df, [v_reg, t_reg] ,include=['z'], keep_regressor_trace=True, group_only_regressors = False)
 
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

def ams(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~C(coherence, Treatment('low'))*ams", 'link_func': lambda x:x}
    
    t_reg={'model': "t~1+C(prioritized, Treatment('no'))", 'link_func': lambda x:x}

    m = hddm.HDDMRegressor(df, [v_reg, t_reg] ,include=['z'], keep_regressor_trace=True, group_only_regressors = False)
 
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

def pams(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~C(coherence, Treatment('low'))*pams", 'link_func': lambda x:x}
    
    t_reg={'model': "t~1+C(prioritized, Treatment('no'))", 'link_func': lambda x:x}

    m = hddm.HDDMRegressor(df, [v_reg, t_reg] ,include=['z'], keep_regressor_trace=True, group_only_regressors = False)
 
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

# bin-wise
def slp_bin(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~C(coherence, Treatment('low'))*slp_bin", 'link_func': lambda x:x}
    
    t_reg={'model': "t~1+C(prioritized, Treatment('no'))", 'link_func': lambda x:x}

    m = hddm.HDDMRegressor(df, [v_reg, t_reg] ,include=['z'], keep_regressor_trace=True, group_only_regressors = False)
 
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

def am_bin(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~C(coherence, Treatment('low'))*am_bin", 'link_func': lambda x:x}
    
    t_reg={'model': "t~1+C(prioritized, Treatment('no'))", 'link_func': lambda x:x}

    m = hddm.HDDMRegressor(df, [v_reg, t_reg] ,include=['z'], keep_regressor_trace=True, group_only_regressors = False)
 
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

def pam_bin(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~C(coherence, Treatment('low'))*pam_bin", 'link_func': lambda x:x}
    
    t_reg={'model': "t~1+C(prioritized, Treatment('no'))", 'link_func': lambda x:x}

    m = hddm.HDDMRegressor(df, [v_reg, t_reg] ,include=['z'], keep_regressor_trace=True, group_only_regressors = False)
 
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

# condition-wise
def slp_cond(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~C(coherence, Treatment('low'))*slp_cond", 'link_func': lambda x:x}
    
    t_reg={'model': "t~1+C(prioritized, Treatment('no'))", 'link_func': lambda x:x}

    m = hddm.HDDMRegressor(df, [v_reg, t_reg] ,include=['z'], keep_regressor_trace=True, group_only_regressors = False)
 
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

def am_cond(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~C(coherence, Treatment('low'))*am_cond", 'link_func': lambda x:x}
    
    t_reg={'model': "t~1+C(prioritized, Treatment('no'))", 'link_func': lambda x:x}

    m = hddm.HDDMRegressor(df, [v_reg, t_reg] ,include=['z'], keep_regressor_trace=True, group_only_regressors = False)
 
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

def pam_cond(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~C(coherence, Treatment('low'))*pam_cond", 'link_func': lambda x:x}
    
    t_reg={'model': "t~1+C(prioritized, Treatment('no'))", 'link_func': lambda x:x}

    m = hddm.HDDMRegressor(df, [v_reg, t_reg] ,include=['z'], keep_regressor_trace=True, group_only_regressors = False)
 
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m
