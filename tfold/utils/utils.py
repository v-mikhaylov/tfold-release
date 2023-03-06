#Victor Mikhaylov, vmikhayl@ias.edu
#Institute for Advanced Study, 2021-2022

import os
import numpy as np
import pandas as pd
import csv
import stat

#from matplotlib import pyplot as plt

def print_hist(a,order=0,use_template=True,max_n=None):
    '''
    print histogram for list/array a; 
    order==0 (default): sort by alphabet;    
    order>0: sort by counts, increasing;
    order<0: sort by counts, decreasing
    '''
    a=np.array(a)
    a_u=np.unique(a)    
    a_n=np.array([np.sum(a==x) for x in a_u])
    if order==0:
        ind=np.argsort(a_u)
    else:
        ind=np.argsort(order*a_n)
    max_len_a=max([len(str(x)) for x in a_u])
    max_len_n=max([len(str(x)) for x in a_n])
    if max_n:
        ind=ind[:max_n]
    if use_template:
        template='{:'+str(max_len_a)+'s} {:'+str(max_len_n)+'n}'
        for i in ind:
            print(template.format(str(a_u[i]),a_n[i]))
    else:
        for i in ind:
            print(a_u[i],a_n[i])

def cluster(a,distance,threshold=0):
    '''
    cluster elements in list 'a' according to the condition distance(x,y)<=threshold;
    returns a list of lists of elements in clusters;
    the dumbest alg possible
    '''
    clusters=[]
    for x in a:
        c_linked=[]
        for i,c in enumerate(clusters):
            #check if element linked to cluster
            for y in c:
                if distance(x,y)<=threshold: 
                    c_linked.append(i)
                    break    
        #merge clusters in c_links + new element into a new cluster    
        merged=[el for i in c_linked for el in clusters[i]]
        merged.append(x)    
        #update clusters    
        clusters=[c for i,c in enumerate(clusters) if i not in c_linked]
        clusters.append(merged)
    return clusters

def split_jobs(n,m):
    '''
    split n jobs over <=m actors;        
    first minimize max number of jobs of one actor, then give p jobs to everybody and <=p to the last one;    
    returns a list of lists of sizes n1,..,nk, k<=m, with indices of jobs assigned to actors (indices randomly permuted)    
    '''    
    #p is max number of jobs of one actor
    p=n//m+int(n%m!=0)
    ind=np.random.permutation(n)
    output=[]
    n_assigned=0
    i=0
    while n_assigned<n:
        output.append(ind[i*p:(i+1)*p])
        i+=1
        n_assigned+=p
    return output    

def make_task(inputs,n_tasks,
              sh_path,python_path,input_dir,input_dir_server=None,
              qos='short',max_run_time=720,slow_nodes=None,gpu=False,exclusive=False,local=False,argstring='',
              cd=None):
    '''
    takes a list of inputs; each input is a list of arguments to be passed to the processing algorithm;
    (can be of len 1 if only one argument);
    takes n_tasks -- number of separate tasks to cut the job into; 
    takes sh_path for the sh script, python_path for the processing alg;
    takes input_dir; in input_dir/inputs will put inputs as tsv files, and will use input_dir/logs to pipe outputs and errors;
    optionally takes input_dir_server (default input_dir) as path to input_dir as seen by the processing server;
    takes slurm qos (default 'short');
    takes max_run_time in minutes (default 720);
    use slow_nodes to give a list of node numbers (e.g. [39,60]) to exclude;
    set flag gpu to use GPUs;
    if local (default False), makes separate jobs without slurm commands;
    if argstring is given, will be added as argument after the input file name;
    if cd (a path) is given, script will have cd $cd (e.g. to a dir that contains a module);
    creates input_%i.tsv files and .sh script    
    '''        
    input_dir_server       =input_dir_server or input_dir
    input_dir_proper       =input_dir+'/inputs'
    input_dir_server_proper=input_dir_server+'/inputs'
    log_dir                =input_dir+'/logs'
    log_dir_server         =input_dir_server+'/logs'
    #make dirs
    for d in [input_dir_proper,log_dir]:
        os.makedirs(d,exist_ok=True)    
    #make input files
    indices=split_jobs(len(inputs),n_tasks)
    n_tasks=len(indices)
    for i,job in enumerate(indices):
        c_inputs=[inputs[j] for j in job]        
        with open(input_dir_proper+'/input_'+str(i)+'.tsv','w',newline='') as f:
            f_csv=csv.writer(f,delimiter='\t')
            for line in c_inputs:
                f_csv.writerow(line)            
    #make .sh        
    if not local:
        lines=[]
        lines.append('#!/bin/bash')
        lines.append('')        
        for x in slow_nodes:
            lines.append('#SBATCH --exclude=node{}'.format(x))     #exclude slow nodes        
        lines.append(f'#SBATCH --array=0-{n_tasks-1}')             #run an array of n_tasks tasks
        lines.append(f'#SBATCH --output={log_dir_server}/output_%a.txt')  #here %a will evaluate to array task id
        lines.append(f'#SBATCH --error={log_dir_server}/error_%a.txt')
        lines.append(f'#SBATCH --ntasks=1')                        #each array element is one task
        if exclusive:
            lines.append(f'#SBATCH --exclusive')                   #request exclusive use of node, e.g. for hhblits
        if gpu:
            lines.append(f'#SBATCH --gpus=1')                      #one gpu per task will use one core
        elif not exclusive:
            lines.append(f'#SBATCH --cpus-per-task=1')             #one cpu per task
        lines.append(f'#SBATCH --qos={qos}')  
        lines.append(f'#SBATCH --time={max_run_time}')             #maximal run time in minutes, same for each array task
        lines.append('')
        if cd:
            lines.append(f'cd {cd}')
        lines.append(f'input_file={input_dir_server_proper}/input_$SLURM_ARRAY_TASK_ID.tsv')        
        lines.append(f'srun python {python_path} $input_file {argstring}')
    else:        
        lines=[]
        lines.append('#!/bin/bash')
        lines.append('')         
        if cd:
            lines.append(f'cd {cd}')
        for i in range(n_tasks):
            lines.append(f'input_file{i}={input_dir_server_proper}/input_{i}.tsv')
            lines.append(f'python {python_path} $input_file{i} {argstring} > {log_dir}/output_{i}.txt 2> {log_dir}/error_{i}.txt')          
    if not sh_path.startswith(('/','./')):
        sh_path='./'+sh_path
    with open(sh_path,'w') as f:
        f.writelines('\n'.join(lines))
    os.chmod(sh_path, stat.S_IRWXU | stat.S_IXGRP | stat.S_IXOTH)      
    return n_tasks

#ROC and plotting
def ROC(labels,predictions): 
    '''
    Compute data for a ROC.
    Takes np.arrays for labels (0s for neg, 1s for pos) and classifier scores (bigger=>more pos, any range),
    returns [fpr, tpr], AUC, where 
    fpr=array[(neg | score >= thr)/(all neg) for possible thr],
    tpr=array[(pos | score >= thr)/(all pos) for possible thr], 
    '''
    s_p=predictions[np.nonzero(labels==1)[0]] #predictions for true positives
    s_n=predictions[np.nonzero(labels==0)[0]] #predictions for true negatives
    n_p=len(s_p)
    n_n=len(s_n) 
    TP=[0]
    FP=[0]
    auc=0
    pts=np.sort(np.unique(predictions))[::-1]    
    for x in pts:
        TP.append(TP[-1]+np.sum(s_p==x))
        FP.append(FP[-1]+np.sum(s_n==x))
        auc+=(TP[-1]+TP[-2])*(FP[-1]-FP[-2])/2
    TP=np.array(TP)
    FP=np.array(FP)
    return [FP/n_n,TP/n_p],auc/(n_p*n_n)

colors=['k','b','g','y','r','c','m','k','k','k','k','k']
def plot_ROC(scores,titles):
    '''
    Takes lists of scores (np.arrays) [[labels0,predictions0],..] and titles [l0, l1..].
    Plots the ROC curves, AUCs,
    and also the number of positive examples for each title.
    '''
    labels_auc=[]
    for i,s in enumerate(scores):
        n_pos=int(np.sum(s[0]))
        if n_pos>0:
            roc_i,auc_i=ROC(s[0],s[1])
            labels_auc.append('{:6s}({:5d}): {:5.3f}'.format(titles[i],n_pos,auc_i))
            plt.plot(roc_i[0],roc_i[1],color=colors[i])
        else:
            labels_auc.append('{:6s}({:5d})'.format(titles[i],0))
            plt.plot([0],[0],color=colors[i])
    plt.plot([0,1],[0,1],linestyle='--',color='black',label='_nolegend_')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    plt.legend(labels_auc) 
    
    
#stats
def agresti_coull(n,N):
    '''
    Agresti-Coull binomial 95% confidence interval (z=2);
    returns edge estimates
    ''' 
    n1,N1=n+2,N+4
    p1=n1/N1
    sigma=(p1*(1-p1)/N1)**0.5
    pm,pp=p1-2*sigma,p1+2*sigma
    return max(0.,pm),min(1.,pp)
def bootstrap(df,func_d,N=1000,plot=False):
    '''
    takes a df and a dict of functions, optionally N and whether to plot;
    returns a df with bootstrapped distributions of values of the functions
    '''
    d={k:[] for k in func_d}
    for i in range(N):
        df1=df.sample(frac=1.,replace=True)
        for k in func_d:            
            d[k].append(func_d[k](df1))
    d=pd.DataFrame(d)
    if plot:
        n=len(d.columns)
        plt.figure(figsize=(5*n,5))
        for i,c in enumerate(d.columns):
            plt.subplot(1,n,i+1)
            plt.hist(d[c].values,histtype='step',bins=100)
            plt.title(c)
        plt.show()
    return d
def get_CI(df): 
    '''get means and CIs for df columns (returns dict of tuples)'''
    d={}
    for k in df.columns:
        d[k]=(df[k].mean(),df[k].quantile(0.025),df[k].quantile(0.975))
    return d
def proportions_p_value(n1,N1,n2,N2,n_repeats=10000):
    '''
    significance of difference between proportions via simulation
    (idea from Coursera; should be used when numbers are small so that the normal approx doesn't work)
    fix total N and total failures, shuffle, see distribution of f1-f2;
    takes numbers of successes and totals for two conditions, returns two-sided p-value
    '''
    n=n1+n2
    N=N1+N2
    x=np.concatenate([np.ones(n),np.zeros(N-n)])
    deltas=[]
    for i in range(n_repeats):
        x=np.random.permutation(x)
        n1_c=np.sum(x[:N1])
        n2_c=n-n1_c
        deltas.append(n1_c/N1-n2_c/N2)
    p=np.sum(np.abs(deltas)>=abs(n1/N1-n2/N2))/len(deltas)
    return p



