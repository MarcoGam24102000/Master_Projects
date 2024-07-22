# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 15:00:33 2023

@author: marco
"""



def estim_prec(set_moves):
    
    print("---- Hear")
    
    for s in set_moves:
        print(" -- " + s)

    sample_moves = ['In',
                    'In',
                    'In',
                    'Out',
                    'Out',
                    'In',
                    'In',
                    'Out',
                    'In',
                    'Out',
                    'Out',
                    'Out',
                    'In',
                    'In',
                    'Out',
                    'Out'
                    ]
    
    
    tot_samples = len(sample_moves)
    
    print(tot_samples)    
   
    right = 0
    prec = 0
    
    if len(set_moves) == tot_samples:
        for ind_sample, sample in enumerate(set_moves):
            if sample == sample_moves[ind_sample]:
                right += 1
            else:
                right += 0
        
        prec = (right/tot_samples)*100
        
        print("Precision was about: " + str(prec) + " % ...")
            
    else:
        print("No the same length to estimate precision")
        prec = -1
    
    return prec 