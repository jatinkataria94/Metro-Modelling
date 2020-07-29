import numpy as np
import itertools
import matplotlib.pyplot as plt


#Initializing metro variables
actual_avg_footfall=13333
ph,nph=8,10
junc,non_junc=8,28
capacity=3000
ph_freq,nph_freq=167,196
dwell=30
ph_train_rate=round(3600/(ph_freq+dwell))
nph_train_rate=round(3600/(nph_freq+dwell))

#Creating a list of tuning parameters (w:weights given to station types; r:ratio of walk-ins,walk-outs)
w1=w2=np.linspace(0,1,21).tolist()
w=[[round(i,2),round(j,2)] for i in w1 for j in w2 if (round(i,2)+round(j,2)==1.0) ]
w=w[1:-1]
r1=r2=np.linspace(0,1,21).tolist()
r=[[round(i,2),round(j,2)] for i in r1 for j in r2 if (round(i,2)+round(j,2)==1.0) ]
r=r[1:-1]


def neg_X_direction(capacity,weight_junc,weight_non_junc,ph_train_rate,nph_train_rate,r_wo,r_wi):
    #for peak hours
    if ph_train_rate!=0:
        ph_inp_train_in=capacity*pow((1-weight_non_junc),17)*pow((1-weight_junc),6)*ph_train_rate
        ph_out_train_out=(ph_inp_train_in/ph_train_rate)*(1-weight_junc)*ph_train_rate
        ph_out_walk_out=r_wo*(ph_inp_train_in-ph_out_train_out)
        ph_inp_walk_in=r_wi*(ph_inp_train_in-ph_out_train_out)*2
    else:
        ph_inp_train_in=ph_out_train_out=ph_out_walk_out=ph_inp_walk_in=0
        
    #for non peak hours
    if nph_train_rate!=0:
        nph_inp_train_in=capacity*pow((1-weight_non_junc),17)*pow((1-weight_junc),6)*nph_train_rate
        nph_out_train_out=(nph_inp_train_in/nph_train_rate)*(1-weight_junc)*nph_train_rate
        nph_out_walk_out=r_wo*(nph_inp_train_in-nph_out_train_out)
        nph_inp_walk_in=r_wi*(nph_inp_train_in-nph_out_train_out)*2
    else:
        nph_inp_train_in=nph_out_train_out=nph_out_walk_out=nph_inp_walk_in=0
    #combined 
    inp_train_in_neg_x=(ph_inp_train_in+nph_inp_train_in)/2
    out_train_out_neg_x=(ph_out_train_out+nph_out_train_out)/2
    out_walk_out_neg_x=(ph_out_walk_out+nph_out_walk_out)/2
    inp_walk_in_neg_x=(ph_inp_walk_in+nph_inp_walk_in)/2
    
    return inp_train_in_neg_x,out_train_out_neg_x,out_walk_out_neg_x,inp_walk_in_neg_x


def pos_X_direction(capacity,weight_junc,weight_non_junc,ph_train_rate,nph_train_rate,r_wo,r_wi):
    #for peak hours
    if ph_train_rate!=0:
        ph_inp_train_in=capacity*pow((1-weight_non_junc),8)*pow((1-weight_junc),1)*ph_train_rate
        ph_out_train_out=(ph_inp_train_in/ph_train_rate)*(1-weight_junc)*ph_train_rate
        ph_out_walk_out=r_wo*(ph_inp_train_in-ph_out_train_out)
        ph_inp_walk_in=r_wi*(ph_inp_train_in-ph_out_train_out)*2
    else:
        ph_inp_train_in=ph_out_train_out=ph_out_walk_out=ph_inp_walk_in=0
    #for non peak hours
    if nph_train_rate!=0:
        nph_inp_train_in=capacity*pow((1-weight_non_junc),8)*pow((1-weight_junc),1)*nph_train_rate
        nph_out_train_out=(nph_inp_train_in/nph_train_rate)*(1-weight_junc)*nph_train_rate
        nph_out_walk_out=r_wo*(nph_inp_train_in-nph_out_train_out)
        nph_inp_walk_in=r_wi*(nph_inp_train_in-nph_out_train_out)*2
    else:
        nph_inp_train_in=nph_out_train_out=nph_out_walk_out=nph_inp_walk_in=0
    #combined 
    inp_train_in_pos_x=(ph_inp_train_in+nph_inp_train_in)/2
    out_train_out_pos_x=(ph_out_train_out+nph_out_train_out)/2
    out_walk_out_pos_x=(ph_out_walk_out+nph_out_walk_out)/2
    inp_walk_in_pos_x=(ph_inp_walk_in+nph_inp_walk_in)/2
    
    return inp_train_in_pos_x,out_train_out_pos_x,out_walk_out_pos_x,inp_walk_in_pos_x

#Material Balance on the station
def mass_balance(total_X_direction_values):
    inp_train_in,out_train_out,out_walk_out,inp_walk_in=total_X_direction_values
    inp=inp_train_in-inp_walk_in
    out=out_train_out-out_walk_out
    gen=cons=0
    acc=inp-out+gen-cons
    return acc



accumulation=[]
params=list(itertools.product(w,r))

#Grid Search to find optimum model parameters
from tqdm import tqdm
for p in tqdm(params):
    weight_junc,weight_non_junc=(p[0][0]/junc),(p[0][1]/non_junc)
    r_wo,r_wi=p[1][0],p[1][1]
    neg_X_direction_values=neg_X_direction(capacity,weight_junc,weight_non_junc,ph_train_rate,nph_train_rate,r_wo,r_wi)
    pos_X_direction_values=pos_X_direction(capacity,weight_junc,weight_non_junc,ph_train_rate,nph_train_rate,r_wo,r_wi)
    
    
    sum_list=[]
    for (item1, item2) in zip(neg_X_direction_values, pos_X_direction_values):
        sum_list.append(item1+item2)
    
    total_X_direction_values=sum_list
    
    
    
    accumulation.append(round(mass_balance(total_X_direction_values)))
    error=[(abs(a-actual_avg_footfall)/(actual_avg_footfall))*100 for a in accumulation]



optimum_params=params[(error.index(min(error)))]
print(optimum_params)


weight_junc,weight_non_junc=(optimum_params[0][0]/junc),(optimum_params[0][1]/non_junc)
r_wo,r_wi= (optimum_params[1][0]),(optimum_params[1][1]) 
print('The best parameters for model are: ',([weight_junc,weight_non_junc],round(min(error),2)))
print('Average hourly footfall for optimum parameters: ',accumulation[(error.index(min(error)))])
print('Total daily footfall for optimum parameters: ',accumulation[(error.index(min(error)))]*(ph+nph))


#Average footfall during peak hours using optimum parameters
neg_X_direction_values=neg_X_direction(capacity,weight_junc,weight_non_junc,ph_train_rate,0,r_wo,r_wi)
pos_X_direction_values=pos_X_direction(capacity,weight_junc,weight_non_junc,ph_train_rate,0,r_wo,r_wi)

sum_list=[]
for (item1, item2) in zip(neg_X_direction_values, pos_X_direction_values):
    sum_list.append(item1+item2)

total_X_direction_values=sum_list

accumulation_ph=round(mass_balance(total_X_direction_values))*2
print('Average hourly footfall during peak hours: ',accumulation_ph)


#Average footfall during non-peak hours using optimum parameters
neg_X_direction_values=neg_X_direction(capacity,weight_junc,weight_non_junc,0,nph_train_rate,r_wo,r_wi)
pos_X_direction_values=pos_X_direction(capacity,weight_junc,weight_non_junc,0,nph_train_rate,r_wo,r_wi)

sum_list=[]
for (item1, item2) in zip(neg_X_direction_values, pos_X_direction_values):
    sum_list.append(item1+item2)

total_X_direction_values=sum_list

accumulation_nph=round(mass_balance(total_X_direction_values))*2
print('Average hourly footfall during non-peak hours: ',accumulation_nph)

#jump in footfall from non-peak to peak hours
rise=((accumulation_ph-accumulation_nph)/(accumulation_nph))*100
print('Rise in average hourly footfall from non-peak to peak hours: %.2f%%'%rise)


#plot error function with 2 subplots (2nd subplot is zoomed in to see mimimum error)
plt.subplot(2,1,1)
plt.plot(error)
plt.ylabel('% Error',fontsize=12,fontname="Times New Roman")

plt.subplot(2,1,2)
plt.plot(error)
plt.ylim(0,10)
plt.xlabel('Parameter configuration number',fontsize=12,fontname="Times New Roman")
plt.ylabel('% Error',fontsize=12,fontname="Times New Roman")


