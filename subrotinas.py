#import numpy as np
import numpy as np
from param import *
from matplotlib.legend import Legend


def salva_figura(fig,str_local,str_pasta,file_param):
  #str_local: local no drive
  #str_pasta: diretorio
  #file_param: Nome do arquivo de parâmetro com extensão .eps
  fig.savefig(file_param, format="eps", dpi=1200, bbox_inches="tight", transparent=True)
  dir=criar_dir(str_pasta)
  str_local=str_local+str_pasta
  str_param="cp "+file_param+" "+str_local+"/"+file_param  
  retorno=[print(dir),
  os.system(str_param)]
  return print(retorno)

def merge_files(str_f1,str_f2):
  # opening first file in append mode and second file in read mode
  f1 = open(str_f1, 'a+')
  f2 = open(str_f2, 'r')
  # appending the contents of the second file to the first file
  f1.write(f2.read()[1:])
  # closing the files
  f1.close()
  f2.close()

def adjust2merge(step_ref,str_file):
  f1=open('new_'+str_file,'w+')
  with open(str_file, 'r') as f:
      d = f.readlines()[1:]
      data = []
      j = 0;
      for i in d:
        k = i.rstrip().split(" ") # cada espaรงo divide a linha em duas colunas           
      
        linha = str(float(k[0])+step_ref)
        
        for n in k[1:]:
          linha=linha+" "+str(n)
        
        f1.write(linha+"\n")  
      j+=1
  f1.close()

def previsao(modelo,x,t,ylimq=None):
    #Plota os três estados em função do tempo desnormalizados juntamente com
    # o valor previsto pela rede
    #modelo a ser recuperado. Requer compilação prévia
    #x = dados reais
    #t = tempo dos dados reais
    x=np.array(x)
    if isinstance(modelo,str)==True:
        model.restore(modelo,verbose=1)
    else:
        print("Model argument ignored")
        
    yhat = model.predict(t)
    fig3=plt.figure()
    label = ['Pbh (bar)','Pwh (bar)','q (m3/s)'];
    for iu in range(0,3):
        ax3=fig3.add_subplot(3,1,iu+1)
        if iu==2:
            ax3.plot(tempo,((yhat[:,iu])*xc[2]+x0[2])*3600, '-b')
            ax3.plot(tempo,(x[2,:]*xc[2]+x0[2])*3600, ':k')
            #ax3.scatter(tempo,(x[2,:]*xc[2]+x0[2])*3600, 'r')
            #ax3.plot([1,nsim],[umin[iu], umin[iu]],'--r')
            #ax3.plot([1,nsim],[umax[iu], umax[iu]],'--r', label='Restrição')
            ax3.set_ylabel(label[iu])
            #ax3.set(xlim=(xi[0], nsim*ts))
            # if iu==0:
            #     #ax2.set(ylim=(30, 70))
            #     print(iu)
            plt.grid(True)
        elif iu==0:
            ax3.plot(tempo,((yhat[:,iu])*xc[0]+x0[0])/1e5, '-b')
            ax3.plot(tempo,(x[0,:]*xc[0]+x0[0])/1e5, ':k')
            #ax3.plot([1,nsim],[umin[iu], umin[iu]],'--r')
            #ax3.plot([1,nsim],[umax[iu], umax[iu]],'--r', label='Restrição')
            ax3.set_ylabel(label[iu])
            #plt.xticks([])
            #ax3.set(xlim=(xi[0], nsim*ts))
            # if iu==0:
            #     #ax2.set(ylim=(30, 70))
            #     print(iu)
            plt.grid(True)
            ax3.set_xticklabels([])
        else:
            ax3.plot(tempo,((yhat[:,iu])*xc[1]+x0[1])/1e5, '-b', label='Predicted')
            ax3.plot(tempo,(x[1,:]*xc[1]+x0[1])/1e5, ':k', label='Observed')
            if ylimq!=None:
                ax3.set(ylim=ylimq)
            #ax3.plot([1,nsim],[umin[iu], umin[iu]],'--r')
            #ax3.plot([1,nsim],[umax[iu], umax[iu]],'--r', label='Restrição')
            ax3.set_ylabel(label[iu])
            #ax3.set(xlim=(xi[0], nsim*ts))
            # if iu==0:
            #     #ax2.set(ylim=(30, 70))
            #     print(iu)
            plt.grid(True)
            ax3.set_xticklabels([])
            
            ax3.legend()
    return fig3


def criar_dir(nome_dir):
  string="mkdir "+loc_drive+nome_dir
  local_backup=loc_drive+nome_dir
  print(os.system(string))
  return local_backup

def salva_modelo_drive(str_local,str_pasta,str_nome,file_param):
  #str_local: local no drive
  #str_pasta: diretorio
  #str_nome: Nome do arquivo do modelo
  #file_param: Nome do arquivo de parâmetro
  dir=criar_dir(str_pasta)
  str_local=str_local+str_pasta
  str_data="cp "+"model_norm/modelo.ckpt-"+str(train_state.best_step)+".data-00000-of-00001 "+str_local+"/"+str_nome+".data-00000-of-00001"
  str_meta="cp "+"model_norm/modelo.ckpt-"+str(train_state.best_step)+".meta "+str_local+"/"+str_nome+".meta"
  str_index="cp "+"model_norm/modelo.ckpt-"+str(train_state.best_step)+".index "+str_local+"/"+str_nome+".index"
  str_param="cp "+file_param+" "+str_local+"/"+file_param
  str_loss="cp loss.dat "+str_local+"/loss.dat"
  str_test="cp test.dat "+str_local+"/test.dat"
  str_train="cp train.dat "+str_local+"/train.dat"
  
  retorno=[print(dir),
  os.system(str_data),
   os.system(str_meta),
   os.system(str_index),
   os.system(str_param),
   os.system(str_test),
   os.system(str_loss),
   os.system(str_train)]
  return print(retorno)

def Lim_c(x):
    return x[1]-x[0]
    
def ler_anchors(str_file):
  #read test.dat or train.dat file to get a specific dataset
    with open(str_file, 'r') as f:
        d = f.readlines()
        #t = np.zeros(len(d));      
        data=[]
        for i in d[1:]:
           data.append([float(i)])   
           
    return np.reshape(np.array(data),(len(data),1))

def ler_dados_rho_PI(str):
    with open(str, 'r') as f:
        d = f.readlines()
        epocas = np.zeros(len(d));
        param = np.zeros((len(d),2));
        j = 0;
        data = []
        for i in d:
            k = i.rstrip().split(" [") # cada espaรงo divide a linha em duas colunas 
            data.append([float(i) if is_float(i) else i for i in k])
            if k[0].isnumeric():
                epocas[j] = float(k[0])
                #print('k',k)
                #n = n.rstrip("]")
                k[1]=k[1].rstrip("]")
                aux=k[1].rstrip().split(", ")
                param[j,:]=np.array([float(aux[0]),float(aux[1])])
                #tau[j]=float(k[1])
            else:
                epocas[j]=epocas[j-1]
                param[j,:]=param[j-1,:]
                    
            j += 1;
    return param, epocas

def plot_rho_PI(str_file,valor_melhor,valor_final=None,r_lim=None,PI_lim=None,str_model=None):
    
    param,epocas=ler_dados_rho_PI(str_file)
    ind=[len(epocas)-1]
    if valor_final != None:
        ind = np.where(epocas == valor_final)
    #print(epocas)
    ind_best = np.where(epocas == valor_melhor)
    #np.where(epocas == valor_melhor)
    #print(ind_best)
    #Converter PI de m3/s/Pa => m3/h/bar
    conv=3.6e8;
    param[:,1]=param[:,1]*conv
    rho_ref=950
    PI_ref=2.32e-9*conv
    conv=1
    #print("Melhor Epoca",epocas[ind_best[0]],int(ind_best[0]))
    if str_model=='LEA':
        rho_ref=836.8898
        PI_ref=2.7e-8*conv
    print(r"Melhor $\rho$",param[ind_best,0]," Referência: ",rho_ref,"Erro: ",100*abs(1-param[ind_best,0]/rho_ref),"%")
    print("Melhor PI",param[ind_best,1]," Referência: ",PI_ref,"Erro: ",100*abs(1-param[ind_best,1]/PI_ref),"%")
    StepAfter=0 # in order to show steps after best epoch
    label=[r"Estimated $\rho$", 'Estimated PI', r"True $\rho$", "True PI"]
    
    fig=plt.figure()
    ax=fig.add_subplot()
    ln1=ax.plot(epocas,param[:,0],color='red', label=label[0])
    ln3=ax.plot(epocas,np.ones_like(epocas)*rho_ref,'--',color='red',label=label[2])
    ln5=ax.plot(epocas[ind_best],param[ind_best,0],'k',marker='x',label=r'Best $\rho$')
    if r_lim!=None:
        ax.set(ylim=(r_lim[0], r_lim[1]))
    
    ax2=ax.twinx()
    ax2.set_ylabel(r'$PI~[m^3/h/bar]$')
   
    ln2=ax2.plot(epocas,param[:,1],color='blue', label=label[1])
    ln4=ax2.plot(epocas,np.ones_like(epocas)*PI_ref,'--',color='blue', label=label[3])
    ln6=ax2.plot(epocas[ind_best],param[ind_best,1],'k',marker='x' ,label='Best PI')
    if PI_lim!=None:
        ax2.set(ylim=(PI_lim[0], PI_lim[1]))
    # added these three lines
    # ln = ln1+ln2#+ln2+ln3
    
    # labs = [l.get_label() for l in ln]
    #ax2.legend(ln, labs, loc='lower right')#,ncol=2)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles2, labels2, loc='lower right',frameon=False)#,ncol=2)
    leg = Legend(ax2, handles, labels,loc='lower center', frameon=False)


    ax2.add_artist(leg)
    ax.set_ylabel(r"$\rho ~[kg/m^3]$")
    ax.set_xlabel('steps')
    plt.grid(True)
    #plt.show()
    return fig


def APRBS(a_range,b_range,nstep):
    # random signal generation
    a = np.random.rand(nstep) * (a_range[1]-a_range[0]) + a_range[0] # range for amplitude
    b = np.random.rand(nstep) *(b_range[1]-b_range[0]) + b_range[0] # range for frequency
    b = np.round(b)
    b = b.astype(int)

    b[0] = 0

    for i in range(1,np.size(b)):
        b[i] = b[i-1]+b[i]

    # Random Signal
    i=0
    random_signal = np.zeros(nstep)
    while b[i]<np.size(random_signal):
        k = b[i]
        random_signal[k:] = a[i]
        i=i+1

    # PRBS
    a = np.zeros(nstep)
    j = 0
    while j < nstep:
        a[j] = 5
        a[j+1] = -5
        j = j+2

    i=0
    prbs = np.zeros(nstep)
    while b[i]<np.size(prbs):
        k = b[i]
        prbs[k:] = a[i]
        i=i+1
    return [random_signal, prbs]

def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False
def GetTrainningPoints(str_file):
    lines = 0
    with open(str_file) as f:
        for line in f:
            lines = lines + 1
    return lines

def ler_dados(str):
    with open(str, 'r') as f:
        d = f.readlines()
        epocas = np.zeros(len(d));
        tau = np.zeros(len(d));
        j = 0;
        data = []
        for i in d:
            k = i.rstrip().split(" ") # cada espaรงo divide a linha em duas colunas 
            data.append([float(i) if is_float(i) else i for i in k])
            epocas[j] = float(k[0])
            n = k[1].lstrip("[")
            n = n.rstrip("]")
            tau[j]=float(n)
            #tau[j]=float(k[1])
            j += 1;
    return tau, epocas
    
def ler_anchors(str):
    with open(str, 'r') as f:
        d = f.readlines()
         
       
        # for i in d[1:]:
        #     # k = i.rstrip().split(" ") # cada espaรงo divide a linha em duas colunas 
        #     # data.append([float(i) if is_float(i) else i for i in k])
        #     # epocas[j] = float(k[0])
        #     # n = k[1].lstrip("[")
        #     # n = n.rstrip("]")
        #     tau[i]=float(d[i])
        #     #tau[j]=float(k[1])
    t = np.zeros(len(d[1:]));
    for i in np.arange(1,len(d)-1):
        t[i]=float(d[i])       
    return t
    
    
def Fig_data(t,x):

  fig3=plt.figure()
  label = ['Pbh (bar)','Pwh (bar)','q(m3/s)'];
  xi=t
  for iu in range(0,5):
      ax3=fig3.add_subplot(5,1,iu+1)
      if iu==2:
          ax3.plot(xi,x[2], label='Medição')
          #ax3.plot([1,nsim],[umin[iu], umin[iu]],'--r')
          #ax3.plot([1,nsim],[umax[iu], umax[iu]],'--r', label='Restrição')
          ax3.set_ylabel(label[iu])
          #ax3.set(xlim=(xi[0], nsim*ts))
          # if iu==0:
          #     #ax2.set(ylim=(30, 70))
          #     print(iu)
          plt.grid(True)
      elif iu==1:
          ax3.plot(xi,x[1], label='Medição')
          #ax3.plot([1,nsim],[umin[iu], umin[iu]],'--r')
          #ax3.plot([1,nsim],[umax[iu], umax[iu]],'--r', label='Restrição')
          ax3.set_ylabel(label[iu])
          #ax3.set(xlim=(xi[0], nsim*ts))
          # if iu==0:
          #     #ax2.set(ylim=(30, 70))
          #     print(iu)
          plt.grid(True)
      elif iu==0:
          ax3.plot(xi,x[0], label='Medição')
          #ax3.plot([1,nsim],[umin[iu], umin[iu]],'--r')
          #ax3.plot([1,nsim],[umax[iu], umax[iu]],'--r', label='Restrição')
          ax3.set_ylabel(label[iu])
          #ax3.set(xlim=(xi[0], nsim*ts))
          # if iu==0:
          #     #ax2.set(ylim=(30, 70))
          #     print(iu)
          plt.grid(True)
      elif iu==3:
          ax3.plot(xi,ex_func(xi)[0], label='Medição')
          #ax3.plot([1,nsim],[umin[iu], umin[iu]],'--r')
          #ax3.plot([1,nsim],[umax[iu], umax[iu]],'--r', label='Restrição')
          ax3.set_ylabel('f (Hz)')
          #ax3.set(xlim=(xi[0], nsim*ts))
          # if iu==0:
          #     #ax2.set(ylim=(30, 70))
          #     print(iu)
          plt.grid(True)
      else:
          ax3.plot(xi,ex_func(xi)[1], label='Medição')
          #ax3.plot([1,nsim],[umin[iu], umin[iu]],'--r')
          #ax3.plot([1,nsim],[umax[iu], umax[iu]],'--r', label='Restrição')
          ax3.set_ylabel('zc(%)')
          #ax3.set(xlim=(xi[0], nsim*ts))
          # if iu==0:
          #     #ax2.set(ylim=(30, 70))
          #     print(iu)
          plt.grid(True)
          
          

def Evol_Parametro(str_file_var):
  parametro,epocas=ler_dados(str_file_var)
  fig=plt.figure()
  ax2=fig.add_subplot()
  ax2.plot(epocas,parametro)
  ax2.set_ylabel('rho')
  return fig
# def RAR_Multi(points_per_line,erro)
#     nx=3 #numero de saídas (estados)
#         for i in np.arange(0,nx):
#         ind = np.argpartition(erro[i,:], -3,0)[-3:]
#         #print('Maiores linha:',i,'\n ',ind,':\n Valores:',err_eq[i,ind[0]],err_eq[i,ind[1]])
#         for j in range(0,len(ind)):
#             #print('add indice:',ind[j])
            
#             data.add_anchors(X[ind[j]])