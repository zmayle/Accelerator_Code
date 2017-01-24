#phase_space.py
#Zachary Mayle
#2/24/16

"""Assigns the same random cavity error to each particle in a bunch. Many bunches go through the accelerator,
and cavity error is assigned to each.
(Lines 181 and 182 are key)"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np

class Particle(object):
    """An instance is a simulated particle in a simulated energy recovering linac.
    Each particle is defined by its energy and its phase (both with respect to the reference particle).
    
    Every particle needs to be able to gain and lose energy and move forward and backward in phase,
    depending upon when it enters the RF cavities relative to the RF cycle.
    
    Attributes:
        _energy [float>=0.0]: the current energy of the particle in MeV
        _phase [float]: the current phase of the particle in degrees"""
    
    
    def __init__(self,e,p):
        assert type(e)==float and type(p)==float
        assert e>=0
        self._energy=e
        self._phase=p
    
    
    def get_energy(self):
        return self._energy
    
    
    def get_phase(self):
        return self._phase
    
    
    def set_phase(self,p):
        self._phase=p
    
    
    def add_energy(self,e):
        """Adds the energy 'e' to the energy of the particle. Positive numbers increase energy,
        and negative numbers reduce energy."""
        assert type(e)==float
        self._energy+=e
    
    
    def add_phase(self,p):
        """Adds the phase 'p' to the phase of the particle". Positive numbers increase phase, and
        negative numbers reduce phase."""
        assert type(p)==float
        self._phase+=p


class Cluster(object):
    """A whole bunch of particles moving through an accelerator together. Defined by the particle objects
    in the cluster, the energy of the reference particle, and the synchronous phase (phase of the reference
    particle).
    
    Particles can gain and lose energy and move forward and backward in phase--each time this happens, the
    cluster list must be updated.
    
    Attributes:
        _refpart [particle object]: the reference particle which has the reference energy
            and the synchronous phase
        _sphase [float]: the synchronous phase (in degrees)
        _refenergy [float>=0.0]: the reference energy
        _pspread [float>=0.0]: the initial phase width, measured from the sphase
        _espread [float>=0.0]: the initial energy width, measured from the refenergy
        _bunch [list of n particle objects]: a list of all the particles in this cluster (there are a total of n)"""
    
    
    def __init__(self,re,sp,espread,pspread,n):
        assert type(re)==float and type(sp)==float and type(espread)==float and type(pspread)==float
        assert type(n)==int and n>=0
        assert re>=0 and espread>=0 and pspread>=0
        self._refpart=Particle(re,sp)
        self._refenergy=re
        self._sphase=sp
        self._espread=espread
        self._pspread=pspread
        self._bunch=self._make_bunch(n)
    
    
    def _make_bunch(self,n):
        """Generates a bunch, or a list of n particles that all have phases and energies within
        the specified initial spreads. The reference particle is the first in the list."""
        plist=[]
        plist.append(self._refpart)
        for i in range(n-1):
            e=random.gauss(self._refenergy,self._espread)
            p=random.gauss(self._sphase,self._pspread)
            #e=random.uniform(self._refenergy-self._espread,self._refenergy+self._espread)
            #p=random.uniform(self._sphase-self._pspread,self._sphase+self._pspread)
            part=Particle(e, p)
            plist.append(part)
        return plist
    
    def get_re(self):
        return self._refenergy
    
    def get_bunch(self):
        return self._bunch
    
    def set_bunch(self,bunch):
        self._bunch=bunch
    
    def set_sphase(self,sp):
        self._sphase=sp
    
    def plot_bunch(self,xmin=None,xmax=None,ymin=None,ymax=None):
        """Plots the bunch of particles on a scatterplot. Phase (relative
        to the synchronous phase) is the x-axis, and energy (relative to
        the reference energy) is the y-axis."""
        xdata=[]
        ydata=[]
        for i in self._bunch:
            x1=i.get_phase()-self._sphase
            y1=(i.get_energy()-self._refenergy)/self._refenergy
            xdata.append(x1)
            ydata.append(y1)
        plt.scatter(xdata,ydata,s=7,marker=".")
        plt.title('Phase Space')
        plt.ylabel('Energy (E-Eo)/Eo')
        plt.xlabel('Phase P-Po')
        plt.grid(True)
        if xmin!=None and ymin!=None and xmax!=None and ymax!=None:
            plt.axis([xmin,xmax,ymin,ymax])
        plt.show()
    
    
    def avg_energy(self):
        sum_all=0.0
        for i in self._bunch:
            sum_all += i.get_energy()
        avg = sum_all/len(self._bunch)
        return avg
    
    
    def rms_energy(self):
        sum_all = 0.0
        for i in self._bunch:
            sum_all += (i.get_energy()-self._refenergy)**2.0
        dErms = (sum_all/len(self._bunch))**(0.5)
        return dErms
    
    
    def rms_phase(self):
        sum_all = 0.0
        for i in self._bunch:
            sum_all += (i.get_phase()-self._sphase)**2.0
        dPhase = (sum_all/len(self._bunch))**(0.5)
        return dPhase
    
    
    def find_correlation(self):
        sum11 = 0.0
        sum12 = 0.0
        sum22 = 0.0
        for i in self._bunch:
            sum11 += (i.get_phase() - self._sphase)**2.0
        for j in self._bunch:
            sum12 += (j.get_phase() - self._sphase)*(j.get_energy()-self._refenergy)
        for k in self._bunch:
            sum22 += (k.get_energy()-self._refenergy)**2.0
        s11 = sum11/len(self._bunch)
        s12 = sum12/len(self._bunch)
        s22 = sum22/len(self._bunch)
        r12 = s12/((s22*s11)**(0.5))
        return r12
    
    
    def EStD(self): #gets standard deviation of Energy spread
        sum1 = 0.0
        sum2 = 0.0
        for i in self._bunch:
            sum1 += i.get_energy()
        Eavg = sum1/len(self._bunch)
        for j in self._bunch:
            sum2 += (j.get_energy())**2.0
        E2avg = sum2/len(self._bunch)
        EStD = (E2avg - Eavg**2.0)**0.5
        return EStD

    def pStD(self):  #gets standard deviation of phase spread
       sum1 = 0.0
       sum2 = 0.0
       for i in self._bunch:
           sum1 += i.get_phase()
       pavg = sum1/len(self._bunch)
       for j in self._bunch:
           sum2 += (j.get_phase())**2.0
       p2avg = sum2/len(self._bunch)
       pStD = (p2avg - pavg**2.0)**0.5
       return pStD
    
    
    def one_cycle(self,accel,j):
        """Passes the bunch through all the cavities one time and changes all of their
        energies and phases accordingly. The reference particle is unaffected by the
        errors in the cavities and is accelerated as if the cavites were ideal. For the
        rest of the particles, the cavity errors are taken into account. Only changes
        the phases of the particles if j is True.
        
        Parameters:
            accel is an Accelerator object
            j is a boolean"""
        #start with reference particle
        for i in accel.get_cavities():
            A=accel.get_refamp()
            sp=math.pi*self._refpart.get_phase()/180.0              #ref part is always at synch phase
            self._refpart.add_energy(A*math.cos(sp))   #ref part always gains design energy
            self._refenergy+=A*math.cos(sp)
        for x in self._bunch[1:]:
            for i in accel.get_cavities():
                ae=i.get_amperror()
                pe=i.get_pherror()
                A=accel.get_refamp()+ae
                p=math.pi*(pe+x.get_phase())/180.0
                E=A*math.cos(p)
                x.add_energy(E)
            if j==True:
                w=accel.get_wave()
                P=100.0*accel.get_r56()*((x.get_energy()-self._refenergy)/self._refenergy)    #CHECK THIS LINE!!!!
                #P=P+expansion_error
                P=P*360.0/w
                x.add_phase(P)
    
    
    def n_cycles(self,accel,n):
        i=1
        j=True
        while i<=n:
            if i==n:
                j=False
            self.one_cycle(accel,j)
            i+=1


    def recovery(self, accel, n):
        self._refpart.add_phase(180.0)
        self.set_sphase(self._sphase+180.0)
        for x in self._bunch[1:]:
            w=accel.get_wave()
            P=100.0*accel.get_r56()*((x.get_energy()-self._refenergy)/self._refenergy)    #CHECK THIS LINE!!!!
            #P=P+expansion_error
            P=P*360.0/w
            x.add_phase(P+180.0)
        self.n_cycles(accel,n)


class Cavity(object):
    """A cavity represents a superconducting cavity in which the bunch might
    gain some energy and shift in phase. A cavity is defined by the error in
    the cavity's amplitude and by the error in the cavity's phase (relative to
    the amplitude and phase of the intended cavity design).
    
    Attributes:
        _amperror [float]: the error (in MeV) of the cavity's energy-delivering
            amplitude, relative to the intended design amplitude
        _pherror [float]: the error (in degrees) of the cavity's phase, relative
            to the intended cavity oscillation"""
    
    def __init__(self,ae,pe):
        self._amperror=ae
        self._pherror=pe
    
    
    def get_amperror(self):
        return self._amperror
    
    
    def get_pherror(self):
        return self._pherror
    
    
class Accelerator(object):
    """Several cavities all lined up together to form a particle accelerating system.
    Basically a list of error-filled cavity objects with the intended design
    amplitude specified.
    
    Attributes:
        _cavities [cavity objects]: the energy-delivering cavities comprising this
            accelerator
        _refamp [float]: the intended design amplitude (in MeV) of all the cavities
            in this accelerator
        _r56 [float]: the value of r56 for this particular accelerator, measured
        in mm/% (determines the change in each particle's path length based on its energy)
        _wave [float]: the wavelength of E field oscillation in mm"""
    
    def __init__(self,n,amp,r56,wave):
        """n=number of cavities"""
        self._refamp=amp
        chain=[]
        for i in range(n):
            ae=random.uniform(-amp*0.001,amp*0.001)  #error for the amp is at most 10^-3 of reference value
            pe=random.uniform(-1.0,1.0)              #error for the phase is at most 1.0 degrees
            cav=Cavity(ae,pe)
            chain.append(cav)
        self._cavities=chain
        self._r56=r56
        self._wave=wave
    
    
    def get_cavities(self):
        return self._cavities
    
    
    def get_refamp(self):
        return self._refamp
    
    
    def get_r56(self):
        return self._r56
    
    
    def get_wave(self):
        return self._wave


def m_bunches(m,ncav,amp,r56,wave,re,sp,espread,pspread,npart,n):
    """Returns a cluster object containing particles from m different clusters that
    each went through num cycles in an accelerator.
    Generates m accelerators and m clusters and runs each bunch through its
    respective accelerator num times. Then adds all of the particles from all
    of the clusters to one list. Creates a new cluster object and sets the new list
    of particles as its bunch.
    
    Parameters:
        m is the number of bunches
        ncav is the number of cavities for each accelerator
        amp is the amplitude of each cavity
        r56 is the r56 value assigned to the accelerator
        wave is the wavelength of each cavity's oscillation
        re is the original reference energy of each cluster
        sp is the synchronous phase of each cluster
        espread is the initial energy spread of each cluster
        pspread is the initial phase spread of each cluster
        npart is the number of particles in each original cluster
        n is the number of cycles through each accelerator
        gain is the energy gain of one cycle"""
        
    list1=[]
    for i in range(m):
        accel=Accelerator(ncav,amp,r56,wave)
        bunch=Cluster(re,sp,espread,pspread,npart)
        bunch.n_cycles(accel,n)
        new_ref=bunch.get_re()
        phasespace=bunch.get_bunch()
        list1=list1+phasespace
    big_bunch=list1
    big_cluster=Cluster(new_ref,sp,espread,pspread,0)
    big_cluster.set_bunch(big_bunch)
    return big_cluster


def m_bunch_recovery(m,ncav,amp,r56,wave,re,sp,espread,pspread,npart,n):
    """This is the same as the m_bunches method, except the simulation includes the
    recovery step as well. The recovery step involves passing the cluster through the linac
    while deccelerating the bunch. The number of passes in the recovery step is the same
    as the number of passes in the accelerating step."""
    first_bunch=[]
    list1=[]
    bunchlist=[]
    accel_list=[]
    recovered_bunch=[]
    for j in range(m):
        bunch=Cluster(re,sp,espread,pspread,npart)
        initial=bunch.get_bunch()
        first_bunch=first_bunch+initial
        bunchlist.append(bunch)
    first_cluster=Cluster(re,sp,espread,pspread,0)
    first_cluster.set_bunch(first_bunch)
    first_cluster.plot_bunch()
    for x in bunchlist:
        accel=Accelerator(ncav,amp,r56,wave)
        x.n_cycles(accel,n)
        new_ref=x.get_re()
        phasespace=x.get_bunch()
        list1=list1+phasespace
        accel_list.append(accel)
    big_bunch=list1
    big_cluster=Cluster(new_ref,sp,espread,pspread,0)
    big_cluster.set_bunch(big_bunch)
    big_cluster.plot_bunch()
    x=0
    while x<len(bunchlist):
        bunchlist[x].recovery(accel_list[x],n)
        new_phasespace=bunchlist[x].get_bunch()
        recovered_bunch=recovered_bunch+new_phasespace
        final_ref=bunchlist[x].get_re()
        x+=1
    final_cluster=Cluster(final_ref,sp+180.0,espread,pspread,0)
    final_cluster.set_bunch(recovered_bunch)
    final_cluster.plot_bunch()
    return final_cluster


def m_bunch_quick_recovery(m,ncav,amp,r56,wave,re,sp,espread,pspread,npart,n):
    recovered_bunch=[]
    for x in range(m):
        bunch=Cluster(re,sp,espread,pspread,npart)
        accel=Accelerator(ncav,amp,r56,wave)
        bunch.n_cycles(accel,n)
        bunch.recovery(accel,n)
        new_phasespace=bunch.get_bunch()
        recovered_bunch=recovered_bunch+new_phasespace
        final_ref=bunch.get_re()
    final_cluster=Cluster(final_ref,sp+180.0,espread,pspread,0)
    final_cluster.set_bunch(recovered_bunch)
    return final_cluster


def scan(ncycle,cav,wave,cav_gain,ref_energy,espread,pspread,npart,n_bunches):
    """Returns a list containing the ideal value of r56, the ideal value of the
    synchronous phase, and the minimum rms energy. Also plots a contour of these values.
    Performs the simulation for m bunches: the accelerating step ONLY.
    Each bunch passes through its respective accelerator (each with unique errors).
    All the bunches are added to one bunch in the end, and the list returned contains
    the ideal values for this total bunch."""
    r56=-5.0        #determines left bound
    sp=0.0          #determines upper bound
    spindex=0.5
    r56index=0.25
    r=0
    s=0
    slist=[]
    rlist=[]
    elist=[]
    graph_width=5.0     #determines graph width
    graph_height=30.0   #determines graph height
    width=int(abs(graph_width/r56index)+1)
    height=int(abs(graph_height/spindex)+1)
    e_matrix = [[0 for x in range(width)] for y in range(height)]
    r_matrix = [[0 for x in range(width)] for y in range(height)]
    s_matrix = [[0 for x in range(width)] for y in range(height)]
    while r<width:
        while s<height:
            amp=cav_gain/math.cos(sp*math.pi/180.0)
            cluster=m_bunches(n_bunches,cav,amp,r56,wave,ref_energy,sp,espread,pspread,npart,ncycle)
            energy=cluster.rms_energy()/cluster.get_re()
            rlist.append(r56)
            slist.append(sp)
            elist.append(energy)
            e_matrix[s][r]=energy
            r_matrix[s][r]=r56
            s_matrix[s][r]=sp
            print "sp: "+str(sp)
            sp-=spindex
            s+=1
        print "r56: "+str(r56)
        r56+=r56index
        r+=1
        s=0
        sp=0.0
    #contour plot
    plt.figure()
    X=np.array(r_matrix)
    Y=np.array(s_matrix)
    Z=np.array(e_matrix)
    print X
    print Y
    print Z
    plt.contour(X, Y, Z, [.0001,.00025,.0005,.001,.005,.01,.05], colors = ['Yellow','Brown','Indigo','Green','Blue','Violet','Black'], linestyles = 'solid')
    plt.show()
    #return ideal values for r56, sp, and Erms
    least=min(elist)
    index=elist.index(least)
    r56least=rlist[index]
    spleast=slist[index]
    return [r56least,spleast,least]


def scan_recovery(ncycle,cav,wave,cav_gain,ref_energy,espread,pspread,npart,n_bunches):
    """Returns a list containing the ideal value of r56, the ideal value of the
    synchronous phase, and the minimum rms energy. Also plots a contour of these values.
    Performs the simulation for m bunches: the accelerating step AND the recovery step.
    Each bunch passes through its respective accelerator (each with unique errors). All the bunches are added to one bunch in the end,
    and the list returned contains the ideal values for this total bunch."""
    r56=-5.0
    sp=0.0
    spindex=0.5
    r56index=0.5
    r=0
    s=0
    slist=[]
    rlist=[]
    elist=[]
    width=int(abs(5.0/r56index)+1)
    height=int(abs(30.0/spindex)+1)
    e_matrix = [[0 for x in range(width)] for y in range(height)]
    r_matrix = [[0 for x in range(width)] for y in range(height)]
    s_matrix = [[0 for x in range(width)] for y in range(height)]
    while r56<=0.0:
        while sp>=-30.0:
            amp=cav_gain/math.cos(sp*math.pi/180.0)
            cluster=m_bunch_quick_recovery(n_bunches,cav,amp,r56,wave,ref_energy,sp,espread,pspread,npart,ncycle)
            energy=cluster.rms_energy()/cluster.get_re()    #change this line to get average energy
            rlist.append(r56)
            slist.append(sp)
            elist.append(energy)
            e_matrix[s][r]=energy
            r_matrix[s][r]=r56
            s_matrix[s][r]=sp
            print "sp: "+str(sp)
            sp-=spindex
            s+=1
        print "r56: "+str(r56)
        r56+=r56index
        r+=1
        s=0
        sp=0.0
    #contour plot
    plt.figure()
    X=np.array(r_matrix)
    Y=np.array(s_matrix)
    Z=np.array(e_matrix)
    print X
    print Y
    print Z
    plt.contour(X, Y, Z, [.0001,.0005,.001,.01,.1,1.0,10.0], colors = ['Orange','Red','Blue','Violet','Black','Green','Indigo'], linestyles = 'solid')
    plt.show()
    #return ideal values for r56, sp, and Erms
    least=min(elist)
    index=elist.index(least)
    r56least=rlist[index]
    spleast=slist[index]
    return [r56least,spleast,least]


if __name__ == '__main__':
    #set parameters
    sp=-8.0
    r56=-1.75
    
    #set constants and construct objects
    re=10.0
    espread=0.6*(0.001*re)
    #pspread=1.0
    pspread=.6
    npart=1000
    nclust=50
    number_cavities=8
    wave=100.0
    cycles=3
    E_gain=40.0
    cav_gain=E_gain/number_cavities
    amp=cav_gain/math.cos(sp*math.pi/180.0)
    ymin=-.001
    ymax=.001
    xmin=-4.0
    xmax=4.0
    
    #plot stuff, then go through n cycles, then plot stuff again
    clust=Cluster(re,sp,espread,pspread,npart)
    clust.plot_bunch()
    big_cluster=m_bunches(nclust,number_cavities,amp,r56,wave,re,sp,espread,pspread,npart,cycles)
    big_cluster.plot_bunch()
    
    #get important values
    ref_energy=big_cluster.get_re()
    Erms=big_cluster.rms_energy()
    Prms=big_cluster.rms_phase()
    correlation=big_cluster.find_correlation()
    Espread=Erms/ref_energy
    Estd=big_cluster.EStD()/ref_energy
    Eavg=big_cluster.avg_energy()
    
    #print important values
    print "Synchronous Phase: "+str(sp)+" degrees"
    print "r56: "+str(r56)
    print "Initial Reference Energy: "+str(re)+" MeV"
    print "Final Reference Energy: "+str(ref_energy)+" MeV"
    print "Energy RMS (measured from the reference energy): "+str(Erms)+" MeV"
    print "Phase RMS (measured from the synchronous phase): "+str(Prms)+" degrees"
    print "Correlation Parameter: "+str(correlation)
    print "Energy Spread: "+str(Espread)
    print "Average Energy: "+str(Eavg)
    #print "Energy Standard Dev: "+str(Estd)