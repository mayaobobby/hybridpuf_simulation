import picos as pcs
import numpy as np

def distinguish_probability(rho_0,rho_1):
    #Return the maximum success probability for distinguishing between rho_0 and rho_1
    #print(rho_0)
    #print(rho_1)
    sdp = pcs.Problem()
    d = rho_0.shape[1]
    I = pcs.Constant(np.diag([1. for i in range(d)]))
    M0 = pcs.HermitianVariable('M0', d)

    sdp.add_constraint(M0 >> 0)
    sdp.add_constraint(M0 << I)


    obj = 0.5+0.5*pcs.trace(M0*(rho_0-rho_1))
    sdp.set_objective('max',obj) 

    sol = sdp.solve(solver='cvxopt',verbosity=0)
    
    return float(obj.real)

def mutually_unbiased_bases_4():
    #Returns 5 mutually unbiased bases of dimension d = 4
    B = np.zeros((5,4,4),dtype = complex)
    
    B[0][0] =  np.array([1,0,0,0])
    B[0][1] =  np.array([0,1,0,0])
    B[0][2] =  np.array([0,0,1,0])
    B[0][3] =  np.array([0,0,0,1])

    B[1][0] = 0.5*np.array([1,1,1,1])
    B[1][1] = 0.5*np.array([1,1,-1,-1])
    B[1][2] = 0.5*np.array([1,-1,-1,1])
    B[1][3] = 0.5*np.array([1,-1,1,-1])

    B[2][0] = 0.5*np.array([1,-1,-1j,-1j])
    B[2][1] = 0.5*np.array([1,-1,1j,1j])
    B[2][2] = 0.5*np.array([1,1,1j,-1j])
    B[2][3] = 0.5*np.array([1,1,-1j,1j])

    B[3][0] = 0.5*np.array([1,-1j,-1j,-1])
    B[3][1] = 0.5*np.array([1,-1j,1j,1])
    B[3][2] = 0.5*np.array([1,1j,1j,-1])
    B[3][3] = 0.5*np.array([1,1j,-1j,1])

    B[4][0] = 0.5*np.array([1,-1j,-1,-1j])
    B[4][1] = 0.5*np.array([1,-1j,1,1j])
    B[4][2] = 0.5*np.array([1,1j,-1,1j])
    B[4][3] = 0.5*np.array([1,1j,1,-1j])

    for i in range(5):
        B[i] = B[i].transpose()

    return B
    

def mutually_unbiased_bases_8():
    #Returns 9 mutually unbiased bases of dimension d = 8
    id = np.diag([1. for i in range(8)])

    M = np.array([[[1.+0j,1.],[1.,-1.]],[[1.,1.],[1j,-1j]]])/np.sqrt(2)
    unitaries = [id, np.diag([1.,1.,1.,1.,1.,-1.,-1.,1.]),np.diag([1.,1.,1.,-1.,1.,-1.,1.,1.]),np.diag([1.,1.,1.,-1.,1.,1.,-1.,1.])]
    preMUB = np.zeros((8,8,8),dtype = complex)
    for i in range(8):
        preMUB[i] = np.kron(np.kron(M[int(i/4)],M[int((i%4)/2)]),M[i%2])
    MUB = np.zeros((9,8,8), dtype = complex)
    MUB[0] = id
    for i in range(8):
        MUB[i+1] = unitaries[min(i,7-i)]@preMUB[i]
           
    return MUB

def MUB_mixed_states(MUB):
    #Given bases matrices, returns mixed states for each state value
    K = MUB.shape[0]
    D = MUB.shape[1]
    assert(D == MUB.shape[2])

    rho = np.zeros((D,D,D),dtype = complex)

    for i in range(D):
        for j in range(K):
            rho[i]+= np.outer(MUB[j,:,i],MUB[j,:,i].conjugate())
    rho/= K
    
    return rho

def verify_MUB(MUB):
    #Check that the matrices obtained are mutually unbiased
    K = MUB.shape[0]
    D = MUB.shape[1]
    assert(D == MUB.shape[2])

    for i in range(K):
        print("Verifying basis",i)
        print(MUB[i].conjugate().transpose()@MUB[i]) #Should be identity
        for j in range(K):
            if(j == i):
                continue
            print(np.sqrt(D)*np.abs(MUB[i].conjugate().transpose()@MUB[j])) #SHould consist entirely of 1s

def mub_probabilities(d):
    #Returns successive bit guessing probabilities for 4 and 8 dimensional MUBs
    if(d == 4):
        #d = 4
        MUB = mutually_unbiased_bases_4()
        #verify_MUB(MUB)
        rho = MUB_mixed_states(MUB)
        p0 = distinguish_probability((rho[0]+rho[2])/2,(rho[1]+rho[3])/2) #Probability of guessing rightmost bit
        p1_0 = distinguish_probability(rho[0],rho[2]) #Prob of guessing left bit if right bit is 0
        p1_1 = distinguish_probability(rho[1],rho[3]) #Prob of guessing left bit if right bit is 1
        p1 = max(p1_0,p1_1) #Max prob of guessing left bit
        p = [p0,p1]
        return p
    
    elif(d==8):
        #d = 8
        MUB = mutually_unbiased_bases_8()
        #verify_MUB(MUB)
        rho = MUB_mixed_states(MUB)
        p0 = distinguish_probability((rho[0]+rho[1]+rho[2]+rho[3])/4,(rho[4]+rho[5]+rho[6]+rho[7])/4) #Probability of guessing leftmost bit
        p1_0 = distinguish_probability((rho[0]+rho[1])/2,(rho[2]+rho[3])/2) #Prob of guessing middle bit if left bit is 0
        p1_1 = distinguish_probability((rho[4]+rho[5])/2,(rho[6]+rho[7])/2) #Prob of guessing middle bit if left bit is 1
        p1 = max(p1_0,p1_1)  #Max prob of guessing middle bit
        p2_0 = distinguish_probability(rho[0],rho[1])
        p2_1 = distinguish_probability(rho[2],rho[3])
        p2_2 = distinguish_probability(rho[4],rho[5])
        p2_3 = distinguish_probability(rho[6],rho[7])
        p2 = max(p2_0,p2_1,p2_2,p2_3) #Max prob of guessing right bit
        p =  [p0,p1,p2]
        return p