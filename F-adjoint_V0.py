import numpy as np
# The activation function and it's derivative
sigma = lambda x: 1/(1+np.exp(-x))
d_sigma= lambda x: sigma(x)*(1-sigma(x))

# The F-propagation function
def F_prop(X, W):
    L=len(W)
    X=X=np.append(X, np.ones((X.shape[0],1)), axis = 1)
    F_X = {'X0':X.T}
    for h in range(1, L+1):
        F_X[f'Y{h}']=np.dot(W[f'W{h}'], F_X[f'X{h-1}'])
        if h!=L:
            F_X[f'X{h}']=np.concatenate((sigma(F_X[f'Y{h}']), np.ones((1,X.shape[0]))), axis = 0)
        else:
            F_X[f'X{h}']=sigma(F_X[f'Y{h}'])
    return F_X

# The F-adjoint propagation function
def F_star_prop(X,y, W):
    L=len(W)
    F_X = F_prop(X, W)
    FX_star={f'X*{L}':(F_X[f'X{L}']-y)}
    Y_star={}
    for h in reversed(range(1, L+1)):
        FX_star[f'Y*{h}']=FX_star[f'X*{h}']*(d_sigma(F_X[f'Y{h}']))
        FX_star[f'X*{h-1}']=np.delete(W[f'W{h}'], -1, axis = 1).T.dot(FX_star[f'Y*{h}'])
    return FX_star

# Update the weights layer-wise by using the F-adjoint propagation: local-learnig approach
def F_star_W(X,y, W,eta1=0.5,eta2=0.5):
    L=len(W)
    F_X = F_prop(X, W)
    FX_star={'X*2':(F_X['X2']-y)}
    Y_star={}
    ## Update of the layer 2 with learning rate eta2
    FX_star['Y*2']=FX_star['X*2']*(d_sigma(F_X['Y2']))
    W['W2']= W['W2'] -eta2*FX_star['Y*2'].dot(F_X['X1'].T)
    ## Update of the layer 1 with learning rate eta1
    FX_star['X*1']=np.delete(W['W2'], -1, axis = 1).T.dot(FX_star['Y*2'])
    FX_star['Y*1']=FX_star['X*1']*(d_sigma(F_X['Y1']))
    W['W1']= W['W1'] -eta1*FX_star['Y*1'].dot(F_X['X0'].T)
    return W

# Update the weights globally by using the F-adjoint propagation: nonlocal-learnig approach
def Grad_star(X,y, W, eta=0.5):
    L=len(W)
    F_X = F_prop(X, W)
    FX_star=F_star_prop(X,y, W)
    Grad={}
    for h in range(1, L+1):
        Grad[f'W{h}']=W[f'W{h}']-eta*FX_star[f'Y*{h}'].dot(F_X[f'X{h-1}'].T)
    return Grad

if __name__ == "__main__":

    # Learning the XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    # X = np.array([[0, 0]]); y=np.array([[1]])
    W1 = np.array([[1, -1, 1], [-1, 1, -1]])
    W2 = np.array([[1, -1, 1]])
    W = {'W1': W1, 'W2': W2}
    import pandas as pd

    np.set_printoptions(precision=3)
    df1 = pd.DataFrame([F_prop(X, W)])
    for row_dict in df1.to_dict(orient='records'):
        print(row_dict)
    df2 = pd.DataFrame([F_star_prop(X,y, W)])
    for row_dict in df2.to_dict(orient='records'):
        print(row_dict)

    print('Learning the XOR dataset with layer-wise function: F_star_W(X,y, W)')
    print('===================================================================')
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    # X = np.array([[0, 0]]); y=np.array([[1]])
    W1 = np.array([[1, -1, 1], [-1, 1, -1]])
    W2 = np.array([[1, -1, 1]])
    W = {'W1': W1, 'W2': W2}
    ##
    X2 = F_prop(X, W)['X2']
    ##
    X2_c = np.where(X2 >= 0.5, 1, 0)
    k = 0
    while (X2_c - y).any() != 0:
        # for k in range(100):
        # print('X',X)
        X2 = F_prop(X, W)['X2']
        X2_c = np.where(X2 >= 0.5, 1, 0)
        FX_star = F_star_prop(X, y, W)

        print('X2', X2, k + 1)
        W = F_star_W(X, y, W)
        # W=Grad_star(X,y, W)
        X = X - FX_star['X*0'].T
        k += 1
        print("================================")

    print('Learning the XOR dataset with global gradient function:Grad_star(X,y, W)')
    print("=========================================================================")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    # X = np.array([[0, 0]]); y=np.array([[1]])*np.sqrt(6)
    W1 = np.array([[1, -1, 1], [-1, 1, 0]]) * (20)
    W2 = np.array([[1, -1, 1]])
    W = {'W1': W1, 'W2': W2}
    ##
    X2 = F_prop(X, W)['X2']
    ##
    X2_c = np.where(X2 >= 0.5, 1, 0)
    k = 0
    while (X2_c - y).any() != 0:
        # for k in range(10):
        # print('X',X)
        X2 = F_prop(X, W)['X2']
        X2_c = np.where(X2 >= 0.5, 1, 0)
        FX_star = F_star_prop(X, y, W)

        print('X2=', X2, k + 1)
        # W=F_star_W(X,y, W)
        W = Grad_star(X, y, W)
        X = X - FX_star['X*0'].T  ##(X0-Xstar0) as input
        k += 1
        print("================================")







